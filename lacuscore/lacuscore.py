#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import re
import sys
import time
import unicodedata
import zlib

from asyncio import Task
from base64 import b64decode, b64encode
from datetime import date, timedelta
from ipaddress import ip_address, IPv4Address, IPv6Address
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Any, overload, cast, Iterator
from uuid import uuid4
from urllib.parse import urlsplit

from dns import resolver
from dns.exception import DNSException
from dns.exception import Timeout as DNSTimeout

from defang import refang  # type: ignore[import-untyped]
from playwrightcapture import Capture, PlaywrightCaptureException
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import DataError
from ua_parser import user_agent_parser  # type: ignore[import-untyped]

from . import task_logger
from .helpers import (
    LacusCoreLogAdapter, CaptureError, RetryCapture, CaptureSettingsError,
    CaptureStatus, CaptureResponse, CaptureResponseJson, CaptureSettings)

if sys.version_info < (3, 11):
    from async_timeout import timeout

    def timeout_expired(timeout_cm, logger, error_message: str) -> None:  # type: ignore[no-untyped-def]
        if timeout_cm.expired:
            logger.warning(f'Timeout expired: {error_message}')

else:
    from asyncio import timeout

    def timeout_expired(timeout_cm, logger, error_message: str) -> None:  # type: ignore[no-untyped-def]
        if timeout_cm.expired():
            logger.warning(f'Timeout expired: {error_message}')


BROWSER = Literal['chromium', 'firefox', 'webkit']


def _secure_filename(filename: str) -> str:
    """Copy of secure_filename in werkzeug, to avoid the dependency.
    Source: https://github.com/pallets/werkzeug/blob/d36aaf12b5d12634844e4c7f5dab4a8282688e12/src/werkzeug/utils.py#L197
    """
    _filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.-]")
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.path.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(_filename_ascii_strip_re.sub("", "_".join(filename.split()))).strip(
        "._"
    )

    return filename


class LacusCore():
    """Capture URLs or web enabled documents using PlaywrightCapture.

    :param redis_connector: Pre-configured connector to a redis instance.
    :param max_capture time: If the capture takes more than that time, break (in seconds)
    :param tor_proxy: URL to a SOCKS 5 tor proxy. If you have tor installed, this is the default: socks5://127.0.0.1:9050.
    :param only_global_lookups: Discard captures that point to non-public IPs.
    :param max_retries: How many times should we re-try a capture if it failed.
    """

    def __init__(self, redis_connector: Redis, /, *,  # type: ignore[type-arg]
                 max_capture_time: int=3600,
                 tor_proxy: str | None=None,
                 only_global_lookups: bool=True,
                 max_retries: int=3,
                 loglevel: str | int='INFO') -> None:
        self.master_logger = logging.getLogger(f'{self.__class__.__name__}')
        self.master_logger.setLevel(loglevel)
        self.redis = redis_connector
        self.max_capture_time = max_capture_time
        self.tor_proxy = tor_proxy
        self.only_global_lookups = only_global_lookups
        self.max_retries = max_retries
        self.dnsresolver: resolver.Resolver = resolver.Resolver()
        self.dnsresolver.cache = resolver.Cache(30)
        self.dnsresolver.timeout = 2
        self.dnsresolver.lifetime = 3

        # NOTE: Remove in 1.8.* - clear old ongoing captures queue in case of need
        if self.redis.type('lacus:ongoing') in ['set', b'set']:  # type: ignore[no-untyped-call]
            self.redis.delete('lacus:ongoing')

    def check_redis_up(self) -> bool:
        """Check if redis is reachable"""
        return bool(self.redis.ping())

    @overload
    def enqueue(self, *, settings: CaptureSettings | None=None) -> str:
        ...

    @overload
    def enqueue(self, *,
                url: str | None=None,
                document_name: str | None=None, document: str | None=None,
                depth: int=0,
                browser: BROWSER | None=None, device_name: str | None=None,
                user_agent: str | None=None,
                proxy: str | dict[str, str] | None=None,
                general_timeout_in_sec: int | None=None,
                cookies: list[dict[str, Any]] | None=None,
                headers: str | dict[str, str] | None=None,
                http_credentials: dict[str, str] | None=None,
                geolocation: dict[str, float] | None=None,
                timezone_id: str | None=None,
                locale: str | None=None,
                color_scheme: str | None=None,
                viewport: dict[str, int] | None=None,
                referer: str | None=None,
                rendered_hostname_only: bool=True,
                with_favicon: bool=False,
                allow_tracking: bool=False,
                force: bool=False,
                recapture_interval: int=300,
                priority: int=0,
                uuid: str | None=None
                ) -> str:
        ...

    def enqueue(self, *,
                settings: CaptureSettings | None=None,
                url: str | None=None,
                document_name: str | None=None, document: str | None=None,
                depth: int=0,
                browser: BROWSER | None=None, device_name: str | None=None,
                user_agent: str | None=None,
                proxy: str | dict[str, str] | None=None,
                general_timeout_in_sec: int | None=None,
                cookies: list[dict[str, Any]] | None=None,
                headers: str | dict[str, str] | None=None,
                http_credentials: dict[str, str] | None=None,
                geolocation: dict[str, float] | None=None,
                timezone_id: str | None=None,
                locale: str | None=None,
                color_scheme: str | None=None,
                viewport: dict[str, int] | None=None,
                referer: str | None=None,
                rendered_hostname_only: bool=True,
                with_favicon: bool=False,
                allow_tracking: bool=False,
                force: bool=False,
                recapture_interval: int=300,
                priority: int=0,
                uuid: str | None=None
                ) -> str:
        """Enqueue settings.

        :param settings: Settings as a dictionary

        :param url: URL to capture (incompatible with document and document_name)
        :param document_name: Filename of the document to capture (required if document is used)
        :param document: Document to capture itself (requires a document_name)
        :param depth: [Dangerous] Depth of the capture. If > 0, the URLs of the rendered document will be extracted and captured. It can take a very long time.
        :param browser: The prowser to use for the capture
        :param device_name: The name of the device, must be something Playwright knows
        :param user_agent: The user agent the browser will use for the capture
        :param proxy: SOCKS5 proxy to use for capturing
        :param general_timeout_in_sec: The capture will raise a timeout it it takes more than that time
        :param cookies: A list of cookies
        :param headers: The headers to pass to the capture
        :param http_credentials: HTTP Credentials to pass to the capture
        :param geolocation: Geolocation of the browser to pass to the capture
        :param timezone_id: The timezone of the browser to pass to the capture
        :param locale: The locale of the browser to pass to the capture
        :param color_scheme: The prefered color scheme of the browser to pass to the capture
        :param viewport: The viewport of the browser used for capturing
        :param referer: The referer URL for the capture
        :param rendered_hostname_only: If depth > 0: only capture URLs with the same hostname as the rendered page
        :param with_favicon: If True, PlaywrightCapture will attempt to get the potential favicons for the rendered URL. It is a dirty trick, see this issue for details: https://github.com/Lookyloo/PlaywrightCapture/issues/45
        :param allow_tracking: If True, PlaywrightCapture will attempt to click through the cookie banners. It is totally dependent on the framework used on the website.
        :param force: Force recapture, even if the same one was already done within the recapture_interval
        :param recapture_interval: The time the enqueued settings are kept in memory to avoid duplicates
        :param priority: The priority of the capture
        :param uuid: The preset priority of the capture, auto-generated if not present. Should only be used if the initiator couldn't enqueue immediately. NOTE: it will be overwritten if the UUID already exists.

        :return: UUID, reference to the capture for later use
        """
        to_enqueue: CaptureSettings
        if settings:
            if 'url' in settings and settings['url'] is not None:
                settings['url'] = settings['url'].strip()
            if settings.get('force') is not None:
                force = settings.pop('force', False)
            if settings.get('recapture_interval') is not None:
                recapture_interval = settings.pop('recapture_interval', 300)
            if settings.get('priority') is not None:
                priority = settings.pop('priority', 0)
            to_enqueue = settings
        else:
            to_enqueue = {'depth': depth, 'rendered_hostname_only': rendered_hostname_only}
            if url:
                to_enqueue['url'] = url.strip()
            elif document_name and document:
                to_enqueue['document_name'] = _secure_filename(document_name)
                to_enqueue['document'] = document
            if browser:
                to_enqueue['browser'] = browser
            if device_name:
                to_enqueue['device_name'] = device_name
            if user_agent:
                to_enqueue['user_agent'] = user_agent
            if proxy:
                to_enqueue['proxy'] = proxy
            if general_timeout_in_sec is not None:  # that would be a terrible idea, but this one could be 0
                to_enqueue['general_timeout_in_sec'] = general_timeout_in_sec
            if cookies:
                to_enqueue['cookies'] = cookies
            if headers:
                to_enqueue['headers'] = headers
            if http_credentials:
                to_enqueue['http_credentials'] = http_credentials
            if geolocation:
                to_enqueue['geolocation'] = geolocation
            if timezone_id:
                to_enqueue['timezone_id'] = timezone_id
            if locale:
                to_enqueue['locale'] = locale
            if color_scheme:
                to_enqueue['color_scheme'] = color_scheme
            if viewport:
                to_enqueue['viewport'] = viewport
            if referer:
                to_enqueue['referer'] = referer
            if with_favicon:
                to_enqueue['with_favicon'] = with_favicon
            if allow_tracking:
                to_enqueue['allow_tracking'] = allow_tracking

        hash_query = hashlib.sha512(pickle.dumps(to_enqueue)).hexdigest()
        if not force:
            if (existing_uuid := self.redis.get(f'lacus:query_hash:{hash_query}')):
                if isinstance(existing_uuid, bytes):
                    return existing_uuid.decode()
                return existing_uuid

        if uuid:
            # Make sure we do not already have a capture with that UUID
            if self.get_capture_status(uuid) == CaptureStatus.UNKNOWN:
                perma_uuid = uuid
            else:
                perma_uuid = str(uuid4())
                self.master_logger.warning(f'UUID {uuid} already exists, forcing a new one: {perma_uuid}.')
        else:
            perma_uuid = str(uuid4())

        mapping_capture: dict[str, bytes | float | int | str] = {}
        for key, value in to_enqueue.items():
            if value is None:
                continue
            if isinstance(value, bool):
                mapping_capture[key] = 1 if value else 0
            elif isinstance(value, (list, dict)):
                if value:
                    mapping_capture[key] = json.dumps(value)
            elif isinstance(value, (bytes, float, int, str)) and value not in ['', b'']:  # we're ok with 0 for example
                mapping_capture[key] = value

        p = self.redis.pipeline()
        p.set(f'lacus:query_hash:{hash_query}', perma_uuid, nx=True, ex=recapture_interval)
        p.hset(f'lacus:capture_settings:{perma_uuid}', mapping=mapping_capture)  # type: ignore[arg-type]
        p.zadd('lacus:to_capture', {perma_uuid: priority if priority is not None else 0})
        try:
            p.execute()
        except DataError:
            self.master_logger.exception(f'Unable to enqueue: {to_enqueue}')
            raise CaptureSettingsError(f'Unable to enqueue: {to_enqueue}')
        return perma_uuid

    def _encode_response(self, capture: CaptureResponse) -> CaptureResponseJson:
        encoded_capture = cast(CaptureResponseJson, capture)
        if capture.get('png') is not None and capture['png'] is not None:  # the second part is not needed, but makes mypy happy
            encoded_capture['png'] = b64encode(capture['png']).decode()
        if capture.get('downloaded_file') is not None and capture['downloaded_file'] is not None:  # the second part is not needed, but makes mypy happy
            encoded_capture['downloaded_file'] = b64encode(capture['downloaded_file']).decode()
        if capture.get('children') and capture['children']:
            encoded_capture['children'] = [self._encode_response(child) for child in capture['children']]

        # A set cannot be dumped in json, it must be turned into a list. If it is empty, we need to remove it.
        if 'potential_favicons' in capture:
            if potential_favicons := capture.pop('potential_favicons'):
                encoded_capture['potential_favicons'] = [b64encode(favicon).decode() for favicon in potential_favicons]
        return encoded_capture

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[True]=True) -> CaptureResponse:
        ...

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[False]) -> CaptureResponseJson:
        ...

    def get_capture(self, uuid: str, *, decode: bool=False) -> CaptureResponse | CaptureResponseJson:
        """Get the results of a capture, in a json compatible format or not

        :param uuid: The UUID if the capture (given by enqueue)
        :param decode: Decode the capture result or not.

        :return: The capture, decoded or not.
        """
        to_return: CaptureResponse = {'status': CaptureStatus.UNKNOWN}
        if self.redis.zscore('lacus:to_capture', uuid):
            to_return['status'] = CaptureStatus.QUEUED
        elif self.redis.zscore('lacus:ongoing', uuid) is not None:
            to_return['status'] = CaptureStatus.ONGOING
        elif response := self._get_capture_response(uuid):
            to_return['status'] = CaptureStatus.DONE
            to_return.update(response)
            if decode:
                return to_return
            return self._encode_response(to_return)
        return to_return

    def get_capture_status(self, uuid: str) -> CaptureStatus:
        """Get the status of a capture

        :param uuid: The UUID if the capture (given by enqueue)

        :return: The status
        """
        if self.redis.zscore('lacus:to_capture', uuid) is not None:
            return CaptureStatus.QUEUED
        elif self.redis.zscore('lacus:ongoing', uuid) is not None:
            return CaptureStatus.ONGOING
        elif self.redis.exists(f'lacus:capture_settings:{uuid}'):
            # we might have popped the UUID out of lacus:to_capture
            # but not pused it in lacus:ongoing yet
            return CaptureStatus.QUEUED
        elif self.redis.exists(f'lacus:capture_results_hash:{uuid}'):
            return CaptureStatus.DONE
        elif self.redis.exists(f'lacus:capture_results:{uuid}'):
            # TODO: remove in 1.8.* - old format used last in 1.6, and kept no more than 10H in redis
            return CaptureStatus.DONE
        return CaptureStatus.UNKNOWN

    def consume_queue(self, max_consume: int) -> Iterator[Task]:  # type: ignore[type-arg]
        """Trigger the capture for captures with the highest priority. Up to max_consume.

        :yield: Captures.
        """
        value: list[tuple[bytes, float]]
        while max_consume > 0:
            value = self.redis.zpopmax('lacus:to_capture')
            if not value:
                # Nothing to capture
                break
            if not value[0]:
                continue
            max_consume -= 1
            uuid: str = value[0][0].decode()
            priority: int = int(value[0][1])
            logger = LacusCoreLogAdapter(self.master_logger, {'uuid': uuid})
            yield task_logger.create_task(self._capture(uuid, priority), name=uuid,
                                          logger=logger,
                                          message='Capture raised an uncaught exception')

    async def _capture(self, uuid: str, priority: int) -> None:
        """Trigger a specific capture

        :param uuid: The UUID if the capture (given by enqueue)
        :param priority: Only for internal use, will decide on the priority of the capture if the try now fails.
        """
        if self.redis.zscore('lacus:ongoing', uuid) is not None:
            # the capture is already ongoing
            await asyncio.sleep(1)
            return

        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': uuid})
        self.redis.zadd('lacus:ongoing', {uuid: time.time()})
        stats_pipeline = self.redis.pipeline()
        today = date.today().isoformat()

        retry = False
        try:
            setting_keys = ['depth', 'rendered_hostname_only', 'url', 'document_name',
                            'document', 'browser', 'device_name', 'user_agent', 'proxy',
                            'general_timeout_in_sec', 'cookies', 'headers', 'http_credentials',
                            'viewport', 'referer', 'geolocation', 'timezone_id', 'locale',
                            'color_scheme', 'with_favicon', 'allow_tracking']
            result: CaptureResponse = {}
            to_capture: CaptureSettings = {}
            document_as_bytes = b''
            url: str = ''
            try:
                for k, v in zip(setting_keys, self.redis.hmget(f'lacus:capture_settings:{uuid}', setting_keys)):
                    if v is None:
                        continue
                    if k in ['url', 'document_name', 'browser', 'device_name', 'user_agent',
                             'referer', 'timezone_id', 'locale', 'color_scheme']:
                        # string
                        to_capture[k] = v.decode()  # type: ignore[literal-required]
                    elif k in ['cookies', 'http_credentials', 'viewport', 'geolocation']:
                        # dicts or list
                        to_capture[k] = json.loads(v)  # type: ignore[literal-required]
                    elif k in ['proxy', 'headers']:
                        # can be dict or str
                        try:
                            to_capture[k] = json.loads(v)  # type: ignore[literal-required]
                        except Exception:
                            to_capture[k] = v.decode()  # type: ignore[literal-required]
                    elif k in ['general_timeout_in_sec', 'depth']:
                        # int
                        to_capture[k] = int(v)  # type: ignore[literal-required]
                    elif k in ['rendered_hostname_only', 'with_favicon', 'allow_tracking']:
                        # bool
                        to_capture[k] = bool(int(v))  # type: ignore[literal-required]
                    elif k == 'document':
                        document_as_bytes = b64decode(v)
                    else:
                        raise CaptureSettingsError(f'Unexpected setting: {k}: {v}')
            except CaptureSettingsError as e:
                raise e
            except Exception as e:
                raise CaptureSettingsError(f'Error while preparing settings: {e}')

            if not to_capture:
                all_entries = self.redis.hgetall(f'lacus:capture_settings:{uuid}')
                result = {'error': f'No capture settings for {uuid} - {all_entries}'}
                raise CaptureError(f'No capture settings for {uuid} - {all_entries}')

            if document_as_bytes:
                # we do not have a URL yet.
                name = to_capture.pop('document_name', None)
                if not name:
                    raise CaptureSettingsError('No document name provided, settings are invalid')
                if not Path(name).suffix:
                    # The browser will simply display the file as text if there is no extension.
                    # Just add HTML as a fallback, as it will be the most comon one.
                    name = f'{name}.html'
                document_name = Path(name).name
                tmp_f = NamedTemporaryFile(suffix=document_name, delete=False)
                with open(tmp_f.name, "wb") as f:
                    f.write(document_as_bytes)
                url = f'file://{tmp_f.name}'
            elif to_capture.get('url') and to_capture['url'] is not None:
                url = to_capture['url'].strip()
                url = refang(url)  # In case we get a defanged url at this stage.
                if url.lower().startswith('file:') and self.only_global_lookups:
                    result = {'error': f'Not allowed to capture a file on disk: {url}'}
                    raise CaptureError(f'Not allowed to capture a file on disk: {url}')
                if (not url.lower().startswith('data:')
                        and not url.lower().startswith('http:')
                        and not url.lower().startswith('https:')
                        and not url.lower().startswith('file:')):
                    url = f'http://{url}'
            else:
                result = {'error': f'No valid URL to capture for {uuid} - {to_capture}'}
                raise CaptureError(f'No valid URL to capture for {uuid} - {to_capture}')

            try:
                splitted_url = urlsplit(url)
            except Exception as e:
                result = {'error': f'Invalid URL: {url} - {e}'}
                raise CaptureError(f'Invalid URL: {url} - {e}')
            proxy = to_capture.get('proxy')
            if self.tor_proxy:
                # check if onion or forced
                if (proxy == 'force_tor'  # if the proxy is set to "force_tor", we use the pre-configured tor proxy, regardless the URL.
                        or (not proxy  # if the TLD is "onion", we use the pre-configured tor proxy
                            and splitted_url.netloc
                            and splitted_url.hostname
                            and splitted_url.hostname.split('.')[-1] == 'onion')):
                    proxy = self.tor_proxy

            if self.only_global_lookups and not proxy and splitted_url.scheme not in ['data', 'file']:
                # not relevant if we also have a proxy, or the thing to capture is a data URI or a file on disk
                if splitted_url.netloc:
                    if splitted_url.hostname and splitted_url.hostname.split('.')[-1] != 'onion':
                        ips_to_check = []
                        # check if the hostname is an IP
                        try:
                            _ip = ip_address(splitted_url.hostname)
                            ips_to_check.append(_ip)
                        except ValueError:
                            # not an IP, try resolving
                            try:
                                ips_to_check = self.__get_ips(logger, splitted_url.hostname)
                            except DNSTimeout as e:
                                # for a timeout, we do not want to retry, as it is likely to timeout again
                                result = {'error': f'DNS Timeout for "{splitted_url.hostname}": {e}'}
                                raise CaptureError(f'DNS Timeout for "{splitted_url.hostname}": {e}')
                            except Exception as e:
                                result = {'error': f'Issue with hostname resolution ({splitted_url.hostname}): {e}. Full URL: "{url}".'}
                                raise CaptureError(f'Issue with hostname resolution ({splitted_url.hostname}): {e}. Full URL: "{url}".')
                        if not ips_to_check:
                            logger.debug(f'Unable to resolve "{splitted_url.hostname}" - Full URL: "{url}".')
                            result = {'error': f'Unable to resolve "{splitted_url.hostname}" - Full URL: "{url}".'}
                            raise RetryCapture(f'Unable to resolve "{splitted_url.hostname}" - Full URL: "{url}".')
                        for ip in ips_to_check:
                            if not ip.is_global:
                                result = {'error': f'Capturing ressources on private IPs ({ip}) is disabled.'}
                                raise CaptureError(f'Capturing ressources on private IPs ({ip}) is disabled.')
                else:
                    result = {'error': f'Unable to find hostname or IP in the query: "{url}".'}
                    raise CaptureError(f'Unable to find hostname or IP in the query: "{url}".')

            browser_engine: BROWSER = "chromium"
            if to_capture.get('user_agent'):
                parsed_string = user_agent_parser.ParseUserAgent(to_capture.get('user_agent'))
                browser_family = parsed_string['family'].lower()
                if browser_family.startswith('chrom'):
                    browser_engine = 'chromium'
                elif browser_family.startswith('firefox'):
                    browser_engine = 'firefox'
                else:
                    browser_engine = 'webkit'

            cookies: list[dict[str, Any]] = []
            if to_capture.get('cookies') and to_capture['cookies'] is not None:
                # In order to properly pass the cookies to playwright,
                # each of then must have a name, a value and either a domain + path or a URL
                # Name and value are mandatory, and we cannot auto-fill them.
                # If the cookie doesn't have a domain + path OR a URL, we fill the domain
                # with the hostname of the URL we try to capture and the path with "/"
                for cookie in to_capture['cookies']:
                    if len(cookie) == 1:
                        # we have a cookie in the format key: value
                        name, value = cookie.popitem()
                        cookie = {'name': name, 'value': value}
                    if 'name' not in cookie or 'value' not in cookie:
                        logger.warning(f'Invalid cookie: {cookie}')
                        continue
                    if 'domain' not in cookie and 'url' not in cookie:
                        cookie['domain'] = splitted_url.hostname
                        cookie['path'] = '/'
                    cookies.append(cookie)

            try:
                logger.debug(f'Capturing {url}')
                stats_pipeline.sadd(f'stats:{today}:captures', url)
                async with Capture(
                        browser=browser_engine,
                        device_name=to_capture.get('device_name'),
                        proxy=proxy,
                        general_timeout_in_sec=to_capture.get('general_timeout_in_sec'),
                        loglevel=self.master_logger.getEffectiveLevel(),
                        uuid=uuid) as capture:
                    # required by Mypy: https://github.com/python/mypy/issues/3004
                    capture.headers = to_capture.get('headers')  # type: ignore[assignment]
                    capture.cookies = cookies  # type: ignore[assignment]
                    capture.viewport = to_capture.get('viewport')  # type: ignore[assignment]
                    capture.user_agent = to_capture.get('user_agent')  # type: ignore[assignment]
                    capture.http_credentials = to_capture.get('http_credentials')  # type: ignore[assignment]
                    capture.geolocation = to_capture.get('geolocation')  # type: ignore[assignment]
                    capture.timezone_id = to_capture.get('timezone_id')  # type: ignore[assignment]
                    capture.locale = to_capture.get('locale')  # type: ignore[assignment]
                    capture.color_scheme = to_capture.get('color_scheme')  # type: ignore[assignment]

                    # make sure the initialization doesn't take too long
                    init_timeout = max(self.max_capture_time / 10, 5)
                    try:
                        async with timeout(init_timeout) as initialize_timeout:
                            await capture.initialize_context()
                    except (TimeoutError, asyncio.exceptions.TimeoutError):
                        timeout_expired(initialize_timeout, logger, 'Initializing took too long.')
                        logger.warning(f'Initializing the context for {url} took longer than the allowed initialization timeout ({init_timeout}s)')
                        raise RetryCapture(f'Initializing the context for {url} took longer than the allowed initialization timeout ({init_timeout}s)')

                    try:
                        async with timeout(self.max_capture_time) as capture_timeout:
                            playwright_result = await capture.capture_page(
                                url, referer=to_capture.get('referer'),
                                depth=to_capture.get('depth', 0),
                                rendered_hostname_only=to_capture.get('rendered_hostname_only', True),
                                with_favicon=to_capture.get('with_favicon', False),
                                allow_tracking=to_capture.get('allow_tracking', False),
                                max_depth_capture_time=self.max_capture_time)
                    except (TimeoutError, asyncio.exceptions.TimeoutError):
                        timeout_expired(capture_timeout, logger, 'Capture took too long.')
                        logger.warning(f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                        raise RetryCapture(f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                    result = cast(CaptureResponse, playwright_result)
                    if 'error' in result and 'error_name' in result:
                        # generate stats
                        if result['error_name'] is not None:
                            stats_pipeline.zincrby(f'stats:{today}:errors', 1, result['error_name'])
            except RetryCapture as e:
                raise e
            except PlaywrightCaptureException as e:
                logger.warning(f'Invalid parameters for the capture of {url} - {e}')
                result = {'error': f'Invalid parameters for the capture of {url} - {e}'}
                raise CaptureError(f'Invalid parameters for the capture of {url} - {e}')
            except asyncio.CancelledError:
                logger.warning(f'The capture of {url} has been cancelled.')
                result = {'error': f'The capture of {url} has been cancelled.'}
                # The capture can be canceled if it has been running for way too long.
                # We can give it another short.
                raise RetryCapture(f'The capture of {url} has been cancelled.')
            except Exception as e:
                logger.exception(f'Something went poorly {url} - {e}')
                result = {'error': f'Something went poorly {url} - {e}'}
                raise CaptureError(f'Something went poorly {url} - {e}')

            if capture.should_retry:
                # PlaywrightCapture considers this capture elligible for a retry
                logger.info('PlaywrightCapture considers it elligible for a retry.')
                raise RetryCapture('PlaywrightCapture considers it elligible for a retry.')
            elif self.redis.exists(f'lacus:capture_retry:{uuid}'):
                # this is a retry that worked
                stats_pipeline.sadd(f'stats:{today}:retry_success', url)
        except RetryCapture:
            # Check if we already re-tried this capture
            _current_retry = self.redis.get(f'lacus:capture_retry:{uuid}')
            if _current_retry is None:
                # No retry yet
                logger.debug(f'Retrying {url} for the first time.')
                retry = True
                self.redis.setex(f'lacus:capture_retry:{uuid}',
                                 self.max_capture_time * (self.max_retries + 1),
                                 self.max_retries)
            else:
                current_retry = int(_current_retry.decode())
                if current_retry > 0:
                    logger.debug(f'Retrying {url} for the {self.max_retries - current_retry + 1}th time.')
                    self.redis.decr(f'lacus:capture_retry:{uuid}')
                    retry = True
                else:
                    error_msg = result['error'] if result.get('error') else 'Unknown error'
                    logger.info(f'Retried too many times {url}: {error_msg}')
                    stats_pipeline.sadd(f'stats:{today}:retry_failed', url)
        except CaptureError:
            if not result:
                result = {'error': "No result key, shouldn't happen"}
                logger.exception(f'Unable to capture: {result["error"]}')
            if url:
                logger.warning(f'Unable to capture {url}: {result["error"]}')
            else:
                logger.warning(f'Unable to capture: {result["error"]}')
        except Exception as e:
            msg = f'Something unexpected happened with {url}: {e}'
            result = {'error': msg}
            logger.exception(msg)
        else:
            if start_time := self.redis.zscore('lacus:ongoing', uuid):
                runtime = time.time() - start_time
                logger.info(f'Capture of {url} finished - Runtime: {runtime}s')
                result['runtime'] = runtime
            else:
                logger.info(f'Capture of {url} finished - No Runtime.')
        finally:
            # NOTE: in this block, we absolutely have to make sure the UUID is removed
            #       from the lacus:ongoing sorted set (it is definitely not ongoing anymore)
            #       and optionally re-added to lacus:to_capture if re want to retry it

            if to_capture.get('document'):
                os.unlink(tmp_f.name)

            if retry:
                if self.redis.zcard('lacus:to_capture') == 0:
                    # Just wait a little bit before retrying
                    await asyncio.sleep(random.randint(5, 10))
                p = self.redis.pipeline()
                p.zrem('lacus:ongoing', uuid)
                p.zadd('lacus:to_capture', {uuid: priority - 1})
                p.execute()
            else:
                retry_redis_error = 3
                while retry_redis_error > 0:
                    try:
                        p = self.redis.pipeline()
                        if result:
                            self._store_capture_response(p, uuid, result)
                        else:
                            logger.warning('Got no result at all for the capture.')
                        p.delete(f'lacus:capture_settings:{uuid}')
                        p.zrem('lacus:ongoing', uuid)
                        p.execute()
                        break
                    except RedisConnectionError as e:
                        logger.warning(f'Unable to store capture result - Redis Connection Error: {e}')
                        retry_redis_error -= 1
                        await asyncio.sleep(random.randint(5, 10))
                else:
                    self.redis.zrem('lacus:ongoing', uuid)
                    stats_pipeline.zincrby(f'stats:{today}:errors', 1, 'Redis Connection')
                    logger.critical('Unable to connect to redis and to push the result of the capture.')

            # Expire stats in 10 days
            expire_time = timedelta(days=10)
            stats_pipeline.expire(f'stats:{today}:errors', expire_time)
            stats_pipeline.expire(f'stats:{today}:retry_failed', expire_time)
            stats_pipeline.expire(f'stats:{today}:retry_success', expire_time)
            stats_pipeline.expire(f'stats:{today}:captures', expire_time)
            stats_pipeline.execute()

    def _store_capture_response(self, pipeline: Redis, capture_uuid: str, results: CaptureResponse,   # type: ignore[type-arg]
                                root_key: str | None=None) -> None:
        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': capture_uuid})
        if root_key is None:
            root_key = f'lacus:capture_results_hash:{capture_uuid}'

        hash_to_set = {}
        if results.get('har'):
            hash_to_set['har'] = pickle.dumps(results['har'])
        if results.get('cookies'):
            hash_to_set['cookies'] = pickle.dumps(results['cookies'])
        if results.get('potential_favicons'):
            hash_to_set['potential_favicons'] = pickle.dumps(results['potential_favicons'])
        if results.get('html') and results['html'] is not None:
            # Need to avoid unicode encore errors, and surrogates are not allowed
            hash_to_set['html'] = results['html'].encode('utf-8', 'surrogateescape')
        if 'children' in results and results['children'] is not None:
            padding_length = len(str(len(results['children'])))
            children = set()
            for i, child in enumerate(results['children']):
                child_key = f'{root_key}_{i:0{padding_length}}'
                if not child:
                    # the child key is empty
                    logger.info(f'The response for {child_key} is empty.')
                    continue
                self._store_capture_response(pipeline, capture_uuid, child, child_key)
                children.add(child_key)
            hash_to_set['children'] = pickle.dumps(children)

        for key in results.keys():
            if key in ['har', 'cookies', 'potential_favicons', 'html', 'children'] or not results.get(key):
                continue
            # these entries can be stored directly
            hash_to_set[key] = results[key]  # type: ignore[literal-required]

        if hash_to_set:
            pipeline.hset(root_key, mapping=hash_to_set)  # type: ignore[arg-type]
            # Make sure the key expires
            pipeline.expire(root_key, 36000)
        else:
            logger.critical(f'Nothing to store (Hash: {hash_to_set}) for {root_key}')

    def _get_capture_response(self, capture_uuid: str, root_key: str | None=None) -> CaptureResponse | None:
        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': capture_uuid})
        if root_key is None:
            root_key = f'lacus:capture_results_hash:{capture_uuid}'

            if not self.redis.exists(root_key):
                if old_response := self.redis.get(f'lacus:capture_results:{capture_uuid}'):
                    # TODO: remove in 1.8.* - old format used last in 1.6, and kept no more than 10H in redis
                    return pickle.loads(zlib.decompress(old_response))
                return None

        # New format and capture done

        to_return: CaptureResponse = {}
        for key, value in self.redis.hgetall(root_key).items():
            if key == b'har':
                to_return['har'] = pickle.loads(value)
            elif key == b'cookies':
                to_return['cookies'] = pickle.loads(value)
            elif key == b'potential_favicons':
                to_return['potential_favicons'] = pickle.loads(value)
            elif key == b'children':
                to_return['children'] = []
                for child_root_key in sorted(pickle.loads(value)):
                    if child := self._get_capture_response(capture_uuid, child_root_key):
                        to_return['children'].append(child)  # type: ignore[union-attr]
            elif key in [b'status']:
                # The value in an int
                to_return[key.decode()] = int(value)  # type: ignore[literal-required]
            elif key in [b'runtime']:
                # The value is a float
                to_return[key.decode()] = float(value)  # type: ignore[literal-required]
            elif key in [b'last_redirected_url', b'error', b'error_name', b'html', b'downloaded_filename']:
                # the value is a string
                to_return[key.decode()] = value.decode()  # type: ignore[literal-required]
            elif key in [b'png', b'downloaded_file']:
                # the value is bytes
                to_return[key.decode()] = value  # type: ignore[literal-required]
            else:
                logger.critical(f'Unexpected key in response: {key} - {value}')
        return to_return

    def clear_capture(self, uuid: str, reason: str) -> None:
        '''Remove a capture from the list, shouldn't happen unless it is in error'''
        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': uuid})
        capture_status = self.get_capture_status(uuid)
        if capture_status == CaptureStatus.ONGOING:
            # Check when it was started.
            if start_time := self.redis.zscore('lacus:ongoing', uuid):
                if start_time > time.time() - self.max_capture_time * 1.1:
                    # The capture started recently, wait before clearing it.
                    logger.warning('The capture is (probably) still going, not clearing.')
                    return
        elif capture_status == CaptureStatus.QUEUED:
            logger.warning('The capture is queued, not clearing.')
            return
        logger.warning(f'Clearing capture: {reason}')
        result: CaptureResponse = {'error': reason}
        p = self.redis.pipeline()
        self._store_capture_response(p, uuid, result)
        p.delete(f'lacus:capture_settings:{uuid}')
        p.zrem('lacus:ongoing', uuid)
        p.execute()

    def __get_ips(self, logger: LacusCoreLogAdapter, hostname: str) -> list[IPv4Address | IPv6Address]:
        # We need to use dnspython for resolving because socket.getaddrinfo will sometimes be stuck for ~10s
        # It is happening when the error code is NoAnswer
        resolved_ips = []
        try:
            answers_a = self.dnsresolver.resolve(hostname, 'A')
            resolved_ips += [ip_address(str(answer)) for answer in answers_a]
        except DNSTimeout as e:
            raise e
        except DNSException as e:
            logger.info(f'No A record for "{hostname}": {e}')
        try:
            answers_aaaa = self.dnsresolver.resolve(hostname, 'AAAA')
            resolved_ips += [ip_address(str(answer)) for answer in answers_aaaa]
        except DNSTimeout as e:
            raise e
        except DNSException as e:
            logger.info(f'No AAAA record for "{hostname}": {e}')
        return resolved_ips
