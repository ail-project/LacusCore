#!/usr/bin/env python3

import asyncio
import ipaddress
import hashlib
import json
import logging
import os
import pickle
import random
import re
import socket
import sys
import time
import unicodedata
import zlib

from asyncio import Task
from base64 import b64decode, b64encode
from datetime import date, timedelta
from enum import IntEnum, unique
from logging import LoggerAdapter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional, Union, Dict, List, Any, TypedDict, overload, Tuple, cast, MutableMapping, Iterator
from uuid import uuid4
from urllib.parse import urlsplit

from defang import refang  # type: ignore
from playwrightcapture import Capture, PlaywrightCaptureException
from playwrightcapture.capture import CaptureResponse as PlaywrightCaptureResponse
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from ua_parser import user_agent_parser  # type: ignore

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


class LacusCoreException(Exception):
    pass


class CaptureError(LacusCoreException):
    pass


class RetryCapture(LacusCoreException):
    pass


@unique
class CaptureStatus(IntEnum):
    '''The status of the capture'''
    UNKNOWN = -1
    QUEUED = 0
    DONE = 1
    ONGOING = 2


class CaptureResponse(PlaywrightCaptureResponse, TypedDict, total=False):
    '''A capture made by Lacus. With the base64 encoded image and downloaded file decoded to bytes.'''

    status: int
    runtime: Optional[float]


class CaptureResponseJson(TypedDict, total=False):
    '''A capture made by Lacus. With the base64 encoded image and downloaded file *not* decoded.'''

    status: int
    last_redirected_url: Optional[str]
    har: Optional[Dict[str, Any]]
    cookies: Optional[List[Dict[str, str]]]
    error: Optional[str]
    html: Optional[str]
    png: Optional[str]
    downloaded_filename: Optional[str]
    downloaded_file: Optional[str]
    children: Optional[List[Any]]
    runtime: Optional[float]


class CaptureSettings(TypedDict, total=False):
    '''The capture settings that can be passed to Lacus.'''

    url: Optional[str]
    document_name: Optional[str]
    document: Optional[str]
    browser: Optional[str]
    device_name: Optional[str]
    user_agent: Optional[str]
    proxy: Optional[Union[str, Dict[str, str]]]
    general_timeout_in_sec: Optional[int]
    cookies: Optional[List[Dict[str, Any]]]
    headers: Optional[Union[str, Dict[str, str]]]
    http_credentials: Optional[Dict[str, str]]
    geolocation: Optional[Dict[str, float]]
    timezone_id: Optional[str]
    locale: Optional[str]
    color_scheme: Optional[str]
    viewport: Optional[Dict[str, int]]
    referer: Optional[str]
    force: Optional[bool]
    recapture_interval: Optional[int]
    priority: Optional[int]
    uuid: Optional[str]

    depth: int
    rendered_hostname_only: bool  # Note: only used if depth is > 0


class LacusCoreLogAdapter(LoggerAdapter):
    """
    Prepend log entry with the UUID of the capture
    """
    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        if self.extra:
            return '[%s] %s' % (self.extra['uuid'], msg), kwargs
        return msg, kwargs


def _json_encode(obj: Union[bytes]) -> str:
    if isinstance(obj, bytes):
        return b64encode(obj).decode()


class LacusCore():
    """Capture URLs or web enabled documents using PlaywrightCapture.

    :param redis_connector: Pre-configured connector to a redis instance.
    :param max_capture time: If the capture takes more than that time, break
    :param tor_proxy: URL to a SOCKS 5 tor proxy. If you have tor installed, this is the default: socks5://127.0.0.1:9050.
    :param only_global_lookups: Discard captures that point to non-public IPs.
    :param max_retries: How many times should we re-try a capture if it failed.
    """

    def __init__(self, redis_connector: Redis, /, *,
                 max_capture_time: int=3600,
                 tor_proxy: Optional[str]=None,
                 only_global_lookups: bool=True,
                 max_retries: int=3,
                 loglevel: str='INFO') -> None:
        self.master_logger = logging.getLogger(f'{self.__class__.__name__}')
        self.master_logger.setLevel(loglevel)
        self.redis = redis_connector
        self.max_capture_time = max_capture_time
        self.tor_proxy = tor_proxy
        self.only_global_lookups = only_global_lookups
        self.max_retries = max_retries

        # NOTE: clear old ongoing captures queue in case of need
        if self.redis.type('lacus:ongoing') in ['set', b'set']:
            self.redis.delete('lacus:ongoing')

    def check_redis_up(self) -> bool:
        """Check if redis is reachable"""
        return bool(self.redis.ping())

    @overload
    def enqueue(self, *, settings: Optional[CaptureSettings]=None) -> str:
        ...

    @overload
    def enqueue(self, *,
                url: Optional[str]=None,
                document_name: Optional[str]=None, document: Optional[str]=None,
                depth: int=0,
                browser: Optional[BROWSER]=None, device_name: Optional[str]=None,
                user_agent: Optional[str]=None,
                proxy: Optional[Union[str, Dict[str, str]]]=None,
                general_timeout_in_sec: Optional[int]=None,
                cookies: Optional[List[Dict[str, Any]]]=None,
                headers: Optional[Union[str, Dict[str, str]]]=None,
                http_credentials: Optional[Dict[str, str]]=None,
                geolocation: Optional[Dict[str, float]]=None,
                timezone_id: Optional[str]=None,
                locale: Optional[str]=None,
                color_scheme: Optional[str]=None,
                viewport: Optional[Dict[str, int]]=None,
                referer: Optional[str]=None,
                rendered_hostname_only: bool=True,
                force: bool=False,
                recapture_interval: int=300,
                priority: int=0,
                uuid: Optional[str]=None
                ) -> str:
        ...

    def enqueue(self, *,
                settings: Optional[CaptureSettings]=None,
                url: Optional[str]=None,
                document_name: Optional[str]=None, document: Optional[str]=None,
                depth: int=0,
                browser: Optional[BROWSER]=None, device_name: Optional[str]=None,
                user_agent: Optional[str]=None,
                proxy: Optional[Union[str, Dict[str, str]]]=None,
                general_timeout_in_sec: Optional[int]=None,
                cookies: Optional[List[Dict[str, Any]]]=None,
                headers: Optional[Union[str, Dict[str, str]]]=None,
                http_credentials: Optional[Dict[str, str]]=None,
                geolocation: Optional[Dict[str, float]]=None,
                timezone_id: Optional[str]=None,
                locale: Optional[str]=None,
                color_scheme: Optional[str]=None,
                viewport: Optional[Dict[str, int]]=None,
                referer: Optional[str]=None,
                rendered_hostname_only: bool=True,
                force: bool=False,
                recapture_interval: int=300,
                priority: int=0,
                uuid: Optional[str]=None
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
        :param force: Force recapture, even if the same one was already done within the recapture_interval
        :param recapture_interval: The time the enqueued settings are kept in memory to avoid duplicates
        :param priority: The priority of the capture
        :param uuid: The preset priority of the capture, auto-generated if not present. Should only be used if the initiator couldn't enqueue immediately. NOTE: it will be overwritten if the UUID already exists.

        :return: UUID, reference to the capture for later use
        """
        to_enqueue: CaptureSettings
        if settings:
            if settings.get('force') is not None:
                force = settings.pop('force')  # type: ignore
            if settings.get('recapture_interval') is not None:
                recapture_interval = settings.pop('recapture_interval')  # type: ignore
            if settings.get('priority') is not None:
                priority = settings.pop('priority')  # type: ignore
            to_enqueue = settings
        else:
            to_enqueue = {'depth': depth, 'rendered_hostname_only': rendered_hostname_only}
            if url:
                to_enqueue['url'] = url
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

        mapping_capture: Dict[str, Union[bytes, float, int, str]] = {}
        for key, value in to_enqueue.items():
            if isinstance(value, bool):
                mapping_capture[key] = 1 if value else 0
            elif isinstance(value, (list, dict)):
                if value:
                    mapping_capture[key] = json.dumps(value)
            elif value is not None and value != '':  # we're ok with 0 for example
                mapping_capture[key] = value  # type: ignore

        p = self.redis.pipeline()
        p.set(f'lacus:query_hash:{hash_query}', perma_uuid, nx=True, ex=recapture_interval)
        p.hset(f'lacus:capture_settings:{perma_uuid}', mapping=mapping_capture)  # type: ignore
        p.zadd('lacus:to_capture', {perma_uuid: priority})
        p.execute()
        return perma_uuid

    def _encode_response(self, capture: CaptureResponse) -> CaptureResponseJson:
        encoded_capture = cast(CaptureResponseJson, capture)
        if capture.get('png') is not None and capture['png'] is not None:  # the second part is not needed, but makes mypy happy
            encoded_capture['png'] = b64encode(capture['png']).decode()
        if capture.get('downloaded_file') is not None and capture['downloaded_file'] is not None:  # the second part is not needed, but makes mypy happy
            encoded_capture['downloaded_file'] = b64encode(capture['downloaded_file']).decode()
        if capture.get('children') and capture['children']:
            for child in capture['children']:
                child = self._encode_response(child)
        return encoded_capture

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[True]=True) -> CaptureResponse:
        ...

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[False]) -> CaptureResponseJson:
        ...

    def get_capture(self, uuid: str, *, decode: bool=False) -> Union[CaptureResponse, CaptureResponseJson]:
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
        elif response := self.redis.get(f'lacus:capture_results:{uuid}'):
            to_return['status'] = CaptureStatus.DONE
            response_json = pickle.loads(zlib.decompress(response))
            to_return.update(response_json)
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
        elif self.redis.exists(f'lacus:capture_results:{uuid}'):
            return CaptureStatus.DONE
        return CaptureStatus.UNKNOWN

    def consume_queue(self, max_consume: int) -> Iterator[Task]:
        """Trigger the capture for captures with the highest priority. Up to max_consume.

        :yield: Captures.
        """
        value: List[Tuple[bytes, float]]
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
            capture = asyncio.create_task(self._capture(uuid, priority))
            capture.set_name(uuid)
            yield capture

    async def _capture(self, uuid: str, priority: int):
        """Trigger a specific capture

        :param uuid: The UUID if the capture (given by enqueue)
        :param priority: Only for internal use, will decide on the priority of the capture if the try now fails.
        """
        if self.redis.zscore('lacus:ongoing', uuid) is not None:
            # the capture is ongoing
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
                            'color_scheme']
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
                        to_capture[k] = v.decode()  # type: ignore
                    elif k in ['cookies', 'http_credentials', 'viewport', 'geolocation']:
                        # dicts or list
                        to_capture[k] = json.loads(v)  # type: ignore
                    elif k in ['proxy', 'headers']:
                        # can be dict or str
                        try:
                            to_capture[k] = json.loads(v)  # type: ignore
                        except Exception:
                            to_capture[k] = v.decode()  # type: ignore
                    elif k in ['general_timeout_in_sec', 'depth']:
                        # int
                        to_capture[k] = int(v)  # type: ignore
                    elif k in ['rendered_hostname_only']:
                        # bool
                        to_capture[k] = bool(int(v))  # type: ignore
                    elif k == 'document':
                        document_as_bytes = b64decode(v)
                    else:
                        raise LacusCoreException(f'Unexpected setting: {k}: {v}')
            except LacusCoreException as e:
                raise e
            except Exception as e:
                raise LacusCoreException(f'Error while preparing settings: {e}')

            if not to_capture:
                result = {'error': f'No capture settings for {uuid}.'}
                raise CaptureError

            if document_as_bytes:
                # we do not have a URL yet.
                name = to_capture.pop('document_name', None)
                if not name:
                    raise LacusCoreException('No document name provided, settings are invalid')
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
                    raise CaptureError
                if not url.lower().startswith('data') and not url.lower().startswith('http') and not url.lower().startswith('file:'):
                    url = f'http://{url}'
            else:
                result = {'error': f'No valid URL to capture for {uuid} - {to_capture}'}
                raise CaptureError

            splitted_url = urlsplit(url)
            proxy = to_capture.get('proxy')
            if self.tor_proxy:
                # check if onion or forced
                if (proxy == 'force_tor'
                        or (not proxy
                            and splitted_url.netloc
                            and splitted_url.hostname
                            and splitted_url.hostname.split('.')[-1] == 'onion')):
                    proxy = self.tor_proxy

            if self.only_global_lookups and not proxy and splitted_url.scheme not in ['data', 'file']:
                # not relevant if we also have a proxy, or the thing to capture is a data URI or a file on disk
                if splitted_url.netloc:
                    if splitted_url.hostname and splitted_url.hostname.split('.')[-1] != 'onion':
                        try:
                            ips_info = socket.getaddrinfo(splitted_url.hostname, None, proto=socket.IPPROTO_TCP)
                        except socket.gaierror:
                            logger.debug(f'Unable to resolve {splitted_url.hostname}.')
                            result = {'error': f'Unable to resolve {splitted_url.hostname}.'}
                            raise RetryCapture
                        except Exception as e:
                            result = {'error': f'Issue with hostname resolution ({splitted_url.hostname}): {e}.'}
                            raise CaptureError
                        for info in ips_info:
                            if not ipaddress.ip_address(info[-1][0]).is_global:
                                result = {'error': f'Capturing ressources on private IPs {info[-1][0]} is disabled.'}
                                raise CaptureError
                else:
                    result = {'error': f'Unable to find hostname or IP in the query: {url}.'}
                    raise CaptureError

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
            try:
                logger.debug(f'Capturing {url}')
                # NOTE: starting with python 3.11, we can use asyncio.timeout
                # async with asyncio.timeout(self.max_capture_time):
                general_timeout = to_capture.get('general_timeout_in_sec')
                stats_pipeline.sadd(f'stats:{today}:captures', url)
                async with Capture(
                        browser=browser_engine,
                        device_name=to_capture.get('device_name'),
                        proxy=proxy,
                        general_timeout_in_sec=general_timeout) as capture:
                    # required by Mypy: https://github.com/python/mypy/issues/3004
                    capture.headers = to_capture.get('headers')  # type: ignore
                    capture.cookies = to_capture.get('cookies')  # type: ignore
                    capture.viewport = to_capture.get('viewport')  # type: ignore
                    capture.user_agent = to_capture.get('user_agent')  # type: ignore
                    capture.http_credentials = to_capture.get('http_credentials')  # type: ignore
                    capture.geolocation = to_capture.get('geolocation')  # type: ignore
                    capture.timezone_id = to_capture.get('timezone_id')  # type: ignore
                    capture.locale = to_capture.get('locale')  # type: ignore
                    capture.color_scheme = to_capture.get('color_scheme')  # type: ignore
                    try:
                        await asyncio.wait_for(capture.initialize_context(), timeout=general_timeout)
                    except (TimeoutError, asyncio.exceptions.TimeoutError):
                        logger.warning(f'Initializing the context for {url} took longer than the allowed general timeout ({general_timeout}s)')
                        raise RetryCapture
                    playwright_result = await asyncio.wait_for(
                        capture.capture_page(
                            url, referer=to_capture.get('referer'),
                            depth=to_capture.get('depth', 0),
                            rendered_hostname_only=to_capture.get('rendered_hostname_only', True),
                            max_depth_capture_time=self.max_capture_time),
                        timeout=self.max_capture_time)
                    if 'error' in playwright_result and 'error_name' in playwright_result:
                        # generate stats
                        if playwright_result['error_name'] is not None:
                            stats_pipeline.zincrby(f'stats:{today}:errors', 1, playwright_result['error_name'])
            except PlaywrightCaptureException as e:
                logger.exception(f'Invalid parameters for the capture of {url} - {e}')
                result = {'error': f'Invalid parameters for the capture of {url} - {e}'}
                raise CaptureError
            except asyncio.CancelledError:
                logger.warning(f'The capture of {url} has been cancelled.')
                result = {'error': f'The capture of {url} has been cancelled.'}
                raise CaptureError
            except (TimeoutError, asyncio.exceptions.TimeoutError):
                logger.warning(f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                result = {'error': f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)'}
                raise CaptureError
            except Exception as e:
                logger.exception(f'Something went poorly {url} - {e}')
                result = {'error': f'Something went poorly {url} - {e}'}
                raise CaptureError

            if capture.should_retry:
                # PlaywrightCapture considers this capture elligible for a retry
                logger.info('PlaywrightCapture considers it elligible for a retry.')
                raise RetryCapture
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
                    logger.debug(f'Retrying {url} for the {self.max_retries-current_retry+1}th time.')
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
            result = cast(CaptureResponse, playwright_result)
            if start_time := self.redis.zscore('lacus:ongoing', uuid):
                runtime = time.time() - start_time
                logger.info(f'Capture of {url} finished - Runtime: {runtime}s')
                result['runtime'] = runtime
            else:
                logger.info(f'Capture of {url} finished - No Runtime.')
        finally:

            if to_capture.get('document'):
                os.unlink(tmp_f.name)

            if retry:
                if self.redis.zcard('lacus:to_capture') == 0:
                    # Just wait a little bit before retrying
                    await asyncio.sleep(random.randint(5, 10))
                self.redis.zadd('lacus:to_capture', {uuid: priority - 1})
            else:
                to_store = zlib.compress(pickle.dumps(result))
                retry_redis_error = 3
                while retry_redis_error > 0:
                    try:
                        p = self.redis.pipeline()
                        p.setex(f'lacus:capture_results:{uuid}', 36000, to_store)
                        p.delete(f'lacus:capture_settings:{uuid}')
                        p.zrem('lacus:ongoing', uuid)
                        p.execute()
                        break
                    except RedisConnectionError as e:
                        logger.warning(f'Unable to store capture result (size: {sys.getsizeof(to_store)} - Redis Connection Error: {e}')
                        retry_redis_error -= 1
                        await asyncio.sleep(random.randint(5, 10))
                else:
                    stats_pipeline.zincrby(f'stats:{today}:errors', 1, 'Redis Connection')
                    logger.critical('Unable to connect to redis and to push the result of the capture.')

            # Expire stats in 10 days
            expire_time = timedelta(days=10)
            stats_pipeline.expire(f'stats:{today}:errors', expire_time)
            stats_pipeline.expire(f'stats:{today}:retry_failed', expire_time)
            stats_pipeline.expire(f'stats:{today}:captures', expire_time)
            stats_pipeline.execute()

    def clear_capture(self, uuid: str, reason: str):
        '''Remove a capture from the list, shouldn't happen unless it is in error'''
        result = {'error': reason}
        p = self.redis.pipeline()
        to_store = zlib.compress(pickle.dumps(result))
        p.setex(f'lacus:capture_results:{uuid}', 36000, to_store)
        p.delete(f'lacus:capture_settings:{uuid}')
        p.zrem('lacus:ongoing', uuid)
        p.execute()
