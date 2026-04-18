#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import random
import re
import socket
import sys
import time
import unicodedata

from asyncio import Task
from base64 import b64decode, b64encode
from datetime import date, timedelta
from ipaddress import ip_address, IPv4Address, IPv6Address
from tempfile import NamedTemporaryFile
from typing import Literal, Any, overload, cast
from collections.abc import AsyncIterator
from uuid import uuid4
from urllib.parse import urlsplit

from dns.resolver import Cache
from dns.asyncresolver import Resolver
from dns.exception import DNSException
from dns.exception import Timeout as DNSTimeout

from lookyloo_models import (CaptureSettingsError, CaptureSettings, ViewportSettings,
                             GeolocationSettings, HttpCredentialsSettings,
                             Cookie)
from playwrightcapture import Capture, PlaywrightCaptureException, InvalidPlaywrightParameter, TrustedTimestampSettings
from pydantic import ValidationError
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import DataError

from . import task_logger
from .helpers import (
    LacusCoreException,
    LacusCoreLogAdapter, CaptureError, RetryCapture,
    CaptureStatus, SessionStatus, CaptureResponse, CaptureResponseJson,
    SessionMetadata, RemoteHeadfullSessionError)
from .session import SessionManager
from .xpra_session import XpraSessionManager

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


def _check_proxy_port_open(proxy: dict[str, str] | str) -> bool:
    if isinstance(proxy, dict):
        to_check = proxy['server']
    else:
        to_check = proxy
    splitted_proxy_url = urlsplit(to_check)
    if not splitted_proxy_url.hostname or not splitted_proxy_url.port:
        raise LacusCoreException('Invalid pre-defined proxy (needs hostname and port): {proxy}')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        return s.connect_ex((splitted_proxy_url.hostname, splitted_proxy_url.port)) == 0


class LacusCore():
    """Capture URLs or web enabled documents using PlaywrightCapture.

    :param redis_connector: Pre-configured connector to a redis instance.
    :param max_capture_time: If the capture takes more than that time, break (in seconds)
    :param expire_results: The capture results are stored in redis. Expire them after they are done (in seconds).
    :param tor_proxy: URL to a SOCKS5 tor proxy. If you have tor installed, this is the default: socks5://127.0.0.1:9050.
    :param i2p_proxy: URL to a HTTP I2P proxy. If you have i2p installed, this is the default: http://127.0.0.1:4444.
    :param only_global_lookups: Discard captures that point to non-public IPs.
    :param max_retries: How many times should we re-try a capture if it failed.
    :param headed_allowed: Allow to launch captures in a local headed browser.
    :param remote_headed_allowed: Allow to trigger a capture in a remote headed browser.
    :param remote_headed_backend_type: The backend type for the remote headed captures (curently, Xpra).
    :param tt_settings: The settings for the Trusted Timestamps.
    """

    def __init__(self, redis_connector: Redis[bytes], /, *,
                 max_capture_time: int=3600,
                 expire_results: int=36000,
                 tor_proxy: dict[str, str] | str | None=None,
                 i2p_proxy: dict[str, str] | str | None=None,
                 only_global_lookups: bool=True,
                 max_retries: int=3,
                 headed_allowed: bool=False,
                 remote_headed_allowed: bool=False,
                 remote_headed_backend_type: str | None=None,
                 tt_settings: TrustedTimestampSettings | None=None,
                 loglevel: str | int='INFO') -> None:
        self.master_logger = logging.getLogger(f'{self.__class__.__name__}')
        self.master_logger.setLevel(loglevel)

        self.redis = redis_connector
        self.max_capture_time = max_capture_time
        self.expire_results = expire_results

        self.tor_proxy = tor_proxy
        self.i2p_proxy = i2p_proxy

        self.tt_settings = tt_settings

        self.only_global_lookups = only_global_lookups
        self.max_retries = max_retries
        self.headed_allowed = headed_allowed
        self.remote_headed_allowed = remote_headed_allowed
        self.remote_headed_backend_type: str | None = None
        self.remote_headed_session_manager: SessionManager | None = None
        if self.remote_headed_allowed:
            self.remote_headed_backend_type = remote_headed_backend_type
            if self.remote_headed_backend_type:
                self.remote_headed_session_manager = self._get_session_manager(self.remote_headed_backend_type, self.redis)
        self.dnsresolver: Resolver = Resolver()
        self.dnsresolver.cache = Cache(900)
        self.dnsresolver.timeout = 2
        self.dnsresolver.lifetime = 3

    def check_redis_up(self) -> bool:
        """Check if redis is reachable"""
        return bool(self.redis.ping())

    @overload
    def enqueue(self, *, settings: CaptureSettings | dict[str, Any] | None=None) -> str:
        ...

    @overload
    def enqueue(self, *,
                url: str | None=None,
                document_name: str | None=None, document: str | None=None,
                depth: int=0,
                browser: BROWSER | None=None, device_name: str | None=None,
                user_agent: str | None=None,
                proxy: str | dict[str, str] | None=None,
                socks5_dns_resolver: str | list[str] | None=None,
                general_timeout_in_sec: int | None=None,
                cookies: list[dict[str, Any]] | list[Cookie] | None=None,
                storage: dict[str, Any] | None=None,
                headers: dict[str, str] | None=None,
                http_credentials: dict[str, str] | HttpCredentialsSettings | None=None,
                geolocation: dict[str, str | int | float] | GeolocationSettings | None=None,
                timezone_id: str | None=None,
                locale: str | None=None,
                color_scheme: str | None=None,
                java_script_enabled: bool=True,
                viewport: dict[str, int | str] | ViewportSettings | None=None,
                referer: str | None=None,
                rendered_hostname_only: bool=True,
                with_screenshot: bool=True,
                with_favicon: bool=False,
                with_trusted_timestamps: bool=False,
                allow_tracking: bool=False,
                headless: bool=True,
                remote_headfull: bool=False,
                max_retries: int | None=None,
                init_script: str | None=None,
                force: bool=False,
                recapture_interval: int=300,
                final_wait: int=5,
                priority: int=0,
                uuid: str | None=None
                ) -> str:
        ...

    def enqueue(self, *,
                settings: CaptureSettings | dict[str, Any] | None=None,
                url: str | None=None,
                document_name: str | None=None, document: str | None=None,
                depth: int=0,
                browser: BROWSER | None=None, device_name: str | None=None,
                user_agent: str | None=None,
                proxy: str | dict[str, str] | None=None,
                socks5_dns_resolver: str | list[str] | None=None,
                general_timeout_in_sec: int | None=None,
                cookies: str | dict[str, str] | list[dict[str, Any]] | list[Cookie] | None=None,
                storage: dict[str, Any] | None=None,
                headers: dict[str, str] | None=None,
                http_credentials: dict[str, str] | HttpCredentialsSettings | None=None,
                geolocation: dict[str, str | int | float] | GeolocationSettings | None=None,
                timezone_id: str | None=None,
                locale: str | None=None,
                color_scheme: str | None=None,
                java_script_enabled: bool=True,
                viewport: dict[str, int | str] | ViewportSettings | None=None,
                referer: str | None=None,
                rendered_hostname_only: bool=True,
                with_screenshot: bool=True,
                with_favicon: bool=False,
                with_trusted_timestamps: bool=False,
                allow_tracking: bool=False,
                headless: bool=True,
                remote_headfull: bool=False,
                max_retries: int | None=None,
                init_script: str | None=None,
                force: bool=False,
                recapture_interval: int=300,
                final_wait: int=5,
                priority: int=0,
                uuid: str | None=None
                ) -> str:
        """Enqueue settings.

        :param settings: Settings as a dictionary

        :param url: URL to capture (incompatible with document and document_name)
        :param document_name: Filename of the document to capture (required if document is used)
        :param document: Document to capture itself (requires a document_name), must be base64 encoded
        :param depth: [Dangerous] Depth of the capture. If > 0, the URLs of the rendered document will be extracted and captured. It can take a very long time.
        :param browser: The prowser to use for the capture
        :param device_name: The name of the device, must be something Playwright knows
        :param user_agent: The user agent the browser will use for the capture
        :param proxy: SOCKS5 proxy to use for capturing
        :param socks5_dns_resolver: DNS resolver for to populate IPs in HAR when a capture is done via a socks5 proxy.
        :param general_timeout_in_sec: The capture will raise a timeout it it takes more than that time
        :param cookies: A list of cookies
        :param storage: A storage state from another capture
        :param headers: The headers to pass to the capture
        :param http_credentials: HTTP Credentials to pass to the capture
        :param geolocation: Geolocation of the browser to pass to the capture
        :param timezone_id: The timezone of the browser to pass to the capture
        :param locale: The locale of the browser to pass to the capture
        :param color_scheme: The prefered color scheme of the browser to pass to the capture
        :param java_script_enabled: If False, javascript will be disabled when rendering the page
        :param viewport: The viewport of the browser used for capturing
        :param referer: The referer URL for the capture
        :param rendered_hostname_only: If depth > 0: only capture URLs with the same hostname as the rendered page
        :param with_screenshot: If False, PlaywrightCapture won't take a screenshot of the rendered URL
        :param with_favicon: If True, PlaywrightCapture will attempt to get the potential favicons for the rendered URL. It is a dirty trick, see this issue for details: https://github.com/Lookyloo/PlaywrightCapture/issues/45
        :param with_trusted_timestamps: If True, PlaywrightCapture will trigger calls to a remote timestamp service. For that to work, this class must have been initialized with tt_settings. See RFC3161 for details: https://www.rfc-editor.org/rfc/rfc3161
        :param allow_tracking: If True, PlaywrightCapture will attempt to click through the cookie banners. It is totally dependent on the framework used on the website.
        :param remote_headfull: If True, the capture will be handled as a remote headfull session.
        :param headless: Whether to run the browser in headless mode. WARNING: requires to run in a graphical environment.
        :param max_retries: The maximum anount of retries for this capture
        :param init_script: A JavaScript that will be executed on each page of the capture.
        :param final_wait: The very last wait time, after the instrumentation is done.

        :param force: Force recapture, even if the same one was already done within the recapture_interval
        :param recapture_interval: The time the enqueued settings are kept in memory to avoid duplicates
        :param priority: The priority of the capture
        :param uuid: The preset priority of the capture, auto-generated if not present. Should only be used if the initiator couldn't enqueue immediately. NOTE: it will be overwritten if the UUID already exists.

        :return: UUID, reference to the capture for later use
        """
        if not settings:
            settings = {'depth': depth, 'rendered_hostname_only': rendered_hostname_only,
                        'url': url, 'document_name': document_name, 'document': document,
                        'browser': browser, 'device_name': device_name,
                        'user_agent': user_agent, 'proxy': proxy,
                        'socks5_dns_resolver': socks5_dns_resolver,
                        'general_timeout_in_sec': general_timeout_in_sec,
                        'cookies': cookies, 'storage': storage, 'headers': headers,
                        'http_credentials': http_credentials, 'geolocation': geolocation,
                        'timezone_id': timezone_id, 'locale': locale,
                        'color_scheme': color_scheme, 'java_script_enabled': java_script_enabled,
                        'viewport': viewport, 'referer': referer,
                        'with_screenshot': with_screenshot, 'with_favicon': with_favicon,
                        'with_trusted_timestamps': with_trusted_timestamps,
                        'allow_tracking': allow_tracking,
                        # Quietly force it to false if remote headed is not allowed.
                        'remote_headfull': remote_headfull if self.remote_headed_allowed else False,
                        'final_wait': final_wait,
                        # Quietly force it to true if headed is not allowed.
                        'headless': headless if self.headed_allowed else True,
                        'init_script': init_script,
                        'max_retries': max_retries}
        if isinstance(settings, dict):
            try:
                to_enqueue = CaptureSettings.model_validate(settings)
            except ValidationError as e:
                self.master_logger.warning(f'Unable to validate settings: {e}.')
                raise CaptureSettingsError('Invalid settings', e)
        else:
            to_enqueue = settings

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
            elif (self.get_capture_status(uuid) == CaptureStatus.DONE
                  and self.get_capture(uuid).get('error') is not None):
                # The UUID exists, the capture is done, but it has an error -> re-capture on the same UUID
                perma_uuid = uuid
            else:
                perma_uuid = str(uuid4())
                self.master_logger.warning(f'UUID {uuid} already exists, forcing a new one: {perma_uuid}.')
        else:
            perma_uuid = str(uuid4())

        if to_enqueue.with_trusted_timestamps and not self.tt_settings:
            self.master_logger.warning('Cannot trigger trusted timestamp, the remote timestamper service settings are missing.')
            to_enqueue.with_trusted_timestamps = False

        p = self.redis.pipeline()
        p.set(f'lacus:query_hash:{hash_query}', perma_uuid, nx=True, ex=recapture_interval)
        p.hset(f'lacus:capture_settings:{perma_uuid}', mapping=to_enqueue.redis_dump())
        p.expire(f'lacus:capture_settings:{perma_uuid}', self.max_capture_time * 100)
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
        if self.redis.zscore('lacus:ongoing', uuid) is not None:
            return CaptureStatus.ONGOING
        if self.redis.exists(f'lacus:capture_settings:{uuid}'):
            # we might have a race condition between when the UUID is popped out of lacus:to_capture,
            # and pushed in lacus:ongoing.
            # if that's the case, we wait for a sec and check lacus:ongoing again
            # If it's still not in ongoing, the UUID is broken and can be consdered unknown.
            # This key is removed anyway once the capture is done.
            max_checks = 10
            for i in range(max_checks):
                time.sleep(.1)
                if self.redis.zscore('lacus:to_capture', uuid) is not None:
                    # Could be re-added in that queue if the capture failed, but will be retried
                    return CaptureStatus.QUEUED
                if self.redis.zscore('lacus:ongoing', uuid) is not None:
                    # The capture is actually ongoing now
                    return CaptureStatus.ONGOING
            # The UUID is still no anywhere to be found, it's broken.
            self.redis.delete(f'lacus:capture_settings:{uuid}')
            return CaptureStatus.UNKNOWN
        if self.redis.exists(f'lacus:capture_results_hash:{uuid}'):
            return CaptureStatus.DONE
        if self.redis.exists(f'lacus:capture_results:{uuid}'):
            # TODO: remove in 1.8.* - old format used last in 1.6, and kept no more than 10H in redis
            return CaptureStatus.DONE
        return CaptureStatus.UNKNOWN

    def _get_session_manager(self, backend_type: str | None, redis: Redis[bytes]) -> SessionManager:
        if not backend_type:
            raise LacusCoreException('No backend type provided for the remote headed session.')
        if backend_type == XpraSessionManager.backend_type:
            return XpraSessionManager(redis)
        raise LacusCoreException(f'Unknown remote headed session backend: {backend_type}')

    async def _initialize_capture_context(self, capture: Capture, logger: LacusCoreLogAdapter, url: str) -> None:
        # make sure the initialization doesn't take too long
        init_timeout = max(self.max_capture_time / 10, 5)
        try:
            async with timeout(init_timeout) as initialize_timeout:
                await capture.initialize_context()
        except (TimeoutError, asyncio.exceptions.TimeoutError):
            timeout_expired(initialize_timeout, logger, 'Initializing took too long.')
            logger.warning(f'Initializing the context for {url} took longer than the allowed initialization timeout ({init_timeout}s)')
            raise RetryCapture(f'Initializing the context for {url} took longer than the allowed initialization timeout ({init_timeout}s)')

    async def _run_remote_headfull_capture(self, *, uuid: str, to_capture: CaptureSettings, url: str,
                                           logger: LacusCoreLogAdapter,
                                           stats_pipeline: Any, today: str) -> tuple[CaptureResponse, bool]:
        if not self.remote_headed_allowed or not self.remote_headed_session_manager:
            raise CaptureError('Remote Headfull captures are disabled by configuration.')
        if not self.headed_allowed:
            raise CaptureError('Remote Headfull captures require headed_allowed=True.')

        result: CaptureResponse = {}
        errors: list[str] = []

        session, metadata, backend_metadata = self.remote_headed_session_manager.start_session(session_name=uuid,
                                                                                               ttl=to_capture.general_timeout_in_sec if to_capture.general_timeout_in_sec is not None else 300)

        try:
            # NOTE: that shouldn't be needed. at this point, the capture should for sure be headless.
            to_capture.headless = False

            logger.debug(f'Initializing remote headed session for {url}')
            stats_pipeline.sadd(f'stats:{today}:captures', url)
            async with Capture(
                    loglevel=self.master_logger.getEffectiveLevel(),
                    uuid=uuid,
                    capture_settings=to_capture,
                    tt_settings=self.tt_settings,
                    env=dict(self.remote_headed_session_manager.get_capture_env(session))) as capture:
                await self._initialize_capture_context(capture, logger, url)

                # prepare and open the page the user will interact with.
                page = await capture.setup_page_capture(allow_tracking=to_capture.allow_tracking)
                await capture.open_page(page, url, errors, to_capture.referer)

                status = SessionStatus.READY
                metadata['status'] = int(status)
                self.remote_headed_session_manager.session_store.write(uuid, metadata,
                                                                       backend_metadata,
                                                                       expire_seconds=self.max_capture_time)

                expires_at_ts = int(session.expires_at.timestamp())
                poll_interval = 1.0
                finish_requested = False

                while not finish_requested:
                    # wait for either the timeout, or the user trigger
                    await asyncio.sleep(poll_interval)
                    now_ts = int(time.time())

                    if expires_at_ts and now_ts >= expires_at_ts:
                        # Got to the expiration time, trigger finish
                        status = SessionStatus.EXPIRED
                        self.request_finish(uuid)

                    # Update the metadata
                    if m := self.get_session_metadata(uuid):
                        metadata = m
                        if metadata.get('finish_requested'):
                            finish_requested = True
                    else:
                        # This shouldn't happen, but just in case (the metadata should exist)
                        status = SessionStatus.ERROR
                        result['error'] = 'Missing metadata, cannot finish capture'
                        raise RemoteHeadfullSessionError('Unable to process capture, missing metadata')

                try:
                    async with timeout(self.max_capture_time) as capture_timeout:
                        playwright_result = await capture.capture_page(
                            page=page,
                            current_page_only=True,
                            max_depth_capture_time=self.max_capture_time,
                            rendered_hostname_only=to_capture.rendered_hostname_only,
                            with_screenshot=to_capture.with_screenshot,
                            with_favicon=to_capture.with_favicon,
                            with_trusted_timestamps=to_capture.with_trusted_timestamps,
                        )
                except (TimeoutError, asyncio.exceptions.TimeoutError):
                    timeout_expired(capture_timeout, logger, 'Capture took too long.')
                    logger.warning(f'[RemoteHeaded] The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                    raise RetryCapture(f'[RemoteHeaded] The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                except PlaywrightCaptureException as e:
                    logger.warning(f'[RemoteHeaded] Unrecoverable exception during capture: {e}')
                    raise CaptureError(f'[RemoteHeaded] Unrecoverable exception during capture: {e}')
                except Exception as e:
                    logger.warning(f'[RemoteHeaded] Totally unrecoverable exception during capture: {e}')
                    raise CaptureError(f'[RemoteHeaded] Totally unrecoverable exception during capture: {e}')

                result = cast(CaptureResponse, playwright_result)
                status = SessionStatus.STOPPED
                if 'error' in result and 'error_name' in result:
                    # generate stats
                    if result['error_name'] is not None:
                        stats_pipeline.zincrby(f'stats:{today}:errors', 1, result['error_name'])
                should_retry = capture.should_retry
                return result, should_retry
        except RetryCapture as e:
            raise e
        except RemoteHeadfullSessionError as e:
            logger.warning(f'[RemoteHeaded] Unable to complete session: {e}')
            status = SessionStatus.ERROR
            raise CaptureError(f'[RemoteHeaded] Unable to complete remote headed session: {e}')
        except (PlaywrightCaptureException, InvalidPlaywrightParameter) as e:
            status = SessionStatus.ERROR
            logger.warning(f'[RemoteHeaded] Invalid parameters for the capture of {url} - {e}')
            raise CaptureError(f'[RemoteHeaded] Invalid parameters for the capture of {url} - {e}')
        except asyncio.CancelledError:
            status = SessionStatus.ERROR
            logger.warning(f'[RemoteHeaded] The capture of {url} has been cancelled.')
            # The capture can be canceled if it has been running for way too long.
            # We can give it another short.
            raise RetryCapture(f'[RemoteHeaded]  The capture of {url} has been cancelled.')
        except Exception as e:
            status = SessionStatus.ERROR
            logger.exception(f'[RemoteHeaded] Something went poorly {url} - {e}')
            raise CaptureError(f'[RemoteHeaded] Something went poorly {url} - {e}')
        finally:
            self.remote_headed_session_manager.stop_session(session, uuid, metadata,
                                                            status=status, expire_seconds=60)
            # NOTE: maybe move that somewhere else
            self.remote_headed_session_manager.cleanup_expired_sessions()

        raise CaptureError('[RemoteHeaded] Should never land there, but that capture failed badly.')

    async def _run_standard_capture(self, *, uuid: str, to_capture: CaptureSettings,
                                    url: str,
                                    logger: LacusCoreLogAdapter,
                                    stats_pipeline: Any, today: str) -> tuple[CaptureResponse, bool]:
        should_retry = False

        try:
            logger.debug(f'Capturing {url}')
            stats_pipeline.sadd(f'stats:{today}:captures', url)
            async with Capture(
                    loglevel=self.master_logger.getEffectiveLevel(),
                    uuid=uuid,
                    capture_settings=to_capture,
                    tt_settings=self.tt_settings) as capture:
                await self._initialize_capture_context(capture, logger, url)
                try:
                    async with timeout(self.max_capture_time) as capture_timeout:
                        playwright_result = await capture.capture_page(
                            url, referer=to_capture.referer,
                            depth=to_capture.depth,
                            rendered_hostname_only=to_capture.rendered_hostname_only,
                            with_screenshot=to_capture.with_screenshot,
                            with_favicon=to_capture.with_favicon,
                            allow_tracking=to_capture.allow_tracking,
                            with_trusted_timestamps=to_capture.with_trusted_timestamps,
                            max_depth_capture_time=self.max_capture_time,
                            final_wait=to_capture.final_wait)
                except (TimeoutError, asyncio.exceptions.TimeoutError):
                    timeout_expired(capture_timeout, logger, 'Capture took too long.')
                    logger.warning(f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                    raise RetryCapture(f'The capture of {url} took longer than the allowed max capture time ({self.max_capture_time}s)')
                except PlaywrightCaptureException as e:
                    logger.warning(f'Unrecoverable exception during capture: {e}')
                    raise CaptureError(f'Unrecoverable exception during capture: {e}')
                except Exception as e:
                    logger.warning(f'Totally unrecoverable exception during capture: {e}')
                    raise CaptureError(f'Totally unrecoverable exception during capture: {e}')
                result = cast(CaptureResponse, playwright_result)
                if 'error' in result and 'error_name' in result:
                    # generate stats
                    if result['error_name'] is not None:
                        stats_pipeline.zincrby(f'stats:{today}:errors', 1, result['error_name'])
                should_retry = capture.should_retry
                return result, should_retry
        except RetryCapture as e:
            logger.info('Attempting to retry.')
            raise e
        except (PlaywrightCaptureException, InvalidPlaywrightParameter) as e:
            logger.warning(f'Invalid parameters for the capture of {url} - {e}')
            raise CaptureError(f'Invalid parameters for the capture of {url} - {e}')
        except asyncio.CancelledError:
            logger.warning(f'The capture of {url} has been cancelled.')
            # The capture can be canceled if it has been running for way too long.
            # We can give it another short.
            raise RetryCapture(f'The capture of {url} has been cancelled.')
        except Exception as e:
            logger.exception(f'Something went poorly {url} - {e}')
            raise CaptureError(f'Something went poorly {url} - {e}')

        raise CaptureError('Should never land there, but that capture failed badly.')

    async def consume_queue(self, max_consume: int) -> AsyncIterator[Task[None]]:
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
            logger = LacusCoreLogAdapter(self.master_logger, {'uuid': uuid})
            yield task_logger.create_task(self._capture(uuid), name=uuid,
                                          logger=logger,
                                          message='Capture raised an uncaught exception')
            # Make sur the task starts.
            await asyncio.sleep(0.1)

    async def _capture(self, uuid: str) -> None:
        """Trigger a specific capture

        :param uuid: The UUID if the capture (given by enqueue)
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
            result: CaptureResponse = {}
            url: str = ''
            _to_capture_b = self.redis.hgetall(f'lacus:capture_settings:{uuid}')

            if not _to_capture_b:
                result = {'error': f'No capture settings for {uuid}'}
                raise CaptureError(f'No capture settings for {uuid}')

            _to_capture = {k.decode(): v.decode() for k, v in _to_capture_b.items()}
            try:
                to_capture = CaptureSettings.model_validate(_to_capture)
            except ValidationError as e:
                logger.warning(f'Settings invalid: {e}')
                raise CaptureSettingsError('Invalid settings', e)

            # NOTE: never retry remote headfull captures
            if to_capture.remote_headfull:
                max_retries = 0
            else:
                # If the class is initialized with max_retries below the one provided in the settings, we use the lowest value
                # NOTE: make sure the variable is initialized *before* we raise any RetryCapture
                max_retries = min([to_capture.max_retries, self.max_retries]) if to_capture.max_retries is not None else self.max_retries

            if to_capture.document:
                # we do not have a URL yet.
                document_as_bytes = b64decode(to_capture.document)
                tmp_f = NamedTemporaryFile(suffix=to_capture.document_name, delete=False)
                with open(tmp_f.name, "wb") as f:
                    f.write(document_as_bytes)
                url = f'file://{tmp_f.name}'
            elif to_capture.url:
                if to_capture.url.lower().startswith('file:') and self.only_global_lookups:
                    result = {'error': f'Not allowed to capture a file on disk: {url}'}
                    raise CaptureError(f'Not allowed to capture a file on disk: {url}')
                url = to_capture.url
            else:
                result = {'error': f'No valid URL to capture for {uuid} - {to_capture}'}
                raise CaptureError(f'No valid URL to capture for {uuid} - {to_capture}')

            try:
                splitted_url = urlsplit(url)
            except Exception as e:
                result = {'error': f'Invalid URL: {url} - {e}'}
                raise CaptureError(f'Invalid URL: {url} - {e}')
            if self.tor_proxy:
                # check if onion or forced
                if (to_capture.proxy == 'force_tor'  # if the proxy is set to "force_tor", we use the pre-configured tor proxy, regardless the URL, legacy feature.
                        or (not to_capture.proxy  # if the TLD is "onion", we use the pre-configured tor proxy
                            and splitted_url.netloc
                            and splitted_url.hostname
                            and splitted_url.hostname.split('.')[-1] == 'onion')):
                    if not _check_proxy_port_open(self.tor_proxy):
                        logger.critical(f'Unable to connect to the default tor proxy: {self.tor_proxy}')
                        raise CaptureError('The selected tor proxy is unreachable, unable to run the capture.')
                    to_capture.proxy = self.tor_proxy
                    logger.info('Using the default tor proxy.')
            if self.i2p_proxy:
                if (not to_capture.proxy  # if the TLD is "i2p", we use the pre-configured I2P proxy
                        and splitted_url.netloc
                        and splitted_url.hostname
                        and splitted_url.hostname.split('.')[-1] == 'i2p'):
                    if not _check_proxy_port_open(self.i2p_proxy):
                        logger.critical(f'Unable to connect to the default tor proxy: {self.i2p_proxy}')
                        raise CaptureError('The selected I2P proxy is unreachable, unable to run the capture.')
                    to_capture.proxy = self.i2p_proxy
                    logger.info('Using the default I2P proxy.')

            if self.only_global_lookups and not to_capture.proxy and splitted_url.scheme not in ['data', 'file']:
                # not relevant if we also have a proxy, or the thing to capture is a data URI or a file on disk
                if splitted_url.netloc:
                    if splitted_url.hostname and splitted_url.hostname.split('.')[-1] not in ['onion', 'i2p']:
                        ips_to_check = []
                        # check if the hostname is an IP
                        try:
                            _ip = ip_address(splitted_url.hostname)
                            ips_to_check.append(_ip)
                        except ValueError:
                            # not an IP, try resolving
                            try:
                                ips_to_check = await self.__get_ips(logger, splitted_url.hostname)
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

            if to_capture.remote_headfull:
                # NOTE: should_retry not used in the case of a remote headfull session.
                result, should_retry = await self._run_remote_headfull_capture(
                    uuid=uuid,
                    to_capture=to_capture,
                    url=url,
                    logger=logger,
                    stats_pipeline=stats_pipeline,
                    today=today,
                )
            else:
                result, should_retry = await self._run_standard_capture(
                    uuid=uuid,
                    to_capture=to_capture,
                    url=url,
                    logger=logger,
                    stats_pipeline=stats_pipeline,
                    today=today,
                )

            if should_retry:
                # PlaywrightCapture considers this capture elligible for a retry
                logger.info('PlaywrightCapture considers it elligible for a retry.')
                raise RetryCapture('PlaywrightCapture considers it elligible for a retry.')
            elif self.redis.exists(f'lacus:capture_retry:{uuid}'):
                # this is a retry that worked
                stats_pipeline.sadd(f'stats:{today}:retry_success', url)
        except RetryCapture as e:
            if not result and str(e):
                result = {'error': str(e)}
            if max_retries == 0:
                error_msg = result['error'] if result.get('error') else 'Unknown error'
                logger.info(f'Retries disabled for {url}: {error_msg}')
            else:
                # Check if we already re-tried this capture
                _current_retry = self.redis.get(f'lacus:capture_retry:{uuid}')
                if _current_retry is None:
                    # No retry yet
                    logger.debug(f'Retrying {url} for the first time.')
                    retry = True
                    self.redis.setex(f'lacus:capture_retry:{uuid}',
                                     self.max_capture_time * (max_retries + 100),
                                     max_retries - 1)
                else:
                    current_retry = int(_current_retry.decode())
                    if current_retry > 0:
                        logger.debug(f'Retrying {url} for the {max_retries - current_retry + 1} time.')
                        self.redis.decr(f'lacus:capture_retry:{uuid}')
                        retry = True
                    else:
                        error_msg = result['error'] if result.get('error') else 'Unknown error'
                        logger.info(f'Retried too many times {url}: {error_msg}')
                        stats_pipeline.sadd(f'stats:{today}:retry_failed', url)
        except CaptureError as e:
            if not result:
                result = {'error': str(e) if str(e) else "No result key, shouldn't happen"}
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
            #       and optionally re-added to lacus:to_capture if we want to retry it
            #
            # In order to have a consistent capture status, the capture UUID must either be in
            # lacus:ongoing (while ongoing), in lacus:to_capture (on retry), or the result stored (on success).
            # If the capture fails to be stored in valkey, we must also remove the capture settings
            # so it is not dangling there.

            try:
                if to_capture.document:
                    os.unlink(tmp_f.name)
            except UnboundLocalError:
                # Missing settings, the capture failed.
                pass

            if retry:
                if self.redis.zcard('lacus:to_capture') == 0:
                    # Just wait a little bit before retrying
                    await asyncio.sleep(random.randint(5, 10))
                p = self.redis.pipeline()
                p.zrem('lacus:ongoing', uuid)
                p.zincrby('lacus:to_capture', -1, uuid)
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
                            result = {'error': 'No result at all for the capture, Playwright failed.'}
                            self._store_capture_response(p, uuid, result)
                        p.delete(f'lacus:capture_settings:{uuid}')
                        p.zrem('lacus:ongoing', uuid)
                        p.execute()
                        break
                    except RedisConnectionError as e:
                        logger.warning(f'Unable to store capture result - Redis Connection Error: {e}')
                        retry_redis_error -= 1
                        await asyncio.sleep(random.randint(5, 10))
                else:
                    # Unrecoverable redis error, remove the capture settings
                    p = self.redis.pipeline()
                    p.delete(f'lacus:capture_settings:{uuid}')
                    p.zrem('lacus:ongoing', uuid)
                    result = {'error': "Unable to store the result of the capture in redis (probably a huge download)."}
                    self._store_capture_response(p, uuid, result)
                    p.execute()
                    stats_pipeline.zincrby(f'stats:{today}:errors', 1, 'Redis Connection')
                    logger.critical('Unable to connect to redis and to push the result of the capture.')

            # Expire stats in 10 days
            stats_expiry = timedelta(days=10)
            stats_pipeline.expire(f'stats:{today}:errors', stats_expiry)
            stats_pipeline.expire(f'stats:{today}:retry_failed', stats_expiry)
            stats_pipeline.expire(f'stats:{today}:retry_success', stats_expiry)
            stats_pipeline.expire(f'stats:{today}:captures', stats_expiry)
            stats_pipeline.execute()

    def _store_capture_response(self, pipeline: Redis, capture_uuid: str, results: CaptureResponse,   # type: ignore[type-arg]
                                root_key: str | None=None) -> None:
        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': capture_uuid})
        if root_key is None:
            root_key = f'lacus:capture_results_hash:{capture_uuid}'

        hash_to_set: dict[str, bytes | int | float | str] = {}
        try:
            if results.get('har'):
                hash_to_set['har'] = pickle.dumps(results['har'])
            if results.get('cookies'):
                hash_to_set['cookies'] = pickle.dumps(results['cookies'])
            if results.get('storage'):
                hash_to_set['storage'] = pickle.dumps(results['storage'])
            if results.get('potential_favicons'):
                hash_to_set['potential_favicons'] = pickle.dumps(results['potential_favicons'])
            if results.get('html') and results['html'] is not None:
                # Need to avoid unicode encode errors, and surrogates are not allowed
                hash_to_set['html'] = results['html'].encode('utf-8', 'surrogateescape')
            if results.get('frames') and results['frames'] is not None:
                hash_to_set['frames'] = pickle.dumps(results['frames'])
            if results.get('trusted_timestamps'):
                hash_to_set['trusted_timestamps'] = pickle.dumps(results['trusted_timestamps'])
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
        except Exception:
            logger.exception('Error while pickling the results.')
            results['error'] = "Error while saving the results (unable to pickle), please retry."

        direct_text_fields = {'last_redirected_url', 'error', 'error_name', 'html', 'downloaded_filename'}
        direct_bytes_fields = {'png', 'downloaded_file'}

        for key in results.keys():
            if key in ['har', 'cookies', 'storage', 'trusted_timestamps', 'potential_favicons',
                       'html', 'frames', 'children'] or not results.get(key):
                continue
            value = results[key]  # type: ignore[literal-required]
            # These entries can usually be stored directly, but Redis hash values
            # must be serialized to primitive wire types first.
            if key == 'status':
                hash_to_set[key] = int(value)
            elif key == 'runtime':
                hash_to_set[key] = float(value)
            elif key in direct_text_fields:
                hash_to_set[key] = str(value)
            elif key in direct_bytes_fields:
                hash_to_set[key] = bytes(value) if isinstance(value, bytearray | memoryview) else value
            else:
                logger.warning(f'Unexpected capture response type for {key}, serializing defensively.')
                hash_to_set[key] = pickle.dumps(value)
        if hash_to_set:
            pipeline.hset(root_key, mapping=hash_to_set)  # type: ignore[arg-type]
            # Make sure the key expires
            pipeline.expire(root_key, self.expire_results)
        else:
            logger.critical(f'Nothing to store (Hash: {hash_to_set}) for {root_key}')

    def _get_capture_response(self, capture_uuid: str, root_key: str | None=None) -> CaptureResponse | None:
        logger = LacusCoreLogAdapter(self.master_logger, {'uuid': capture_uuid})
        if root_key is None:
            root_key = f'lacus:capture_results_hash:{capture_uuid}'

        # New format and capture done

        to_return: CaptureResponse = {}
        for key, value in self.redis.hgetall(root_key).items():
            if key == b'har':
                to_return['har'] = pickle.loads(value)
            elif key == b'cookies':
                to_return['cookies'] = pickle.loads(value)
            elif key == b'storage':
                to_return['storage'] = pickle.loads(value)
            elif key == b'potential_favicons':
                to_return['potential_favicons'] = pickle.loads(value)
            elif key == b'trusted_timestamps':
                to_return['trusted_timestamps'] = pickle.loads(value)
            elif key == b'frames':
                to_return['frames'] = pickle.loads(value)
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
                logger.critical(f'Unexpected key in response: {key.decode()} - {value.decode()}')
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

    async def __get_ips(self, logger: LacusCoreLogAdapter, hostname: str) -> list[IPv4Address | IPv6Address]:
        # We need to use dnspython for resolving because socket.getaddrinfo will sometimes be stuck for ~10s
        # It is happening when the error code is NoAnswer
        resolved_ips = []
        max_timeout_retries = 3
        _current_retries = 0
        while _current_retries < max_timeout_retries:
            _current_retries += 1
            try:
                answers_a = await self.dnsresolver.resolve(hostname, 'A')
                resolved_ips += [ip_address(str(answer)) for answer in answers_a]
            except DNSTimeout as e:
                if _current_retries < max_timeout_retries:
                    logger.info(f'DNS Timeout for "{hostname}" (A record), retrying.')
                    await asyncio.sleep(1)
                    continue
                raise e
            except DNSException as e:
                logger.debug(f'No A record for "{hostname}": {e}')
            break

        _current_retries = 0
        while _current_retries < max_timeout_retries:
            _current_retries += 1
            try:
                answers_aaaa = await self.dnsresolver.resolve(hostname, 'AAAA')
                resolved_ips += [ip_address(str(answer)) for answer in answers_aaaa]
            except DNSTimeout as e:
                if _current_retries < max_timeout_retries:
                    logger.info(f'DNS Timeout for "{hostname}" (AAAA record), retrying.')
                    await asyncio.sleep(1)
                    continue
                raise e
            except DNSException as e:
                logger.debug(f'No AAAA record for "{hostname}": {e}')
            break
        return resolved_ips

    def request_finish(self, uuid: str) -> bool:
        """Mark a remote headfull session as ready for final capture.

        Returns the updated metadata, or None if no session exists.
        """
        if not self.remote_headed_allowed or not self.remote_headed_session_manager:
            raise RemoteHeadfullSessionError('Remote headfull captures are disabled by configuration.')
        return self.remote_headed_session_manager.session_store.request_finish(uuid)

    def get_session_metadata(self, uuid: str) -> SessionMetadata | None:
        """Return public session metadata for a capture UUID, or None if no session exists."""
        if not self.remote_headed_allowed or not self.remote_headed_session_manager:
            raise RemoteHeadfullSessionError('Remote headfull captures are disabled by configuration.')
        record = self.remote_headed_session_manager.session_store.read(uuid)
        if not record:
            return None
        return cast(SessionMetadata, dict(record.metadata))

    def get_session_backend_metadata(self, uuid: str) -> dict[str, Any] | None:
        """Return backend-specific metadata for trusted session transport callers."""
        if not self.remote_headed_allowed or not self.remote_headed_session_manager:
            raise RemoteHeadfullSessionError('Remote headfull captures are disabled by configuration.')
        record = self.remote_headed_session_manager.session_store.read(uuid)
        if not record:
            return None
        return dict(record.backend_metadata)
