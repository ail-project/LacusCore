#!/usr/bin/env python3

import ipaddress
import hashlib
import json
import logging
import os
import pickle
import re
import socket
import unicodedata

from base64 import b64decode, b64encode
from enum import IntEnum, unique
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional, Union, Dict, List, Any, TypedDict, overload, Tuple, cast
from uuid import uuid4
from urllib.parse import urlsplit

from defang import refang  # type: ignore
from playwrightcapture import Capture, PlaywrightCaptureException
from playwrightcapture.capture import CaptureResponse as PlaywrightCaptureResponse
from redis import Redis
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


@unique
class CaptureStatus(IntEnum):
    UNKNOWN = -1
    QUEUED = 0
    DONE = 1
    ONGOING = 2


class CaptureResponse(TypedDict, total=False):

    status: int
    last_redirected_url: str
    har: Optional[Dict[str, Any]]
    cookies: Optional[List[Dict[str, str]]]
    error: Optional[str]
    html: Optional[str]
    png: Optional[bytes]
    downloaded_filename: Optional[str]
    downloaded_file: Optional[bytes]
    children: Optional[List[Any]]


class CaptureResponseJson(TypedDict, total=False):

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


class CaptureSettings(TypedDict, total=False):

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
    http_credentials: Optional[Dict[str, int]]
    viewport: Optional[Dict[str, int]]
    referer: Optional[str]

    depth: Optional[int]
    rendered_hostname_only: bool  # Note: only used if depth is > 0


def _json_encode(obj: Union[bytes]) -> str:
    if isinstance(obj, bytes):
        return b64encode(obj).decode()


class LacusCore():

    def __init__(self, redis_connector: Redis, tor_proxy: Optional[str]=None, only_global_lookups: bool=True) -> None:
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.logger.setLevel('INFO')

        self.redis = redis_connector
        self.tor_proxy = tor_proxy
        self.only_global_lookups = only_global_lookups

    def check_redis_up(self):
        return self.redis.ping()

    def enqueue(self, *, url: Optional[str]=None,
                document_name: Optional[str]=None, document: Optional[str]=None,
                depth: int=0,
                browser: Optional[BROWSER]=None, device_name: Optional[str]=None,
                user_agent: Optional[str]=None,
                proxy: Optional[Union[str, Dict[str, str]]]=None,
                general_timeout_in_sec: Optional[int]=None,
                cookies: Optional[List[Dict[str, Any]]]=None,
                headers: Optional[Union[str, Dict[str, str]]]=None,
                http_credentials: Optional[Dict[str, int]]=None,
                viewport: Optional[Dict[str, int]]=None,
                referer: Optional[str]=None,
                rendered_hostname_only: bool=True,
                force: bool=False,
                recapture_interval: int=300,
                priority: int=0
                ) -> str:
        to_enqueue: CaptureSettings = {'depth': depth, 'rendered_hostname_only': rendered_hostname_only}
        if url:
            to_enqueue['url'] = url
        elif document_name and document:
            to_enqueue['document_name'] = _secure_filename(document_name)
            to_enqueue['document'] = document
        else:
            raise Exception('Needs either a URL or a document_name *and* a document.')
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
        if viewport:
            to_enqueue['viewport'] = viewport
        if referer:
            to_enqueue['referer'] = referer

        if not force:
            hash_query = hashlib.sha512(pickle.dumps(to_enqueue)).hexdigest()
            if (existing_uuid := self.redis.get(f'lacus:query_hash:{hash_query}')):
                return existing_uuid.decode()
        perma_uuid = str(uuid4())

        mapping_capture: Dict[str, Union[bytes, float, int, str]] = {}
        for key, value in to_enqueue.items():
            if isinstance(value, bool):
                mapping_capture[key] = 1 if value else 0
            elif isinstance(value, (list, dict)):
                if value:
                    mapping_capture[key] = json.dumps(value)
            elif value is not None:
                mapping_capture[key] = value  # type: ignore

        p = self.redis.pipeline()
        p.set(f'lacus:query_hash:{hash_query}', perma_uuid, nx=True, ex=recapture_interval)
        p.hset(f'lacus:capture_settings:{perma_uuid}', mapping=mapping_capture)  # type: ignore
        p.zadd('lacus:to_capture', {perma_uuid: priority})
        p.execute()
        print(perma_uuid)
        return perma_uuid

    def _decode_response(self, capture: CaptureResponseJson) -> CaptureResponse:
        decoded_capture = cast(CaptureResponse, capture)
        if capture.get('png') and capture['png']:
            decoded_capture['png'] = b64decode(capture['png'])
        if capture.get('downloaded_file') and capture['downloaded_file']:
            decoded_capture['downloaded_file'] = b64decode(capture['downloaded_file'])
        if capture.get('children') and capture['children']:
            for child in capture['children']:
                child = self._decode_response(child)
        return decoded_capture

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[True]=True) -> CaptureResponse:
        ...

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[False]) -> CaptureResponseJson:
        ...

    def get_capture(self, uuid: str, *, decode: bool=False) -> Union[CaptureResponse, CaptureResponseJson]:
        to_return: CaptureResponseJson = {'status': CaptureStatus.UNKNOWN}
        if self.redis.zscore('lacus:to_capture', uuid):
            to_return['status'] = CaptureStatus.QUEUED
        elif self.redis.sismember('lacus:ongoing', uuid):
            to_return['status'] = CaptureStatus.ONGOING
        elif response := self.redis.get(f'lacus:capture_results:{uuid}'):
            to_return['status'] = CaptureStatus.DONE
            to_return.update(json.loads(response))
            if decode:
                return self._decode_response(to_return)
        return to_return

    def get_capture_status(self, uuid: str) -> CaptureStatus:
        if self.redis.zscore('lacus:to_capture', uuid):
            return CaptureStatus.QUEUED
        elif self.redis.sismember('lacus:ongoing', uuid):
            return CaptureStatus.ONGOING
        elif self.redis.exists(f'lacus:capture_results:{uuid}'):
            return CaptureStatus.DONE
        return CaptureStatus.UNKNOWN

    async def consume_queue(self) -> Optional[str]:
        value: List[Tuple[bytes, float]] = self.redis.zpopmax('lacus:to_capture')
        if not value or not value[0]:
            return None
        uuid: str = value[0][0].decode()
        await self.capture(uuid)
        return uuid

    async def capture(self, uuid: str):
        if self.redis.sismember('lacus:ongoing', uuid):
            # the capture is ongoing
            return

        p = self.redis.pipeline()
        p.zrem('lacus:to_capture', uuid)  # In case the capture is triggered manually
        p.sadd('lacus:ongoing', uuid)
        p.execute()
        try:
            setting_keys = ['depth', 'rendered_hostname_only', 'url', 'document_name',
                            'document', 'browser', 'device_name', 'user_agent', 'proxy',
                            'general_timeout_in_sec', 'cookies', 'headers', 'http_credentials',
                            'viewport', 'referer']
            result: PlaywrightCaptureResponse
            to_capture = {}
            for k, v in zip(setting_keys, self.redis.hmget(f'lacus:capture_settings:{uuid}', setting_keys)):
                if v is not None:
                    to_capture[k] = v if k in ['document'] else v.decode()  # Do not decode the document
            if not to_capture:
                result = {'error': f'No capture settings for {uuid}.'}
                raise CaptureError

            if to_capture.get('document'):
                # we do not have a URL yet.
                document_name = Path(to_capture['document_name']).name
                tmp_f = NamedTemporaryFile(suffix=document_name, delete=False)
                with open(tmp_f.name, "wb") as f:
                    f.write(to_capture['document'])
                url: str = f'file://{tmp_f.name}'
            elif to_capture.get('url'):
                url = to_capture['url'].strip()
                url = refang(url)  # In case we get a defanged url at this stage.
                if not url.startswith('data') and not url.startswith('http') and not url.startswith('file'):
                    url = f'http://{url}'
            else:
                result = {'error': f'No valid URL to capture for {uuid}.'}
                raise CaptureError

            splitted_url = urlsplit(url)
            if self.only_global_lookups and splitted_url.scheme not in ['data', 'file']:
                if splitted_url.netloc:
                    if splitted_url.hostname and splitted_url.hostname.split('.')[-1] != 'onion':
                        try:
                            ip = socket.gethostbyname(splitted_url.hostname)
                        except socket.gaierror:
                            self.logger.info('Name or service not known')
                            result = {'error': f'Unable to resolve {splitted_url.hostname}.'}
                            raise CaptureError
                        if not ipaddress.ip_address(ip).is_global:
                            result = {'error': 'Capturing ressources on private IPs is disabled.'}
                            raise CaptureError
                else:
                    result = {'error': 'Unable to find hostname or IP in the query.'}
                    raise CaptureError

            proxy = to_capture.get('proxy')
            # check if onion
            if (not proxy and splitted_url.netloc and splitted_url.hostname
                    and splitted_url.hostname.split('.')[-1] == 'onion'):
                proxy = self.tor_proxy

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
                self.logger.info(f'Capturing {url}')
                async with Capture(browser=browser_engine,
                                   device_name=to_capture.get('device_name'),
                                   proxy=proxy) as capture:
                    # required by Mypy: https://github.com/python/mypy/issues/3004
                    capture.headers = to_capture.get('headers')  # type: ignore
                    if to_capture.get('cookies'):
                        capture.cookies = json.loads(to_capture.get('cookies'))  # type: ignore
                    capture.viewport = to_capture.get('viewport')
                    capture.user_agent = to_capture.get('user_agent')  # type: ignore
                    await capture.initialize_context()
                    result = await capture.capture_page(url, referer=to_capture.get('referer'))
            except PlaywrightCaptureException as e:
                self.logger.exception(f'Invalid parameters for the capture of {url} - {e}')
                result = {'error': 'Invalid parameters for the capture of {url} - {e}'}
                raise CaptureError
            except Exception as e:
                self.logger.exception(f'Something went poorly {url} - {e}')
                result = {'error': f'Something went poorly {url} - {e}'}
                raise CaptureError

        except CaptureError:
            self.logger.warning(f'Unable to capture {url} - {uuid}: {result["error"]}')
        else:
            self.logger.info(f'Successfully captured {url} - {uuid}')
        finally:

            if to_capture.get('document'):
                os.unlink(tmp_f.name)

            p = self.redis.pipeline()
            p.setex(f'lacus:capture_results:{uuid}', 36000, json.dumps(result, default=_json_encode))
            p.delete(f'lacus:capture_settings:{uuid}')
            p.srem('lacus:ongoing', uuid)
            p.execute()
