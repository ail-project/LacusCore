#!/usr/bin/env python3

import ipaddress
import hashlib
import json
import logging
import os
import socket

from base64 import b64decode, b64encode
from enum import IntEnum, unique
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Optional, Union, Dict, List, Any, TypedDict, overload
from uuid import uuid4
from urllib.parse import urlsplit

from defang import refang  # type: ignore
from playwrightcapture import Capture, PlaywrightCaptureException
from playwrightcapture.capture import CaptureResponse as PlaywrightCaptureResponse
from redis import Redis
from ua_parser import user_agent_parser  # type: ignore

BROWSER = Literal['chromium', 'firefox', 'webkit']


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
    headers: Optional[Dict[str, str]]
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
                headers: Optional[Dict[str, str]]=None,
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
            to_enqueue['document_name'] = document_name
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
            hash_query = hashlib.sha512(json.dumps(to_enqueue).encode()).hexdigest()
            if (existing_uuid := self.redis.get(f'query_hash:{hash_query}')):
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

        self.redis.set(f'query_hash:{hash_query}', perma_uuid, nx=True, ex=recapture_interval)
        self.redis.hset(perma_uuid, mapping=mapping_capture)  # type: ignore
        self.redis.zadd('to_capture', {perma_uuid: priority})
        return perma_uuid

    def _decode_response(self, capture: CaptureResponseJson) -> CaptureResponse:
        if capture.get('png'):
            capture['png'] = b64decode(capture['png'])  # type: ignore
        if capture.get('downloaded_file'):
            capture['downloaded_file'] = b64decode(capture['downloaded_file'])  # type: ignore
        if capture.get('children') and capture['children']:
            for child in capture['children']:
                child = self._decode_response(child)
        return capture  # type: ignore

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[True]=True) -> CaptureResponse:
        ...

    @overload
    def get_capture(self, uuid: str, *, decode: Literal[False]) -> CaptureResponseJson:
        ...

    def get_capture(self, uuid: str, *, decode: bool=False) -> Union[CaptureResponse, CaptureResponseJson]:
        if self.redis.zscore('to_capture', uuid):
            return {'status': CaptureStatus.QUEUED.value}  # type: ignore
        elif self.redis.sismember('ongoing', uuid):
            return {'status': CaptureStatus.ONGOING.value}  # type: ignore
        elif response := self.redis.get(uuid):
            capture = json.loads(response)
            capture['status'] = CaptureStatus.DONE.value
            if decode:
                return self._decode_response(capture)
            return capture
        else:
            return {'status': CaptureStatus.UNKNOWN.value}  # type: ignore

    async def capture(self, uuid: str) -> bytes:
        success = True
        if not self.redis.zscore('to_capture', uuid):
            # the capture was already done
            return self.redis.get(uuid)  # type: ignore
        p = self.redis.pipeline()
        p.zrem('to_capture', uuid)  # In case the capture is triggered manually
        p.sadd('ongoing', uuid)
        p.execute()
        to_capture = self.redis.hgetall(uuid)
        result: PlaywrightCaptureResponse

        if to_capture.get(b'document'):
            # we do not have a URL yet.
            document_name = Path(to_capture[b'document_name'].decode()).name
            tmp_f = NamedTemporaryFile(suffix=document_name, delete=False)
            with open(tmp_f.name, "wb") as f:
                f.write(to_capture[b'document'])
            url: str = f'file://{tmp_f.name}'
        elif to_capture.get(b'url'):
            url = to_capture[b'url'].decode().strip()
            url = refang(url)  # In case we get a defanged url at this stage.
            if not url.startswith('data') and not url.startswith('http') and not url.startswith('file'):
                url = f'http://{url}'
        else:
            raise Exception('No valid URL to capture')

        splitted_url = urlsplit(url)
        if self.only_global_lookups and splitted_url.scheme not in ['data', 'file']:
            if splitted_url.netloc:
                if splitted_url.hostname and splitted_url.hostname.split('.')[-1] != 'onion':
                    try:
                        ip = socket.gethostbyname(splitted_url.hostname)
                    except socket.gaierror:
                        self.logger.info('Name or service not known')
                        success = False
                        result = {'error': f'Unable to resolve {splitted_url.hostname}.'}
                    if not ipaddress.ip_address(ip).is_global:
                        success = False
                        result = {'error': 'Capturing ressources on private IPs is disabled.'}
            else:
                success = False
                result = {'error': 'Unable to find hostname or IP in the query.'}

        proxy = to_capture.get(b'proxy')
        # check if onion
        if (not proxy and splitted_url.netloc and splitted_url.hostname
                and splitted_url.hostname.split('.')[-1] == 'onion'):
            proxy = self.tor_proxy

        browser_engine: BROWSER = "chromium"
        if to_capture.get(b'user_agent'):
            parsed_string = user_agent_parser.ParseUserAgent(to_capture.get(b'user_agent'))
            browser_family = parsed_string['family'].lower()
            if browser_family.startswith('chrom'):
                browser_engine = 'chromium'
            elif browser_family.startswith('firefox'):
                browser_engine = 'firefox'
            else:
                browser_engine = 'webkit'

        if success:
            try:
                async with Capture(browser=browser_engine,
                                   device_name=to_capture.get(b'device_name'),
                                   proxy=proxy) as capture:
                    # required by Mypy: https://github.com/python/mypy/issues/3004
                    capture.headers = to_capture.get(b'headers')  # type: ignore
                    if to_capture.get(b'cookies_pseudofile'):
                        capture.cookies = json.loads(to_capture.get(b'cookies_pseudofile'))  # type: ignore
                    capture.viewport = to_capture.get(b'viewport')
                    capture.user_agent = to_capture.get(b'user_agent')  # type: ignore
                    await capture.initialize_context()
                    result = await capture.capture_page(url, referer=to_capture.get(b'referer'))
            except PlaywrightCaptureException as e:
                self.logger.exception(f'Invalid parameters for the capture of {url} - {e}')
                success = False
                result = {'error': 'Invalid parameters for the capture of {url} - {e}'}
            except Exception as e:
                self.logger.exception(f'Something went poorly {url} - {e}')
                success = False
                result = {'error': f'Something went poorly {url} - {e}'}

        if to_capture.get(b'document'):
            os.unlink(tmp_f.name)

        # Overwrite the capture params
        json_capture = json.dumps(result, default=_json_encode).encode()
        self.redis.setex(uuid, 36000, json_capture)
        self.redis.srem('ongoing', uuid)

        if success:
            self.logger.info(f'Successfully captured {url} - {uuid}')
        else:
            self.logger.warning(f'Unable to capture {url} - {uuid}: {result["error"]}')
        return json_capture
