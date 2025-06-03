#!/usr/bin/env python3

from __future__ import annotations

import json
import sys

from datetime import datetime, timedelta
from enum import IntEnum, unique
from logging import LoggerAdapter
from typing import Any, Literal
from collections.abc import MutableMapping, Mapping

from defang import refang
from pydantic import BaseModel, field_validator, model_validator, ValidationError
from pydantic_core import from_json

from playwrightcapture.capture import CaptureResponse as PlaywrightCaptureResponse


if sys.version_info < (3, 12):
    from typing_extensions import TypedDict

    class Cookie(TypedDict, total=False):
        name: str
        value: str
        domain: str
        path: str
        expires: float
        httpOnly: bool
        secure: bool
        sameSite: Literal["Lax", "None", "Strict"]
else:
    from typing import TypedDict
    from playwright._impl._api_structures import Cookie  # , StorageState


class LacusCoreException(Exception):
    pass


class CaptureError(LacusCoreException):
    pass


class RetryCapture(LacusCoreException):
    pass


class CaptureSettingsError(LacusCoreException):
    '''Can handle Pydantic validation errors'''

    def __init__(self, message: str, pydantic_validation_errors: ValidationError | None=None) -> None:
        super().__init__(message)
        self.pydantic_validation_errors = pydantic_validation_errors


class LacusCoreLogAdapter(LoggerAdapter):  # type: ignore[type-arg]
    """
    Prepend log entry with the UUID of the capture
    """
    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        if self.extra:
            return '[{}] {}'.format(self.extra['uuid'], msg), kwargs
        return msg, kwargs


@unique
class CaptureStatus(IntEnum):
    '''The status of the capture'''
    UNKNOWN = -1
    QUEUED = 0
    DONE = 1
    ONGOING = 2


class CaptureResponse(PlaywrightCaptureResponse, TypedDict, total=False):
    '''A capture made by Lacus. With the base64 encoded image and downloaded file decoded to bytes.'''

    # Need to make sure the type is what's expected down the line
    children: list[CaptureResponse] | None  # type: ignore[misc]

    status: int
    runtime: float | None


class CaptureResponseJson(TypedDict, total=False):
    '''A capture made by Lacus. With the base64 encoded image and downloaded file *not* decoded.'''

    status: int
    last_redirected_url: str | None
    har: dict[str, Any] | None
    cookies: list[Cookie] | None
    # NOTE: should be that, but StorageState doesn't define the indexeddb
    # storage: StorageState | None
    storage: dict[str, Any] | None
    error: str | None
    html: str | None
    png: str | None
    downloaded_filename: str | None
    downloaded_file: str | None
    children: list[CaptureResponseJson] | None
    runtime: float | None
    potential_favicons: list[str] | None


class CaptureSettings(BaseModel):
    '''The capture settings that can be passed to Lacus.'''

    url: str | None = None
    document_name: str | None = None
    document: str | None = None
    browser: Literal['chromium', 'firefox', 'webkit'] | None = None
    device_name: str | None = None
    user_agent: str | None = None
    proxy: str | dict[str, str] | None = None
    socks5_dns_resolver: str | list[str] | None = None
    general_timeout_in_sec: int | None = None
    cookies: list[Cookie] | None = None
    # NOTE: should be that, but StorageState doesn't define the indexeddb
    # storage: StorageState | None = None
    storage: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    http_credentials: dict[str, str] | None = None
    geolocation: dict[str, str | int | float] | None = None
    timezone_id: str | None = None
    locale: str | None = None
    color_scheme: Literal['dark', 'light', 'no-preference', 'null'] | None = None
    java_script_enabled: bool = True
    viewport: dict[str, str | int] | None = None
    referer: str | None = None
    with_screenshot: bool = True
    with_favicon: bool = True
    allow_tracking: bool = False
    headless: bool = True
    force: bool = False
    recapture_interval: int = 300
    priority: int = 0
    max_retries: int | None = None
    uuid: str | None = None

    depth: int = 0
    rendered_hostname_only: bool = True  # Note: only used if depth is > 0

    @model_validator(mode="before")
    @classmethod
    def empty_str_to_none(cls, data: Any) -> dict[str, Any] | Any:
        if isinstance(data, dict):
            # Make sure all the strings are stripped, and None if empty.
            to_return: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    if v_stripped := v.strip():
                        if v_stripped[0] in ['{', '[']:
                            to_return[k] = from_json(v_stripped)
                        else:
                            to_return[k] = v_stripped
                else:
                    to_return[k] = v
            return to_return
        return data

    @model_validator(mode='after')
    def check_capture_element(self) -> CaptureSettings:
        if self.document_name and not self.document:
            raise CaptureSettingsError('You must provide a document if you provide a document name')
        if self.document and not self.document_name:
            raise CaptureSettingsError('You must provide a document name if you provide a document')

        if self.url and (self.document or self.document_name):
            raise CaptureSettingsError('You cannot provide both a URL and a document to capture')
        if not self.url and not (self.document and self.document_name):
            raise CaptureSettingsError('You must provide either a URL or a document to capture')
        return self

    @field_validator('url', mode='after')
    @classmethod
    def load_url(cls, url: str | None) -> str | None:
        if isinstance(url, str):
            #  In case we get a defanged url at this stage.
            _url = refang(url)  # type: ignore[no-untyped-call]
            if (not _url.lower().startswith('data:')
                    and not _url.lower().startswith('http:')
                    and not _url.lower().startswith('https:')
                    and not _url.lower().startswith('file:')):
                _url = f'http://{_url}'
            return _url
        return url

    @field_validator('document_name', mode='after')
    @classmethod
    def load_document_name(cls, document_name: str | None) -> str | None:
        if isinstance(document_name, str):
            if '.' not in document_name:
                # The browser will simply display the file as text if there is no extension.
                # Just add HTML as a fallback, as it will be the most comon one.
                document_name = f'{document_name}.html'
            return document_name
        return None

    @field_validator('browser', mode='before')
    @classmethod
    def load_browser(cls, browser: Any) -> str | None:
        if isinstance(browser, str) and browser in ['chromium', 'firefox', 'webkit']:
            return browser
        # There are old captures where the browser is not a playwright browser name, so we ignore it.
        return None

    @field_validator('proxy', mode='before')
    @classmethod
    def load_proxy_json(cls, proxy: Any) -> str | dict[str, str] | None:
        if not proxy:
            return None
        if isinstance(proxy, str):
            # Just the proxy
            return proxy
        elif isinstance(proxy, dict):
            return proxy
        return None

    @field_validator('cookies', mode='before')
    @classmethod
    def load_cookies_json(cls, cookies: Any) -> list[dict[str, Any]] | None:

        def __prepare_cookie(cookie: dict[str, Any]) -> dict[str, str | float | bool]:
            if len(cookie) == 1:
                # {'name': 'value'} => {'name': 'name', 'value': 'value'}
                name, value = cookie.popitem()
                if name and value:
                    cookie = {'name': name, 'value': value}
            if not cookie.get('name') or not cookie.get('value'):
                # invalid cookie, ignoring
                return {}

            if 'expires' in cookie and isinstance(cookie['expires'], str):
                # Make it a float, as expected by Playwright
                try:
                    cookie['expires'] = datetime.fromisoformat(cookie['expires']).timestamp()
                except ValueError:
                    # if it ends with a Z, it fails in python < 3.12
                    # And we don't really care.
                    # make it expire 10 days from now
                    cookie['expires'] = (datetime.now() + timedelta(days=10)).timestamp()
            return cookie

        if not cookies:
            return None
        if isinstance(cookies, str):
            # might be a json dump, try to load it and ignore otherwise
            try:
                cookies = json.loads(cookies)
            except json.JSONDecodeError as e:
                print(e)
                # Cookies are invalid, ignoring.
                return None
        if isinstance(cookies, dict):
            # might be a single cookie in the format name: value, make it a list
            cookies = [cookies]
        if isinstance(cookies, list):
            # make sure the cookies are in the right format
            to_return = []
            for cookie in cookies:
                if isinstance(cookie, dict):
                    to_return.append(__prepare_cookie(cookie))
            return to_return
        return None

    @field_validator('storage', mode='before')
    @classmethod
    def load_storage_json(cls, storage: Any) -> dict[str, Any] | None:
        """That's the storage as exported from Playwright:
            https://playwright.dev/python/docs/api/class-browsercontext#browser-context-storage-state
        """
        if not storage:
            return None
        if isinstance(storage, str):
            # might be a json dump, try to load it and ignore otherwise
            try:
                storage = json.loads(storage)
            except json.JSONDecodeError:
                # storage is invalid, ignoring.
                return None
        if isinstance(storage, dict) and 'cookies' in storage and 'origins' in storage:
            return storage
        return None

    @field_validator('headers', mode='before')
    @classmethod
    def load_headers_json(cls, headers: Any) -> dict[str, str] | None:
        if not headers:
            return None
        if isinstance(headers, str):
            # make it a dict
            new_headers = {}
            for header_line in headers.splitlines():
                if header_line and ':' in header_line:
                    splitted = header_line.split(':', 1)
                    if splitted and len(splitted) == 2:
                        header, h_value = splitted
                        if header.strip() and h_value.strip():
                            new_headers[header.strip()] = h_value.strip()
            return new_headers
        elif isinstance(headers, dict):
            return headers
        return None

    @field_validator('http_credentials', mode='before')
    @classmethod
    def load_http_creds_json(cls, http_credentials: Any) -> dict[str, str] | None:
        if not http_credentials:
            return None
        if isinstance(http_credentials, str):
            # ignore
            return None
        elif isinstance(http_credentials, dict):
            return http_credentials
        return None

    @field_validator('http_credentials', mode='after')
    @classmethod
    def check_http_creds(cls, http_credentials: dict[str, str] | None) -> dict[str, str] | None:
        if not http_credentials:
            return None
        if 'username' in http_credentials and 'password' in http_credentials:
            return http_credentials
        raise CaptureSettingsError(f'HTTP credentials must have a username and a password: {http_credentials}')

    @field_validator('geolocation', mode='before')
    @classmethod
    def load_geolocation_json(cls, geolocation: Any) -> dict[str, float] | None:
        if not geolocation:
            return None
        if isinstance(geolocation, str):
            # ignore
            return None
        elif isinstance(geolocation, dict):
            return geolocation
        return None

    @field_validator('geolocation', mode='after')
    @classmethod
    def check_geolocation(cls, geolocation: dict[str, float] | None) -> dict[str, float] | None:
        if not geolocation:
            return None
        if 'latitude' in geolocation and 'longitude' in geolocation:
            return geolocation
        raise CaptureSettingsError(f'A geolocation must have a latitude and a longitude: {geolocation}')

    @field_validator('viewport', mode='before')
    @classmethod
    def load_viewport_json(cls, viewport: Any) -> dict[str, int] | None:
        if not viewport:
            return None
        if isinstance(viewport, str):
            # ignore
            return None
        elif isinstance(viewport, dict):
            return viewport
        return None

    @field_validator('viewport', mode='after')
    @classmethod
    def check_viewport(cls, viewport: dict[str, int] | None) -> dict[str, int] | None:
        if not viewport:
            return None
        if 'width' in viewport and 'height' in viewport:
            return viewport
        raise CaptureSettingsError(f'A viewport must have a width and a height: {viewport}')

    def redis_dump(self) -> Mapping[str | bytes, bytes | float | int | str]:
        mapping_capture: dict[str | bytes, bytes | float | int | str] = {}
        for key, value in dict(self).items():
            if value is None:
                continue
            if isinstance(value, bool):
                mapping_capture[key] = 1 if value else 0
            elif isinstance(value, (list, dict)):
                if value:
                    mapping_capture[key] = json.dumps(value)
            elif isinstance(value, (bytes, float, int, str)) and value not in ['', b'']:  # we're ok with 0 for example
                mapping_capture[key] = value
        return mapping_capture
