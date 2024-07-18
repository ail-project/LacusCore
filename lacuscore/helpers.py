#!/usr/bin/env python3

from __future__ import annotations

import json

from enum import IntEnum, unique
from logging import LoggerAdapter
from typing import MutableMapping, Any, TypedDict, Literal, Mapping

from pydantic import BaseModel, field_validator, model_validator
from pydantic_core import from_json

from playwrightcapture.capture import CaptureResponse as PlaywrightCaptureResponse


class LacusCoreException(Exception):
    pass


class CaptureError(LacusCoreException):
    pass


class RetryCapture(LacusCoreException):
    pass


class CaptureSettingsError(LacusCoreException):
    pass


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
    cookies: list[dict[str, str]] | None
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
    general_timeout_in_sec: int | None = None
    cookies: list[dict[str, Any]] | None = None
    headers: dict[str, str] | None = None
    http_credentials: dict[str, str] | None = None
    geolocation: dict[str, float] | None = None
    timezone_id: str | None = None
    locale: str | None = None
    color_scheme: str | None = None
    viewport: dict[str, int] | None = None
    referer: str | None = None
    with_favicon: bool = False
    allow_tracking: bool = False
    force: bool = False
    recapture_interval: int = 300
    priority: int = 0
    uuid: str | None = None

    depth: int = 0
    rendered_hostname_only: bool = True  # Note: only used if depth is > 0

    @model_validator(mode='after')
    def check_capture_element(self) -> CaptureSettings:
        if self.url and (self.document or self.document_name):
            raise CaptureSettingsError('You cannot provide both a URL and a document to capture')
        if not self.url and not (self.document and self.document_name):
            raise CaptureSettingsError('You must provide either a URL or a document to capture')
        return self

    @field_validator('url', mode='after')
    @classmethod
    def load_url(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator('document_name', mode='after')
    @classmethod
    def load_document_name(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator('proxy', mode='before')
    @classmethod
    def load_proxy_json(cls, v: Any) -> str | dict[str, str] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v.startswith('{'):
                return from_json(v)
            # Just the proxy
            return v
        elif isinstance(v, dict):
            return v
        return None

    @field_validator('cookies', mode='before')
    @classmethod
    def load_cookies_json(cls, v: Any) -> list[dict[str, Any]] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v.startswith('['):
                return from_json(v)
            # Cookies are invalid, ignoring.
        elif isinstance(v, list):
            return v
        return None

    @field_validator('headers', mode='before')
    @classmethod
    def load_headers_json(cls, v: Any) -> dict[str, str] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v[0] == '{':
                return from_json(v)
            else:
                # make it a dict
                new_headers = {}
                for header_line in v.splitlines():
                    if header_line and ':' in header_line:
                        splitted = header_line.split(':', 1)
                        if splitted and len(splitted) == 2:
                            header, h_value = splitted
                            if header.strip() and h_value.strip():
                                new_headers[header.strip()] = h_value.strip()
                return new_headers
        elif isinstance(v, dict):
            return v
        return None

    @field_validator('http_credentials', mode='before')
    @classmethod
    def load_http_creds_json(cls, v: Any) -> dict[str, str] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v.startswith('{'):
                return from_json(v)
        elif isinstance(v, dict):
            return v
        return None

    @field_validator('geolocation', mode='before')
    @classmethod
    def load_geolocation_json(cls, v: Any) -> dict[str, float] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v.startswith('{'):
                return from_json(v)
        elif isinstance(v, dict):
            return v
        return None

    @field_validator('viewport', mode='before')
    @classmethod
    def load_viewport_json(cls, v: Any) -> dict[str, int] | None:
        if not v:
            return None
        if isinstance(v, str):
            if v.startswith('{'):
                return from_json(v)
        elif isinstance(v, dict):
            return v
        return None

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
