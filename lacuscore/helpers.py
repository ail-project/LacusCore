#!/usr/bin/env python3

from __future__ import annotations

from enum import IntEnum, unique
from logging import LoggerAdapter
from typing import MutableMapping, Any, TypedDict

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


class CaptureSettings(TypedDict, total=False):
    '''The capture settings that can be passed to Lacus.'''

    url: str | None
    document_name: str | None
    document: str | None
    browser: str | None
    device_name: str | None
    user_agent: str | None
    proxy: str | dict[str, str] | None
    general_timeout_in_sec: int | None
    cookies: list[dict[str, Any]] | None
    headers: str | dict[str, str] | None
    http_credentials: dict[str, str] | None
    geolocation: dict[str, float] | None
    timezone_id: str | None
    locale: str | None
    color_scheme: str | None
    viewport: dict[str, int] | None
    referer: str | None
    with_favicon: bool
    allow_tracking: bool
    force: bool
    recapture_interval: int
    priority: int
    uuid: str | None

    depth: int
    rendered_hostname_only: bool  # Note: only used if depth is > 0
