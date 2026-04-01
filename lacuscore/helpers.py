#!/usr/bin/env python3

from __future__ import annotations

import sys

from enum import IntEnum, unique
from logging import LoggerAdapter
from typing import Any
from collections.abc import MutableMapping

from playwrightcapture import (CaptureResponse as PlaywrightCaptureResponse,
                               FramesResponse as PlaywrightFramesResponse,
                               )

from lookyloo_models import Cookie

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class LacusCoreException(Exception):
    pass


class CaptureError(LacusCoreException):
    pass


class RetryCapture(LacusCoreException):
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


@unique
class SessionStatus(IntEnum):
    '''The status of an interactive session'''

    UNKNOWN = -1
    STARTING = 0
    READY = 1
    ERROR = 2
    STOPPED = 3
    EXPIRED = 4
    CAPTURE_REQUESTED = 5


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
    frames: PlaywrightFramesResponse | None
    png: str | None
    downloaded_filename: str | None
    downloaded_file: str | None
    children: list[CaptureResponseJson] | None
    trusted_timestamps: dict[str, str] | None
    runtime: float | None
    potential_favicons: list[str] | None


class SessionMetadata(TypedDict, total=False):
    """Backend-agnostic interactive session metadata stored in Redis."""

    status: int
    backend_type: str
    view_url: str
    created_at: int
    expires_at: int
    capture_requested_at: int


class XpraSessionMetadata(TypedDict, total=False):
    """XPRA-specific transport metadata stored separately from session state."""

    display: str
    socket_path: str


class StoredSessionMetadata(SessionMetadata, XpraSessionMetadata, total=False):
    """Compatibility view combining public session metadata and backend state."""
