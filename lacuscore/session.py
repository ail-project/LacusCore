#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .helpers import SessionMetadata, SessionStatus
from .session_store import SessionMetadataStore


@dataclass
class Session:
    """Backend-agnostic representation of an interactive session.

    This structure captures the minimal set of fields that callers need in
    order to expose and manage an interactive browsing session, independent
    of the underlying implementation (xpra or otherwise).
    """

    created_at: datetime
    expires_at: datetime


class SessionManager(Protocol):
    """Protocol for objects able to manage interactive sessions.

    Concrete backends (for example, an xpra-based implementation) are free
    to accept additional keyword-only arguments in ``start_session``. The
    return type must at least conform to :class:`Session`.
    """

    backend_type: str
    session_store: SessionMetadataStore

    def start_session(self, *, session_name: str, ttl: int) -> tuple[Session, SessionMetadata, dict[str, str]]:  # pragma: no cover - structural contract
        ...

    def stop_session(self, session: Session, uuid: str, metadata: SessionMetadata,
                     *, status: SessionStatus, expire_seconds: int) -> bool:  # pragma: no cover - structural contract
        ...

    def serialize_backend_metadata(self, session: Session) -> dict[str, Any]:  # pragma: no cover - structural contract
        ...

    def restore_session(self, *, created_at: datetime, expires_at: datetime,
                        backend_metadata: Mapping[str, str]) -> Session:  # pragma: no cover - structural contract
        ...

    def get_capture_env(self, session: Session) -> Mapping[str, str | float | bool]:  # pragma: no cover - structural contract
        ...

    def cleanup_expired_sessions(self) -> None:
        ...
