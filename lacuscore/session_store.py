#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from redis import Redis

from .helpers import SessionMetadata, SessionStatus, InteractiveSessionError


@dataclass
class StoredSessionRecord:
    metadata: SessionMetadata
    backend_metadata: dict[str, Any]


class SessionMetadataStore:
    """Redis-backed storage for interactive session lifecycle and backend state."""

    _int_fields = {'status', 'created_at', 'expires_at', 'request_finish'}

    def __init__(self, redis: Redis[bytes]) -> None:
        self.redis: Redis[bytes] = redis

    @staticmethod
    def core_key(uuid: str) -> str:
        return f'lacus:interactive_session:{uuid}'

    @staticmethod
    def backend_key(uuid: str, backend_type: str) -> str:
        return f'lacus:interactive_session:{uuid}:{backend_type}'

    def _decode_hash(self, raw: dict[bytes, bytes]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            key = raw_key.decode()
            value: Any = raw_value.decode()
            if key in self._int_fields:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            decoded[key] = value
        return decoded

    def write(self, uuid: str, metadata: SessionMetadata,
              backend_metadata: dict[str, str], *, expire_seconds: int) -> None:
        """Persist session metadata and optional backend state to Redis."""
        if 'backend_type' not in metadata or not metadata.get('backend_type'):
            raise InteractiveSessionError('No backend type in the metatda the session is invalid')

        interactive_session_key = 'lacus:interactive_session'
        now = datetime.now().timestamp()
        core_key = self.core_key(uuid)
        backend_key = self.backend_key(uuid, metadata['backend_type'])

        pipeline = self.redis.pipeline()
        pipeline.zadd(interactive_session_key, mapping={uuid: now})
        pipeline.hset(core_key, mapping=metadata)  # type: ignore[arg-type]
        if backend_metadata:
            pipeline.hset(backend_key, mapping=backend_metadata)  # type: ignore[arg-type]

        pipeline.expire(core_key, expire_seconds)
        pipeline.expire(backend_key, expire_seconds)
        pipeline.execute()

    def mark_terminal(self, uuid: str, metadata: SessionMetadata, *, status: SessionStatus,
                      expire_seconds: int=60) -> None:
        """Update session status to a terminal state and set a short expiry."""
        if 'backend_type' not in metadata or not metadata.get('backend_type'):
            raise InteractiveSessionError('No backend type in the metadata the session is invalid')
        core_key = self.core_key(uuid)
        backend_key = self.backend_key(uuid, metadata['backend_type'])

        pipeline = self.redis.pipeline()
        pipeline.hset(core_key, 'status', int(status))
        pipeline.expire(core_key, expire_seconds)
        pipeline.expire(backend_key, expire_seconds)
        pipeline.execute()

    def read(self, uuid: str) -> StoredSessionRecord | None:
        """Load session and backend metadata for a capture UUID, or None if absent."""
        core_key = self.core_key(uuid)
        raw_core_metadata = self.redis.hgetall(core_key)
        if not raw_core_metadata:
            return None

        core_metadata = self._decode_hash(raw_core_metadata)
        if 'backend_type' not in core_metadata or not core_metadata.get('backend_type'):
            raise InteractiveSessionError('No backend type in the metadata the session is invalid')

        backend_key = self.backend_key(uuid, core_metadata['backend_type'])
        raw_backend_metadata = self.redis.hgetall(backend_key)
        backend_metadata = self._decode_hash(raw_backend_metadata) if raw_backend_metadata else {}

        return StoredSessionRecord(
            metadata=cast(SessionMetadata, core_metadata),
            backend_metadata=backend_metadata,
        )

    def request_finish(self, uuid: str) -> bool:
        """Mark a session as ready for final capture."""
        core_key = self.core_key(uuid)
        if self.redis.exists(core_key):
            self.redis.hset(core_key, 'finish_requested', 1)
            # NOTE: just in case, somehow, the key expires between exists and hset
            self.redis.expire(core_key, 360)
            return True
        return False

    def scan_expired(self) -> list[tuple[str, StoredSessionRecord]]:
        """Return all non-terminal sessions whose expires_at is in the past."""
        expired_sessions: list[tuple[str, StoredSessionRecord]] = []
        now_ts = datetime.now().timestamp()
        old_uuids = []
        for uuid, start_ts in self.redis.zscan_iter('lacus:interactive_session'):
            # make sure we don't have a uuid older than 1h in there
            if (datetime.fromtimestamp(start_ts) + timedelta(hours=1)).timestamp() < now_ts:
                old_uuids.append(uuid)
            record = self.read(uuid.decode())
            if not record:
                # no metadata available
                old_uuids.append(uuid)
                continue

            status_val = int(record.metadata.get('status', int(SessionStatus.UNKNOWN)))
            if status_val in (int(SessionStatus.STOPPED), int(SessionStatus.EXPIRED), int(SessionStatus.ERROR)):
                # if that value is set, the session was already stoped.
                continue

            if uuid in old_uuids:
                expired_sessions.append((uuid.decode(), record))
                continue

            expires_at_ts = int(record.metadata.get('expires_at', 0) or 0)
            if expires_at_ts and expires_at_ts <= now_ts:
                expired_sessions.append((uuid.decode(), record))
                old_uuids.append(uuid)

        if old_uuids:
            self.redis.zrem('lacus:interactive_session', *old_uuids)

        return expired_sessions
