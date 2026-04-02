#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, cast

from redis import Redis
from redis.exceptions import WatchError

from .helpers import SessionMetadata, SessionStatus, StoredSessionMetadata


@dataclass
class StoredSessionRecord:
    metadata: SessionMetadata
    backend_metadata: dict[str, Any]

    def merged_metadata(self) -> StoredSessionMetadata:
        merged = dict(self.metadata)
        merged.update(self.backend_metadata)
        return cast(StoredSessionMetadata, merged)


class SessionMetadataStore:
    """Redis-backed storage for interactive session lifecycle and backend state."""

    _int_fields = {'status', 'created_at', 'expires_at', 'capture_requested_at'}
    _legacy_xpra_fields = {'display', 'socket_path'}

    def __init__(self, redis: Redis[bytes], *, finish_key: str='capture_requested_at') -> None:
        self.redis: Redis[bytes] = redis
        self.finish_key = finish_key

    @staticmethod
    def core_key(uuid: str) -> str:
        return f'lacus:session:{uuid}'

    @staticmethod
    def backend_key(uuid: str, backend_type: str) -> str:
        return f'lacus:session:{uuid}:{backend_type}'

    def _decode_hash(self, raw: dict[bytes, Any]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key_bytes, value_bytes in raw.items():
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else str(key_bytes)
            value: Any = value_bytes.decode() if isinstance(value_bytes, bytes) else value_bytes
            if key in self._int_fields:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            decoded[key] = value
        return decoded

    def _extract_legacy_backend_metadata(self, core_metadata: dict[str, Any]) -> dict[str, Any]:
        backend_metadata: dict[str, Any] = {}
        for key in self._legacy_xpra_fields:
            if key in core_metadata:
                backend_metadata[key] = core_metadata.pop(key)
        return backend_metadata

    def write(self, uuid: str, metadata: SessionMetadata,
              backend_metadata: dict[str, Any] | None=None, *, expire_seconds: int | None=None) -> None:
        core_metadata = dict(metadata)
        backend_type = str(core_metadata.get('backend_type') or 'xpra')
        core_metadata['backend_type'] = backend_type

        pipeline = self.redis.pipeline()
        pipeline.hset(self.core_key(uuid), mapping=core_metadata)  # type: ignore[arg-type]
        if backend_metadata is not None:
            backend_key = self.backend_key(uuid, backend_type)
            if backend_metadata:
                pipeline.hset(backend_key, mapping=backend_metadata)  # type: ignore[arg-type]
            else:
                pipeline.delete(backend_key)

        if expire_seconds is not None:
            pipeline.expire(self.core_key(uuid), expire_seconds)
            pipeline.expire(self.backend_key(uuid, backend_type), expire_seconds)

        pipeline.execute()

    def mark_terminal(self, uuid: str, metadata: SessionMetadata, *, status: SessionStatus,
                      expire_seconds: int=60) -> None:
        core_metadata = dict(metadata)
        backend_type = str(core_metadata.get('backend_type') or 'xpra')
        core_metadata['backend_type'] = backend_type
        core_metadata['status'] = int(status)

        pipeline = self.redis.pipeline()
        pipeline.hset(self.core_key(uuid), mapping=core_metadata)  # type: ignore[arg-type]
        pipeline.hdel(self.core_key(uuid), self.finish_key)
        pipeline.expire(self.core_key(uuid), expire_seconds)
        pipeline.expire(self.backend_key(uuid, backend_type), expire_seconds)
        pipeline.execute()

    def read(self, uuid: str) -> StoredSessionRecord | None:
        raw_core_metadata = self.redis.hgetall(self.core_key(uuid))
        if not raw_core_metadata:
            return None

        core_metadata = self._decode_hash(raw_core_metadata)
        backend_type = str(core_metadata.get('backend_type') or 'xpra')
        core_metadata['backend_type'] = backend_type

        raw_backend_metadata = self.redis.hgetall(self.backend_key(uuid, backend_type))
        if raw_backend_metadata:
            backend_metadata = self._decode_hash(raw_backend_metadata)
        else:
            backend_metadata = self._extract_legacy_backend_metadata(core_metadata)

        return StoredSessionRecord(
            metadata=cast(SessionMetadata, core_metadata),
            backend_metadata=backend_metadata,
        )

    def request_finish(self, uuid: str) -> StoredSessionRecord | None:
        core_key = self.core_key(uuid)
        now_ts = int(time.time())

        with self.redis.pipeline() as pipeline:
            while True:
                try:
                    pipeline.watch(core_key)
                    # After watch(), hgetall executes immediately and returns
                    # a dict rather than a Pipeline future.
                    raw_core_metadata: dict[bytes, Any] = pipeline.hgetall(core_key)  # type: ignore[assignment]
                    if not raw_core_metadata:
                        pipeline.reset()
                        return None

                    core_metadata = self._decode_hash(raw_core_metadata)
                    status_val = int(core_metadata.get('status', int(SessionStatus.UNKNOWN)))
                    if status_val in (int(SessionStatus.STOPPED), int(SessionStatus.EXPIRED), int(SessionStatus.ERROR)):
                        pipeline.unwatch()
                        return self.read(uuid)

                    requested_at = int(core_metadata.get(self.finish_key, 0) or 0)
                    if requested_at:
                        pipeline.unwatch()
                        return self.read(uuid)

                    pipeline.multi()
                    pipeline.hset(core_key, mapping={self.finish_key: now_ts})
                    pipeline.execute()
                    return self.read(uuid)
                except WatchError:
                    continue

    def scan_expired(self, now_ts: int) -> list[tuple[str, StoredSessionRecord]]:
        expired_sessions: list[tuple[str, StoredSessionRecord]] = []
        for key in self.redis.scan_iter('lacus:session:*'):
            key_str = key.decode() if isinstance(key, bytes) else str(key)
            if key_str.count(':') != 2:
                continue

            uuid = key_str.rsplit(':', 1)[-1]
            record = self.read(uuid)
            if not record:
                continue

            status_val = int(record.metadata.get('status', int(SessionStatus.UNKNOWN)))
            if status_val in (int(SessionStatus.STOPPED), int(SessionStatus.EXPIRED), int(SessionStatus.ERROR)):
                continue

            expires_at_ts = int(record.metadata.get('expires_at', 0) or 0)
            if expires_at_ts and expires_at_ts <= now_ts:
                expired_sessions.append((uuid, record))

        return expired_sessions
