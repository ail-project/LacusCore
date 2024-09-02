#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from datetime import datetime, date

from redis import Redis


class LacusCoreMonitoring():

    def __init__(self, redis_connector: Redis[str]):
        self.redis = redis_connector

    def check_redis_up(self) -> bool:
        return bool(self.redis.ping())

    def get_ongoing_captures(self) -> list[tuple[str, datetime]]:
        return [(uuid, datetime.fromtimestamp(timestamp)) for uuid, timestamp in self.redis.zrevrangebyscore('lacus:ongoing', '+Inf', 0, withscores=True)]

    def get_capture_settings(self, uuid: str) -> dict[str, str]:
        return self.redis.hgetall(f'lacus:capture_settings:{uuid}')

    def get_enqueued_captures(self) -> list[tuple[str, float]]:
        return self.redis.zrevrangebyscore('lacus:to_capture', '+Inf', '-Inf', withscores=True)

    def get_capture_result(self, uuid: str) -> str | None:
        return self.redis.get(f'lacus:capture_results:{uuid}')

    def get_capture_result_size(self, uuid: str) -> str | None:
        return self.redis.memory_usage(f'lacus:capture_results:{uuid}')

    def get_stats(self, d: datetime | date | str | None=None, /, *, cardinality_only: bool=False) -> dict[str, Any]:
        if d is None:
            _date = date.today().isoformat()
        elif isinstance(d, str):
            _date = d
        elif isinstance(d, datetime):
            _date = d.date().isoformat()
        elif isinstance(d, date):
            _date = d.isoformat()
        else:
            raise Exception('Invalid type for date ({type(d)})')
        to_return: dict[str, list[tuple[str, float]] | int | set[str]] = {}
        if errors := self.redis.zrevrangebyscore(f'stats:{_date}:errors', '+Inf', 0, withscores=True):
            to_return['errors'] = errors
        if cardinality_only:
            if retry_failed := self.redis.scard(f'stats:{_date}:retry_failed'):
                to_return['retry_failed'] = retry_failed
            if retry_success := self.redis.scard(f'stats:{_date}:retry_success'):
                to_return['retry_success'] = retry_success
            if captures := self.redis.scard(f'stats:{_date}:captures'):
                to_return['captures'] = captures
        else:
            if retry_failed_list := self.redis.smembers(f'stats:{_date}:retry_failed'):
                to_return['retry_failed'] = retry_failed_list
            if retry_success_list := self.redis.smembers(f'stats:{_date}:retry_success'):
                to_return['retry_success'] = retry_success_list
            if captures_list := self.redis.smembers(f'stats:{_date}:captures'):
                to_return['captures'] = captures_list
        return to_return
