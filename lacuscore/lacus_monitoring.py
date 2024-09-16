#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from datetime import datetime, date

from glide import GlideClient, InfBound, RangeByScore, ScoreBoundary


class LacusCoreMonitoring():

    def __init__(self, redis_connector: GlideClient):
        self.redis = redis_connector

    async def check_redis_up(self) -> bool:
        return bool(await self.redis.ping())

    async def get_ongoing_captures(self) -> list[tuple[str, datetime]]:
        ongoing = await self.redis.zrange_withscores('lacus:ongoing', RangeByScore(InfBound.POS_INF, ScoreBoundary(0)))
        return [(uuid.decode(), datetime.fromtimestamp(timestamp)) for uuid, timestamp in ongoing.items()]

    async def get_capture_settings(self, uuid: str) -> dict[str, str]:
        captures = await self.redis.hgetall(f'lacus:capture_settings:{uuid}')
        return {k.decode(): v.decode() for k, v in captures.items()}

    async def get_enqueued_captures(self) -> list[tuple[str, float]]:
        enqueued = await self.redis.zrange_withscores('lacus:to_capture', RangeByScore(InfBound.POS_INF, InfBound.NEG_INF))
        return [(uuid.decode(), timestamp) for uuid, timestamp in enqueued.items()]

    async def get_capture_result(self, uuid: str) -> str | None:
        if result := await self.redis.get(f'lacus:capture_results:{uuid}'):
            return result.decode()
        return None

    # async def get_capture_result_size(self, uuid: str) -> str | None:
    #     return await self.redis.memory_usage(f'lacus:capture_results:{uuid}')

    async def get_stats(self, d: datetime | date | str | None=None, /, *, cardinality_only: bool=False) -> dict[str, Any]:
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
        if errors := await self.redis.zrange_withscores(f'stats:{_date}:errors', RangeByScore(InfBound.POS_INF, ScoreBoundary(0))):
            to_return['errors'] = [(uuid.decode(), timestamp) for uuid, timestamp in errors.items()]
        if cardinality_only:
            if retry_failed := await self.redis.scard(f'stats:{_date}:retry_failed'):
                to_return['retry_failed'] = retry_failed
            if retry_success := await self.redis.scard(f'stats:{_date}:retry_success'):
                to_return['retry_success'] = retry_success
            if captures := await self.redis.scard(f'stats:{_date}:captures'):
                to_return['captures'] = captures
        else:
            if retry_failed_list := await self.redis.smembers(f'stats:{_date}:retry_failed'):
                to_return['retry_failed'] = {error.decode() for error in retry_failed_list}
            if retry_success_list := await self.redis.smembers(f'stats:{_date}:retry_success'):
                to_return['retry_success'] = {success.decode() for success in retry_success_list}
            if captures_list := await self.redis.smembers(f'stats:{_date}:captures'):
                to_return['captures'] = {capture.decode() for capture in captures_list}
        return to_return
