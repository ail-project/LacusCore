#!/usr/bin/env python3

from typing import List, Tuple, Dict, Optional, Union, Any

from datetime import datetime, date

from redis import Redis


class LacusCoreMonitoring():

    def __init__(self, redis_connector: Redis):
        self.redis = redis_connector

    def check_redis_up(self) -> bool:
        return bool(self.redis.ping())

    def get_ongoing_captures(self) -> List[Tuple[str, datetime]]:
        return [(uuid, datetime.fromtimestamp(timestamp)) for uuid, timestamp in self.redis.zrevrangebyscore('lacus:ongoing', '+Inf', 0, withscores=True)]

    def get_capture_settings(self, uuid: str) -> Dict[str, str]:
        return self.redis.hgetall(f'lacus:capture_settings:{uuid}')

    def get_enqueued_captures(self) -> List[Tuple[str, float]]:
        return self.redis.zrevrangebyscore('lacus:to_capture', '+Inf', '-Inf', withscores=True)

    def get_capture_result(self, uuid: str) -> Optional[str]:
        return self.redis.get(f'lacus:capture_results:{uuid}')

    def get_capture_result_size(self, uuid: str) -> Optional[str]:
        return self.redis.memory_usage(f'lacus:capture_results:{uuid}')

    def get_stats(self, d: Optional[Union[datetime, date, str]]=None, /):
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
        to_return: Dict[str, Any] = {}
        if errors := self.redis.zrevrangebyscore(f'stats:{_date}:errors', '+Inf', 0, withscores=True):
            to_return['errors'] = errors
        if retry_failed := self.redis.smembers(f'stats:{_date}:retry_failed'):
            to_return['retry_failed'] = retry_failed
        if retry_success := self.redis.smembers(f'stats:{_date}:retry_success'):
            to_return['retry_success'] = retry_success
        if captures := self.redis.smembers(f'stats:{_date}:captures'):
            to_return['captures'] = captures
        return to_return
