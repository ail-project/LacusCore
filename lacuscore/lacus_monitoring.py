#!/usr/bin/env python3

from typing import List, Tuple, Dict, Optional

from datetime import datetime

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
