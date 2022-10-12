#!/usr/bin/env python3

from typing import List, Tuple

from datetime import datetime

from redis import Redis


class LacusCoreMonitoring():

    def __init__(self, redis_connector: Redis):
        self.redis = redis_connector

    def check_redis_up(self) -> bool:
        return bool(self.redis.ping())

    def get_ongoing_captures(self) -> List[Tuple[str, datetime]]:
        return [(uuid, datetime.fromtimestamp(timestamp)) for uuid, timestamp in self.redis.zrevrangebyscore('lacus:ongoing', '+Inf', 0, withscores=True)]
