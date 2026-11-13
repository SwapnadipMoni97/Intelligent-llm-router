import asyncio
import time
from collections import deque


class CircuitBreakers:
    def __init__(self, threshold: int = 3, recovery_seconds: float = 30):
        self.threshold, self.recovery_seconds = threshold, recovery_seconds
        self.failures: dict[str, int] = {}
        self.opened_at: dict[str, float] = {}

    def is_available(self, key: str) -> bool:
        opened = self.opened_at.get(key)
        return opened is None or time.monotonic() - opened >= self.recovery_seconds

    def success(self, key: str) -> None:
        self.failures.pop(key, None)
        self.opened_at.pop(key, None)

    def failure(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if self.failures[key] >= self.threshold:
            self.opened_at[key] = time.monotonic()


class InMemoryRateLimiter:
    """Single-process development limiter; replace with shared Redis in multi-worker deployments."""
    def __init__(self, requests_per_minute: int):
        self.limit = requests_per_minute
        self.requests: dict[str, deque[float]] = {}
        self.lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self.lock:
            bucket = self.requests.setdefault(key, deque())
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True
