"""Token-bucket rate limiting for the web tools.

The bucket holds up to `burst` tokens and refills at `per_minute` tokens per minute. Each
call takes one token; an empty bucket rejects the call and says when to retry. The state
lives in the server process, which is why the Web MCP server is long-lived (Streamable
HTTP), not spawned per call.
"""

import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        per_minute: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute <= 0 or burst < 1:
            raise ValueError("per_minute must be > 0 and burst >= 1")
        self.rate = per_minute / 60.0  # tokens per second
        self.capacity = float(burst)
        self.tokens = float(burst)
        self._clock = clock
        self._updated = clock()
        self._lock = threading.Lock()

    def try_acquire(self) -> tuple[bool, float]:
        """Take one token. Returns (allowed, seconds until a token is available)."""
        with self._lock:
            now = self._clock()
            self.tokens = min(
                self.capacity, self.tokens + (now - self._updated) * self.rate
            )
            self._updated = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True, 0.0
            return False, round((1 - self.tokens) / self.rate, 1)
