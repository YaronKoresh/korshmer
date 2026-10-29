from __future__ import annotations

import math
import numbers
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        if (
            isinstance(rate, bool)
            or not isinstance(rate, numbers.Real)
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(
                f"rate must be a finite number greater than zero, got {rate!r}"
            )
        if (
            isinstance(burst, bool)
            or not isinstance(burst, numbers.Integral)
            or burst < 1
        ):
            raise ValueError(
                f"burst must be a whole number of at least 1, got {burst!r}"
            )
        self.rate = float(rate)
        self.burst = int(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._last = clock()

    def allow(self) -> bool:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
