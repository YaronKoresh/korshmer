from __future__ import annotations

import math

import numpy as np

CURVES = ("equal_power", "linear")


class CrossfadeMixer:
    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        curve: str = "equal_power",
        initial: str | None = None,
    ):
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")
        if (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels < 1
        ):
            raise ValueError("channels must be a positive integer")
        if curve not in CURVES:
            raise ValueError(f"curve must be one of {CURVES}")
        self.sample_rate = sample_rate
        self.channels = channels
        self.curve = curve
        self._weights: dict[str, float] = {initial: 1.0} if initial else {}
        self._start: dict[str, float] = dict(self._weights)
        self._target: str | None = initial
        self._total = 0
        self._done = 0

    @property
    def current(self) -> str | None:
        return self._target

    @property
    def transitioning(self) -> bool:
        return self._done < self._total

    def switch(self, name: str, duration_ms: float = 500.0) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a scene needs a name")
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise ValueError("duration_ms must be a non-negative number")
        if name == self._target and not self.transitioning:
            return
        self._start = self._current_gains()
        self._target = name
        self._total = int(duration_ms * self.sample_rate / 1000.0)
        self._done = 0
        if self._total == 0:
            self._weights = {name: 1.0}
            self._start = dict(self._weights)

    def _progress(self, frames: np.ndarray) -> np.ndarray:
        if self._total == 0:
            return np.ones_like(frames, dtype=np.float64)
        return np.clip(frames / self._total, 0.0, 1.0)

    def _gain_curve(
        self, p: np.ndarray, w0: float, is_target: bool
    ) -> np.ndarray:
        if self.curve == "linear":
            g = w0 * (1.0 - p)
            return g + p if is_target else g
        power = w0 * w0 * (1.0 - p)
        if is_target:
            power = power + p
        return np.sqrt(power)

    def _current_gains(self) -> dict[str, float]:
        if not self.transitioning:
            return {self._target: 1.0} if self._target else {}
        p = self._progress(np.array([float(self._done)]))
        names = set(self._start) | {self._target}
        return {
            n: float(
                self._gain_curve(p, self._start.get(n, 0.0), n == self._target)[
                    0
                ]
            )
            for n in names
        }

    def mix(self, sources: dict[str, bytes]) -> bytes:
        arrays = {
            n: np.frombuffer(b, dtype="<i2", count=len(b) // 2)
            for n, b in sources.items()
            if b
        }
        if not arrays:
            return b""
        width = self.channels
        n = max(a.size for a in arrays.values())
        n -= n % width
        if n == 0:
            return b""
        frames = n // width
        idx = np.arange(frames, dtype=np.float64) + self._done
        p = self._progress(idx)
        mixed = np.zeros(n, dtype=np.float64)
        names = set(self._start) | ({self._target} if self._target else set())
        for name in names:
            a = arrays.get(name)
            if a is None:
                continue
            g = self._gain_curve(
                p, self._start.get(name, 0.0), name == self._target
            )
            m = min(a.size, n)
            m -= m % width
            mixed[:m] += a[:m] * np.repeat(g, width)[:m]
        self._done = min(self._total, self._done + frames)
        if not self.transitioning and self._target:
            self._weights = {self._target: 1.0}
            self._start = dict(self._weights)
        return np.clip(np.rint(mixed), -32768, 32767).astype("<i2").tobytes()


class SceneSwitcher:
    def __init__(self, obs, mixer: CrossfadeMixer):
        self.obs = obs
        self.mixer = mixer

    def switch(self, scene: str, duration_ms: float = 500.0) -> bool:
        self.mixer.switch(scene, duration_ms)
        return (
            bool(self.obs.switch_scene(scene))
            if self.obs is not None
            else False
        )
