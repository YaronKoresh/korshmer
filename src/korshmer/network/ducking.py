from __future__ import annotations

import math

import numpy as np

_FULL_SCALE = 32768.0


class Ducker:
    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        depth_db: float = 12.0,
        threshold_db: float = -40.0,
        attack_ms: float = 50.0,
        release_ms: float = 300.0,
        enabled: bool = True,
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
        self.sample_rate = sample_rate
        self.channels = channels
        self.enabled = bool(enabled)
        self.configure(depth_db, threshold_db, attack_ms, release_ms)
        self._gain = 1.0

    def configure(
        self,
        depth_db: float,
        threshold_db: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        for name, value in (
            ("depth_db", depth_db),
            ("threshold_db", threshold_db),
            ("attack_ms", attack_ms),
            ("release_ms", release_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= depth_db <= 60:
            raise ValueError("depth_db must be between 0 and 60")
        if not -100 <= threshold_db <= 0:
            raise ValueError("threshold_db must be between -100 and 0")
        if attack_ms <= 0 or release_ms <= 0:
            raise ValueError("attack_ms and release_ms must be positive")
        self.depth_db = float(depth_db)
        self.threshold_db = float(threshold_db)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)

    @property
    def gain(self) -> float:
        return self._gain

    def reset(self) -> None:
        self._gain = 1.0

    @staticmethod
    def _to_array(pcm: bytes) -> np.ndarray:
        return np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2)

    def voice_active(self, voice: np.ndarray) -> bool:
        if voice.size == 0:
            return False
        rms = float(
            np.sqrt(np.mean((voice.astype(np.float64) / _FULL_SCALE) ** 2))
        )
        if rms <= 0.0:
            return False
        return 20.0 * math.log10(rms) >= self.threshold_db

    def process(self, program: bytes, voice: bytes | None = None) -> bytes:
        if not self.enabled:
            return program
        prog = self._to_array(program)
        vo = self._to_array(voice) if voice else np.zeros(0, dtype="<i2")
        active = self.voice_active(vo)
        target = 10.0 ** (-self.depth_db / 20.0) if active else 1.0
        if not active and self._gain == 1.0:
            return (
                program if not vo.any() else self._mix(prog, vo, 1.0, program)
            )
        n = prog.size
        frames = n // self.channels
        if frames == 0:
            return program
        time_ms = self.attack_ms if target < self._gain else self.release_ms
        step = (1.0 - 10.0 ** (-self.depth_db / 20.0)) / max(
            1.0, time_ms * self.sample_rate / 1000.0
        )
        idx = np.arange(1, frames + 1, dtype=np.float64)
        if target < self._gain:
            gains = np.maximum(self._gain - step * idx, target)
        else:
            gains = np.minimum(self._gain + step * idx, target)
        self._gain = float(gains[-1])
        gain_samples = np.repeat(gains, self.channels)
        return self._mix(prog, vo, gain_samples, program)

    def _mix(self, prog, vo, gain, original: bytes) -> bytes:
        n = prog.size
        mixed = prog.astype(np.float64)
        used = n - n % self.channels
        if isinstance(gain, np.ndarray):
            mixed[:used] *= gain[:used]
        if vo.size:
            m = min(n, vo.size)
            mixed[:m] += vo[:m]
        out = np.clip(np.rint(mixed), -32768, 32767).astype("<i2").tobytes()
        return out + original[n * 2 :]
