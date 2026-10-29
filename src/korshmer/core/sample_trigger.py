from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

MAX_VOICES: int = 16

MAX_SAMPLE_SECONDS: float = 60.0


class SampleMode(Enum):
    ONE_SHOT = "one_shot"
    LOOP = "loop"


class _Voice:
    __slots__ = ("note", "data", "mode", "pos")

    def __init__(self, note, data, mode):
        self.note = note
        self.data = data
        self.mode = mode
        self.pos = 0


def _to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("a sample must be 1-D or 2-D audio")
    if arr.shape[1] == channels:
        return arr
    if arr.shape[1] == 1:
        return np.repeat(arr, channels, axis=1)
    if channels == 1:
        return arr.mean(axis=1, keepdims=True)
    out = np.zeros((arr.shape[0], channels), dtype=np.float32)
    n = min(channels, arr.shape[1])
    out[:, :n] = arr[:, :n]
    return out


def _resample(arr: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return arr
    n = max(1, int(round(len(arr) * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.stack(
        [np.interp(x_new, x_old, arr[:, c]) for c in range(arr.shape[1])],
        axis=1,
    ).astype(np.float32)


class SampleBank:
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._commands: deque = deque()

        self._raw: dict[int, tuple[np.ndarray, int, SampleMode]] = {}

        self._samples: dict[int, tuple[np.ndarray, SampleMode]] = {}
        self._voices: list[_Voice] = []

    def assign(
        self,
        note: int,
        data: np.ndarray,
        mode: SampleMode = SampleMode.ONE_SHOT,
        sample_rate: int | None = None,
    ) -> None:
        if (
            not isinstance(note, int)
            or isinstance(note, bool)
            or not 0 <= note <= 127
        ):
            raise ValueError(f"note must be 0-127, got {note!r}")
        mode = SampleMode(mode)
        rate = int(sample_rate or self.sample_rate)
        if rate <= 0:
            raise ValueError("sample_rate must be positive")
        raw = np.asarray(data, dtype=np.float32)
        if len(raw) == 0:
            raise ValueError("a sample cannot be empty")
        if len(raw) > MAX_SAMPLE_SECONDS * rate:
            raise ValueError(f"sample longer than {MAX_SAMPLE_SECONDS:g} s")
        arr = self._convert(raw, rate)
        self._raw[note] = (raw, rate, mode)
        self._commands.append(("assign", note, arr, mode))

    def _convert(self, raw: np.ndarray, rate: int) -> np.ndarray:
        return _resample(
            _to_channels(raw, self.channels), rate, self.sample_rate
        )

    def configure(self, sample_rate: int, channels: int) -> None:
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if (int(sample_rate), int(channels)) == (
            self.sample_rate,
            self.channels,
        ):
            return
        self.sample_rate, self.channels = int(sample_rate), int(channels)
        self._commands.append(("stop_all", None, None, None))
        for note, (raw, rate, mode) in self._raw.items():
            self._commands.append(
                ("assign", note, self._convert(raw, rate), mode)
            )

    def unassign(self, note: int) -> None:
        self._raw.pop(note, None)
        self._commands.append(("unassign", note, None, None))

    def trigger(self, note: int) -> None:
        self._commands.append(("on", note, None, None))

    def release(self, note: int) -> None:
        self._commands.append(("off", note, None, None))

    def stop_all(self) -> None:
        self._commands.append(("stop_all", None, None, None))

    def handle_action(self, action) -> bool:
        name = getattr(action, "action", None)
        if name == "note_on":
            self.trigger(action.value)
        elif name == "note_off":
            self.release(action.value)
        else:
            return False
        return True

    @property
    def active_voices(self) -> int:
        return len(self._voices)

    def _apply_commands(self) -> None:
        while self._commands:
            try:
                kind, note, data, mode = self._commands.popleft()
            except IndexError:
                break
            if kind == "assign":
                self._samples[note] = (data, mode)
            elif kind == "unassign":
                self._samples.pop(note, None)
                self._voices = [v for v in self._voices if v.note != note]
            elif kind == "on":
                entry = self._samples.get(note)
                if entry is not None and len(self._voices) < MAX_VOICES:
                    self._voices.append(_Voice(note, entry[0], entry[1]))
            elif kind == "off":
                self._voices = [
                    v
                    for v in self._voices
                    if not (v.note == note and v.mode is SampleMode.LOOP)
                ]
            elif kind == "stop_all":
                self._voices = []

    def mix_into(self, out: np.ndarray, gain: float = 1.0) -> None:
        self._apply_commands()
        if not self._voices:
            return
        frames = len(out)
        alive = []
        for v in self._voices:
            filled = 0
            n_total = len(v.data)
            while filled < frames:
                if v.pos >= n_total:
                    if v.mode is SampleMode.LOOP:
                        v.pos = 0
                    else:
                        break
                n = min(frames - filled, n_total - v.pos)
                out[filled : filled + n] += v.data[v.pos : v.pos + n] * gain
                v.pos += n
                filled += n
            if v.mode is SampleMode.LOOP or v.pos < n_total:
                alive.append(v)
        self._voices = alive
