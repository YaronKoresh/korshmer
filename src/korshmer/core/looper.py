from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable

import numpy as np

from korshmer.core.sample_trigger import _to_channels
from korshmer.utils.tempo import validate_bpm


class Looper:
    def __init__(
        self,
        bpm: float = 120.0,
        bars: int = 2,
        beats_per_bar: int = 4,
        sample_rate: int = 44100,
        channels: int = 1,
        latency_frames: int = 0,
    ):
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.latency_frames = max(0, int(latency_frames))
        self.beats_per_bar = self._positive(beats_per_bar, "beats_per_bar")
        self.bars = self._positive(bars, "bars")
        self.bpm = validate_bpm(bpm)
        self.pending_bpm: float | None = None
        self._lock = threading.Lock()
        self._dir: str | None = None
        self._layers: list[str] = []
        self._mix: np.ndarray | None = None
        self._pos = 0
        self._armed = False
        self._rec: np.ndarray | None = None
        self._rec_pos = 0
        self._stream = None

    @staticmethod
    def _positive(value, name) -> int:
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive whole number")
        return int(value)

    @property
    def loop_frames(self) -> int:
        if self._mix is not None:
            return len(self._mix)
        seconds = self.bars * self.beats_per_bar * 60.0 / self.bpm
        return max(1, int(round(seconds * self.sample_rate)))

    def set_bars(self, bars: int) -> None:
        if self._layers:
            raise RuntimeError("clear the loop before changing its length")
        self.bars = self._positive(bars, "bars")

    def set_bpm(self, bpm: float) -> bool:
        bpm = validate_bpm(bpm)
        with self._lock:
            if self._layers or self._rec is not None:
                self.pending_bpm = bpm
                return False
            self.bpm = bpm
            self.pending_bpm = None
            return True

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def recording(self) -> bool:
        return self._rec is not None

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def playing(self) -> bool:
        return self._mix is not None

    def arm_record(self) -> None:
        with self._lock:
            if self._rec is not None:
                return
            if self._mix is None:
                self._begin_record()
            else:
                self._armed = True

    def cancel_record(self) -> None:
        with self._lock:
            self._armed = False
            self._rec = None
            if not self._layers and self.pending_bpm is not None:
                self.bpm, self.pending_bpm = self.pending_bpm, None

    def _begin_record(self) -> None:
        self._armed = False
        self._rec = np.zeros(
            (self.loop_frames, self.channels), dtype=np.float32
        )
        self._rec_pos = 0

    def feed_input(self, block: np.ndarray) -> None:
        with self._lock:
            if self._rec is None:
                return
            data = _to_channels(block, self.channels)
            n = min(len(data), len(self._rec) - self._rec_pos)
            self._rec[self._rec_pos : self._rec_pos + n] = data[:n]
            self._rec_pos += n
            if self._rec_pos >= len(self._rec):
                self._finish_layer()

    def _finish_layer(self) -> None:
        layer, self._rec = self._rec, None
        shift = min(self.latency_frames, len(layer))
        if shift:
            layer = np.concatenate(
                [layer[shift:], np.zeros((shift, self.channels), np.float32)]
            )
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="korshmer-looper-")
        path = os.path.join(self._dir, f"layer{len(self._layers)}.npy")
        np.save(path, layer)
        self._layers.append(path)
        mix = layer.copy() if self._mix is None else self._mix + layer
        if self._mix is None:
            self._pos = 0
        self._mix = mix

    def undo_layer(self) -> bool:
        with self._lock:
            if not self._layers:
                return False
            path = self._layers.pop()
            try:
                os.remove(path)
            except OSError:
                pass
            self._mix = self._sum_layers()
            return True

    def _sum_layers(self) -> np.ndarray | None:
        if not self._layers:
            return None
        total = np.zeros(
            (len(np.load(self._layers[0], mmap_mode="r")), self.channels),
            np.float32,
        )
        for path in self._layers:
            total += np.load(path, mmap_mode="r")
        return total

    def clear(self) -> None:
        with self._lock:
            self._armed = False
            self._rec = None
            self._mix = None
            self._layers = []
            self._pos = 0
            if self._dir is not None:
                shutil.rmtree(self._dir, ignore_errors=True)
                self._dir = None
            if self.pending_bpm is not None:
                self.bpm, self.pending_bpm = self.pending_bpm, None

    def mix_out(self, out: np.ndarray) -> None:
        mix = self._mix
        if mix is None or out.shape[1] != mix.shape[1]:
            out.fill(0)
            return
        pos = self._pos
        filled = 0
        total = len(mix)
        while filled < len(out):
            if pos >= total:
                pos = 0
                if self._armed and self._rec is None:
                    self._start_armed()
            n = min(len(out) - filled, total - pos)
            np.clip(mix[pos : pos + n], -1.0, 1.0, out=out[filled : filled + n])
            pos += n
            filled += n
        if pos >= total:
            pos = 0
            if self._armed and self._rec is None:
                self._start_armed()
        self._pos = pos

    def _start_armed(self) -> None:

        with self._lock:
            if self._armed and self._rec is None and self._mix is not None:
                self._begin_record()

    def start_output(self, open_stream: Callable | None = None) -> None:
        if self._stream is not None:
            return
        started = open_stream is None
        if open_stream is None:
            import sounddevice as sd

            from korshmer.utils.audio_host import (
                open_and_start_output_stream,
            )

            def open_stream(**kw):
                return open_and_start_output_stream(sd, **kw)

        def callback(outdata, frames, time, status):
            self.mix_out(outdata)

        self._stream = open_stream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
        )
        if not started:
            self._stream.start()

    def stop_output(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def close(self) -> None:
        self.stop_output()
        self.clear()
