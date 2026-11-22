from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
from collections.abc import Callable
from typing import Any

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
        self._stream: Any = None
        self._serial = 0
        self._unsaved: dict[str, np.ndarray] = {}
        self._unsaved_lock = threading.Lock()
        self._jobs: queue.SimpleQueue[tuple[str, np.ndarray] | None] = (
            queue.SimpleQueue()
        )
        self._writer: threading.Thread | None = None

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
        self._prepare_storage()
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

    def _prepare_storage(self) -> None:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="korshmer-looper-")
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(
                target=self._write_layers,
                name="korshmer-looper-writer",
                daemon=True,
            )
            self._writer.start()

    def _still_unsaved(self, path: str, layer: np.ndarray) -> bool:
        with self._unsaved_lock:
            return self._unsaved.get(path) is layer

    def _write_layers(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            path, layer = job
            if not self._still_unsaved(path, layer):
                continue
            try:
                np.save(path, layer)
            except OSError:
                continue
            with self._unsaved_lock:
                kept = self._unsaved.get(path) is layer
                if kept:
                    del self._unsaved[path]
            if not kept:
                self._remove_orphan(path)

    def _remove_orphan(self, path: str) -> None:
        folder = os.path.dirname(path)
        try:
            os.remove(path)
            if folder != self._dir:
                os.rmdir(folder)
        except OSError:
            return

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
        if layer is None:
            return
        shift = min(self.latency_frames, len(layer))
        if shift:
            layer = np.concatenate(
                [layer[shift:], np.zeros((shift, self.channels), np.float32)]
            )
        self._serial += 1
        path = os.path.join(self._dir or "", f"layer{self._serial}.npy")
        with self._unsaved_lock:
            self._unsaved[path] = layer
        if self._dir is not None:
            self._jobs.put((path, layer))
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
        self._discard(path)
        while True:
            with self._lock:
                kept = list(self._layers)
            try:
                total = self._sum_layers(kept)
            except (OSError, ValueError):
                if self._layers == kept:
                    raise
                continue
            with self._lock:
                if self._layers == kept:
                    self._mix = total
                    return True

    def _discard(self, path: str) -> None:
        with self._unsaved_lock:
            unsaved = self._unsaved.pop(path, None) is not None
        if not unsaved:
            try:
                os.remove(path)
            except OSError:
                pass

    def _layer(self, path: str) -> np.ndarray:
        with self._unsaved_lock:
            layer = self._unsaved.get(path)
        if layer is None:
            layer = np.load(path, mmap_mode="r")
        return layer

    def _sum_layers(self, paths: list[str]) -> np.ndarray | None:
        if not paths:
            return None
        total = np.zeros(
            (len(self._layer(paths[0])), self.channels), np.float32
        )
        for path in paths:
            total += self._layer(path)
        return total

    def clear(self) -> None:
        with self._lock:
            self._armed = False
            self._rec = None
            self._mix = None
            self._layers = []
            self._pos = 0
            folder, self._dir = self._dir, None
            if self.pending_bpm is not None:
                self.bpm, self.pending_bpm = self.pending_bpm, None
        with self._unsaved_lock:
            self._unsaved.clear()
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)

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
            from korshmer.utils.audio_host import (
                open_and_start_output_stream,
                sounddevice_module,
            )

            def open_stream(**kw):
                return open_and_start_output_stream(sounddevice_module(), **kw)

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
        writer, self._writer = self._writer, None
        if writer is not None:
            self._jobs.put(None)
            writer.join(timeout=5.0)
        self.clear()
