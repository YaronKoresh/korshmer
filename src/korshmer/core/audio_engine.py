import gc
import logging
import threading
import time as _time
from enum import IntEnum

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QObject, QTimer, Signal

from korshmer.audio.loader import (
    StreamedAudio,
    check_audio_file_size,
    check_audio_file_type,
    check_decoded_audio_memory,
    open_streamed_audio,
    read_audio_frames,
    validate_buffer_size,
)
from korshmer.constants import (
    AUDIO_DEVICE_OPEN_ATTEMPTS,
    AUDIO_DEVICE_RETRY_DELAY_SECONDS,
    BUFFER_CPU_HIGH_PERCENT,
    BUFFER_CPU_LOW_PERCENT,
    BUFFER_LOW_LOAD_SECONDS,
    BUFFER_UNDERRUN_THRESHOLD,
    BUFFER_UNDERRUN_WINDOW_SECONDS,
    DEFAULT_PLAYBACK_BUFFER_SIZE,
    MAX_PLAYBACK_BUFFER_SIZE,
    MIN_PLAYBACK_BUFFER_SIZE,
    PERFORMANCE_LOGGER_NAME,
    SEGMENT_CACHE_MAX_BYTES,
)
from korshmer.core.looper import Looper
from korshmer.core.sample_trigger import SampleBank
from korshmer.core.segment_cache import SegmentCache
from korshmer.exceptions import AudioFileTooLargeError, AudioFormatError
from korshmer.utils.audio_host import (
    active_output_latency,
    open_and_start_output_stream,
)
from korshmer.utils.playback import validate_loop_points
from korshmer.utils.tempo import validate_bpm
from korshmer.utils.tracked_lock import TrackedLock
from korshmer.workers.audio_worker import AudioPrefetcher


class PlaybackState(IntEnum):
    STOPPED = 0
    PLAYING = 1
    PAUSED = 2


class AudioEngine(QObject):
    positionChanged = Signal(int)
    playbackStateChanged = Signal(int)
    volumeChanged = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.data: np.ndarray | None = None
        self.fs: int = 44100
        self.idx: int = 0
        self.loop_start_idx: int | None = None
        self.loop_end_idx: int | None = None
        self.stream: sd.OutputStream | None = None
        self._volume: float = 1.0
        self.current_path: str | None = None
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)

        self._state: PlaybackState = PlaybackState.STOPPED
        self._state_lock = TrackedLock("AudioEngine.state", reentrant=True)
        self._retry_cancel = threading.Event()
        self._logger = logging.getLogger(__name__)

        self.samples = SampleBank(self.fs)
        self.looper = Looper(sample_rate=self.fs)
        self._segment_cache = SegmentCache(SEGMENT_CACHE_MAX_BYTES)
        self._prefetch: AudioPrefetcher | None = None

        self._default_buffer_size: int = DEFAULT_PLAYBACK_BUFFER_SIZE
        self._current_buffer_size: int = self._default_buffer_size
        self._min_buffer_size: int = MIN_PLAYBACK_BUFFER_SIZE
        self._max_buffer_size: int = MAX_PLAYBACK_BUFFER_SIZE

        self._stream_start_time: float | None = None
        self._stream_start_idx: int = 0

        self._underrun_count: int = 0
        self._total_underruns: int = 0
        self._logged_underruns: int = 0
        self._perf_logger = logging.getLogger(PERFORMANCE_LOGGER_NAME)
        self._underrun_window_start: float | None = None
        self._low_load_since: float | None = None

    def _transition_state(
        self, new_state: PlaybackState, caller: str = ""
    ) -> bool:
        with self._state_lock:
            old_state = self._state

            valid_transitions = {
                PlaybackState.STOPPED: {PlaybackState.PLAYING},
                PlaybackState.PLAYING: {
                    PlaybackState.PAUSED,
                    PlaybackState.STOPPED,
                },
                PlaybackState.PAUSED: {
                    PlaybackState.PLAYING,
                    PlaybackState.STOPPED,
                },
            }

            if new_state == old_state:
                return True

            if new_state not in valid_transitions.get(old_state, set()):
                self._logger.warning(
                    f"Invalid state transition: {old_state.name} -> {new_state.name} "
                    f"(caller: {caller or 'unknown'})"
                )
                return False

            self._state = new_state
            self._logger.info(
                f"Playback state: {old_state.name} -> {new_state.name} "
                f"(caller: {caller or 'unknown'})"
            )
            self.playbackStateChanged.emit(new_state.value)
            return True

    def _check_memory_usage(self, threshold: float = 0.8) -> bool:
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0

            if memory_percent >= threshold:
                self._logger.warning(
                    f"High memory usage detected: {memory_percent * 100:.1f}% "
                    f"(threshold: {threshold * 100:.1f}%)"
                )
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to check memory usage: {e}")
            return False

    def _adjust_buffer_size(
        self, cpu_percent: float | None = None, now: float | None = None
    ) -> int:
        if now is None:
            now = _time.monotonic()
        if cpu_percent is None:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
            except Exception as e:
                self._logger.error(f"Failed to read CPU load: {e}")
                cpu_percent = 0.0
        with self._state_lock:
            if self._underrun_window_start is None:
                self._underrun_window_start = now
            underruns = self._underrun_count
            old_size = self._current_buffer_size

            if (
                underruns >= BUFFER_UNDERRUN_THRESHOLD
                or cpu_percent >= BUFFER_CPU_HIGH_PERCENT
            ):
                new_size = min(self._max_buffer_size, old_size * 2)
                self._reset_buffer_tracking(now)
                reason = f"{underruns} underruns, CPU {cpu_percent:.0f}%"
            elif cpu_percent <= BUFFER_CPU_LOW_PERCENT and underruns == 0:
                new_size = old_size
                reason = f"low CPU load {cpu_percent:.0f}%"
                if self._low_load_since is None:
                    self._low_load_since = now
                elif now - self._low_load_since >= BUFFER_LOW_LOAD_SECONDS:
                    new_size = max(self._min_buffer_size, old_size // 2)
                    self._reset_buffer_tracking(now)
            else:
                new_size = old_size
                reason = ""
                self._low_load_since = None
                if (
                    now - self._underrun_window_start
                    >= BUFFER_UNDERRUN_WINDOW_SECONDS
                ):
                    self._reset_buffer_tracking(now)

            if new_size != old_size:
                self._current_buffer_size = new_size
                self._logger.info(
                    f"Playback buffer size {old_size} -> {new_size} "
                    f"({reason}); applies on next stream start"
                )
            return self._current_buffer_size

    def _reset_buffer_tracking(self, now: float) -> None:
        self._underrun_count = 0
        self._underrun_window_start = now
        self._low_load_since = None

    def _reduce_buffer_size(self) -> None:
        old_size = self._current_buffer_size
        new_size = max(self._min_buffer_size, self._current_buffer_size // 2)

        if new_size < old_size:
            self._current_buffer_size = new_size
            self._logger.info(
                f"Reduced buffer size from {old_size} to {new_size} due to memory pressure "
                f"(min floor: {self._min_buffer_size})"
            )

            cleared = len(self._segment_cache)
            self._segment_cache.clear()
            self._logger.debug(f"Cleared {cleared} cached segments")

            gc.collect()
        else:
            self._logger.warning(
                f"Buffer size already at minimum floor ({self._min_buffer_size}), "
                "cannot reduce further"
            )

    def load(self, path: str) -> bool:
        try:
            if self._check_memory_usage(threshold=0.8):
                self._reduce_buffer_size()

            if self.stream:
                self.stop()
            self._stop_prefetch()
            self._segment_cache.clear()
            check_audio_file_size(path)
            check_audio_file_type(path)

            streamed = open_streamed_audio(path)
            if streamed is not None:
                d, fs = streamed, streamed.samplerate
            else:
                check_decoded_audio_memory(path)
                d, fs = sf.read(path, dtype="float32")

            if d is None or len(d) == 0:
                self._logger.error(
                    f"Failed to load audio: empty data from {path}"
                )
                return False

            if d.ndim == 1:
                d = d.reshape(-1, 1)

            device_info = sd.query_devices(kind="output")
            device_sr = int(device_info["default_samplerate"])

            if fs != device_sr:
                self._logger.warning(
                    f"Audio sample rate ({fs} Hz) differs from device rate ({device_sr} Hz). "
                    f"Resampling may be needed for optimal playback."
                )

            previous = self.data
            self.data = d
            self.fs = fs
            self.idx = 0
            self.clear_loop()
            self.current_path = path
            if isinstance(previous, StreamedAudio) and previous is not d:
                previous.close()
            self._logger.info(
                f"Loaded audio: {path} ({fs} Hz, {len(d)} samples)"
            )
            return True
        except FileNotFoundError:
            self._logger.error(f"Audio file not found: {path}")
            return False
        except (AudioFileTooLargeError, AudioFormatError) as e:
            self._logger.error(f"Refusing to load audio: {e}")
            return False
        except Exception as e:
            self._logger.error(f"Failed to load audio {path}: {e}")
            return False

    def read_segment(self, start_frame: int, stop_frame: int):
        path = self.current_path
        if path is None:
            return None
        key = (path, int(start_frame), int(stop_frame))
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached
        try:
            segment = read_audio_frames(path, start_frame, stop_frame)
        except (OSError, RuntimeError, ValueError) as e:
            self._logger.error(f"Failed to read segment from {path}: {e}")
            return None
        self._segment_cache.put(key, segment)
        return segment

    def play(self) -> None:
        with self._state_lock:
            if self.data is None:
                self._logger.warning("Cannot play: no audio data loaded")
                return

            if len(self.data) == 0:
                self._logger.error("Cannot play: audio data is empty")
                return

            if self._state == PlaybackState.PLAYING:
                return

            if not self._transition_state(PlaybackState.PLAYING, caller="play"):
                return

            if self.idx >= len(self.data):
                self._logger.info(
                    "Resetting playback position to start (was at end)"
                )
                self.idx = 0

            try:
                validate_buffer_size(self._current_buffer_size)
            except ValueError as e:
                self._logger.error(
                    f"Invalid playback buffer size, falling back to "
                    f"{self._default_buffer_size}: {e}"
                )
                self._current_buffer_size = self._default_buffer_size

            self._start_stream()

    def _open_and_start_stream(self) -> None:
        self._stream_start_time = None
        self._stream_start_idx = 0
        self.samples.configure(self.fs, self.data.shape[1])
        self.stream = open_and_start_output_stream(
            sd,
            samplerate=self.fs,
            channels=self.data.shape[1],
            callback=self._cb,
            latency="high",
            blocksize=self._current_buffer_size,
        )

    def _start_stream(self) -> None:
        self._retry_cancel.clear()
        try:
            for attempt in range(1, AUDIO_DEVICE_OPEN_ATTEMPTS + 1):
                try:
                    self._open_and_start_stream()
                    break
                except sd.PortAudioError as e:
                    self._logger.error(
                        f"Audio device error (attempt {attempt}/"
                        f"{AUDIO_DEVICE_OPEN_ATTEMPTS}): {e}"
                    )
                    self._close_failed_stream()
                    sd.default.reset()
                    if attempt == AUDIO_DEVICE_OPEN_ATTEMPTS:
                        raise
                    if self._retry_cancel.wait(
                        AUDIO_DEVICE_RETRY_DELAY_SECONDS
                    ):
                        self._logger.info("Audio device retry cancelled")
                        return
            self._log_output_latency()
            self.timer.start()
            self._start_prefetch()
        except sd.PortAudioError:
            self._transition_state(PlaybackState.STOPPED, caller="play_error")
        except Exception as e:
            self._logger.error(f"Playback start error: {e}")
            self._transition_state(PlaybackState.STOPPED, caller="play_error")

    def _close_failed_stream(self) -> None:
        stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.close()
        except Exception as e:
            self._logger.debug(f"Failed stream close error: {e}")

    def _log_output_latency(self) -> None:
        latency = active_output_latency()
        if latency > 0:
            self._perf_logger.info(
                f"Output latency: {latency * 1000:.1f} ms "
                f"(buffer {self._current_buffer_size} frames, {self.fs} Hz)"
            )

    def _log_audio_dropouts(self) -> None:
        total = self._total_underruns
        if total > self._logged_underruns:
            self._perf_logger.warning(
                f"Audio dropouts: +{total - self._logged_underruns} "
                f"(total {total})"
            )
            self._logged_underruns = total

    def _start_prefetch(self) -> None:

        data = self.data
        if data is None:
            return
        self._stop_prefetch()
        prefetch = AudioPrefetcher(
            lambda start, stop: data[start:stop],
            block_frames=self._current_buffer_size,
        )
        prefetch.start(self.idx, len(data))
        self._prefetch = prefetch

    def _stop_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch = None

    def _close_stream(self, action: str) -> None:

        self._stop_prefetch()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self._logger.error(f"Error closing stream on {action}: {e}")
            finally:
                self.stream = None
                self.timer.stop()

    def pause(self) -> None:
        with self._state_lock:
            if self._state != PlaybackState.PLAYING:
                return

            if not self._transition_state(PlaybackState.PAUSED, caller="pause"):
                return

            self._close_stream("pause")

    def stop(self) -> None:
        self._retry_cancel.set()
        with self._state_lock:
            if not self._transition_state(PlaybackState.STOPPED, caller="stop"):
                if self._state != PlaybackState.STOPPED:
                    return

            self._close_stream("stop")
            self.idx = 0

        self.positionChanged.emit(0)

    def setPosition(self, ms: int) -> None:
        with self._state_lock:
            self.idx = int((ms / 1000) * self.fs)

            if self.stream is not None:
                self._stream_start_time = None
            if self._prefetch is not None:
                self._prefetch.seek(self.idx)

        self.positionChanged.emit(self.position())

    def position(self) -> int:
        return int((self.idx / self.fs) * 1000) if self.fs else 0

    def duration(self) -> int:
        return (
            int((len(self.data) / self.fs) * 1000)
            if (self.data is not None and self.fs)
            else 0
        )

    def playbackState(self) -> int:
        with self._state_lock:
            return self._state.value

    def setVolume(self, v: float) -> None:
        self._volume = v
        self.volumeChanged.emit(v)

    def volume(self) -> float:
        return self._volume

    def set_tempo(self, bpm: float) -> None:
        bpm = validate_bpm(bpm)
        self.looper.set_bpm(bpm)

    def set_loop(self, start_ms, end_ms) -> bool:
        if self.data is None or self.duration() <= 0:
            self._logger.warning("Cannot set loop: no audio loaded")
            return False

        points = validate_loop_points(start_ms, end_ms, self.duration())
        if points is None:
            return False
        start_ms, end_ms = points

        start_idx = int((start_ms / 1000) * self.fs)
        end_idx = min(int((end_ms / 1000) * self.fs), len(self.data))
        if start_idx >= end_idx:
            self._logger.warning(
                "Rejected loop: region collapses to less than one sample "
                f"({start_idx} >= {end_idx})"
            )
            return False

        with self._state_lock:
            self.loop_start_idx = start_idx
            self.loop_end_idx = end_idx
        return True

    def clear_loop(self) -> None:
        with self._state_lock:
            self.loop_start_idx = None
            self.loop_end_idx = None
            if self._prefetch is not None:
                self._prefetch.seek(self.idx)

    def _cb(self, outdata: np.ndarray, frames: int, time, status) -> None:
        try:
            self._cb_main(outdata, frames, time, status)
        finally:
            samples = self.samples
            if samples.channels == outdata.shape[1]:
                samples.mix_into(outdata, self._volume)

    def _cb_main(self, outdata: np.ndarray, frames: int, time, status) -> None:

        if status:
            self._logger.warning(f"Audio stream status: {status}")
            if getattr(status, "output_underflow", True):
                self._underrun_count += 1
                self._total_underruns += 1

        chunk = len(outdata)
        if self.data is None or len(self.data) == 0:
            self._logger.error(
                "Audio callback called with no data, filling with silence"
            )
            outdata.fill(0)
            raise sd.CallbackStop

        loop_start = self.loop_start_idx
        loop_end = self.loop_end_idx
        if loop_start is not None and loop_end is not None:
            filled = 0
            while filled < chunk:
                if self.idx >= loop_end or self.idx < loop_start:
                    self.idx = loop_start
                    self._stream_start_time = None
                n = min(chunk - filled, loop_end - self.idx)
                outdata[filled : filled + n] = (
                    self.data[self.idx : self.idx + n] * self._volume
                )
                self.idx += n
                filled += n
            return

        if time and hasattr(time, "outputBufferDacTime"):
            if self._stream_start_time is None:
                self._stream_start_time = time.outputBufferDacTime
                self._stream_start_idx = self.idx
            else:
                elapsed = time.outputBufferDacTime - self._stream_start_time
                if elapsed > 0:
                    samples_played = self._stream_start_idx + int(
                        elapsed * self.fs
                    )

                    if abs(samples_played - self.idx) > self.fs // 10:
                        self.idx = samples_played

        if self.idx + chunk > len(self.data):
            rem = len(self.data) - self.idx
            outdata[:rem] = self.data[self.idx :] * self._volume
            outdata[rem:] = 0
            self.idx += rem
            raise sd.CallbackStop
        else:
            prefetch = self._prefetch
            block = (
                prefetch.take(self.idx, chunk) if prefetch is not None else None
            )
            if block is None:
                block = self.data[self.idx : self.idx + chunk]
            outdata[:] = block * self._volume
            self.idx += chunk

    def _tick(self) -> None:
        self.positionChanged.emit(self.position())
        self._adjust_buffer_size()
        self._log_audio_dropouts()
        loop_active = (
            self.loop_start_idx is not None and self.loop_end_idx is not None
        )
        if (
            not loop_active
            and self.data is not None
            and self.idx >= len(self.data)
        ):
            self.stop()
