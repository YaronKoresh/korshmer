import math
import numbers
import os
import threading

import numpy as np
import psutil
import soundfile as sf

from korshmer.audio_processing.validation import validate_channel_count
from korshmer.constants import (
    DECODED_AUDIO_BYTES_PER_SAMPLE,
    MAX_AUDIO_CHANNELS,
    MAX_AUDIO_FILE_SIZE_BYTES,
    MAX_DECODED_AUDIO_BYTES,
    MAX_PLAYBACK_BUFFER_SIZE,
    MIN_PLAYBACK_BUFFER_SIZE,
    STREAMING_AUDIO_THRESHOLD_BYTES,
)
from korshmer.exceptions import AudioFileTooLargeError, AudioFormatError


def _has_audio_signature(header: bytes) -> bool:
    if header[0:4] == b"RIFF" and header[8:12] == b"WAVE":
        return True
    if header[0:4] == b"RF64":
        return True
    if header[0:4] == b"fLaC":
        return True
    if header[0:4] == b"OggS":
        return True
    if header[0:4] == b"FORM" and header[8:12] in (b"AIFF", b"AIFC"):
        return True
    if header[4:8] == b"ftyp":
        return True
    if header[0:3] == b"ID3":
        return True

    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    return False


def check_audio_file_type(path):
    with open(path, "rb") as handle:
        header = handle.read(12)
    if not _has_audio_signature(header):
        raise AudioFormatError(
            f"File content is not a recognized audio format: {path}",
            recovery_hint=(
                "Make sure the file is a real WAV, MP3, FLAC, OGG, AIFF or "
                "MP4/M4A audio file, not just renamed to an audio extension."
            ),
        )
    return True


def validate_buffer_size(value):
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and math.isnan(value):
            raise ValueError("Playback buffer size must not be NaN")
        raise ValueError(
            f"Playback buffer size must be an integer, got "
            f"{type(value).__name__}: {value!r}"
        )
    if not MIN_PLAYBACK_BUFFER_SIZE <= value <= MAX_PLAYBACK_BUFFER_SIZE:
        raise ValueError(
            f"Playback buffer size must be between "
            f"{MIN_PLAYBACK_BUFFER_SIZE} and {MAX_PLAYBACK_BUFFER_SIZE} "
            f"samples, got {value}"
        )
    return value


def _format_bytes(n):
    return f"{n} bytes ({n / (1024 * 1024 * 1024):.2f} GB)"


def _require_channel_count(path, channels, max_channels):
    try:
        return validate_channel_count(channels, max_channels)
    except ValueError:
        raise AudioFileTooLargeError(
            f"Audio file has {channels} channels; supported range is "
            f"1 to {max_channels}: {path}",
            recovery_hint=f"Mix the file down to {max_channels} channels "
            "or fewer before loading it.",
        ) from None


def check_audio_file_channels(path, max_channels=MAX_AUDIO_CHANNELS):
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return _require_channel_count(path, int(info.channels), max_channels)


def check_decoded_audio_memory(
    path, max_bytes=MAX_DECODED_AUDIO_BYTES, max_channels=MAX_AUDIO_CHANNELS
):
    info = sf.info(path)
    frames = int(info.frames)
    channels = _require_channel_count(path, int(info.channels), max_channels)
    if frames <= 0:
        raise AudioFormatError(
            f"Audio file header reports no audio frames ({frames}): {path}",
            recovery_hint="The file is empty or its header is damaged.",
        )

    required = frames * channels * DECODED_AUDIO_BYTES_PER_SAMPLE
    if required > max_bytes:
        raise AudioFileTooLargeError(
            f"Decoded audio would need {_format_bytes(required)} of memory; "
            f"limit is {_format_bytes(max_bytes)}: {path}",
            recovery_hint="Trim or split the file before loading it.",
        )

    available = psutil.virtual_memory().available
    if required > available:
        raise AudioFileTooLargeError(
            f"Decoded audio would need {_format_bytes(required)} of memory; "
            f"only {_format_bytes(available)} is available: {path}",
            recovery_hint="Close other programs or load a shorter file.",
        )
    return required


def read_audio_frames(path, start_frame, stop_frame):
    for name, value in (
        ("start_frame", start_frame),
        ("stop_frame", stop_frame),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f"{name} must be an integer, got {type(value).__name__}: "
                f"{value!r}"
            )
    start_frame = int(start_frame)
    stop_frame = int(stop_frame)
    with sf.SoundFile(str(path)) as handle:
        total = int(handle.frames)
        if not 0 <= start_frame < stop_frame <= total:
            raise ValueError(
                f"Frame range [{start_frame}, {stop_frame}) is outside the "
                f"file's {total} frames: {path}"
            )
        handle.seek(start_frame)
        return handle.read(
            stop_frame - start_frame, dtype="float32", always_2d=True
        )


class StreamedAudio:
    ndim = 2

    def __init__(self, path):
        self._lock = threading.Lock()
        self._handle = sf.SoundFile(str(path))
        self.frames = int(self._handle.frames)
        self.channels = int(self._handle.channels)
        self.samplerate = int(self._handle.samplerate)
        self.seekable = bool(self._handle.seekable())

    @property
    def shape(self):
        return (self.frames, self.channels)

    def __len__(self):
        return self.frames

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("StreamedAudio supports only unit-step slices")
        start, stop, _ = key.indices(self.frames)
        count = max(0, stop - start)
        with self._lock:
            if self._handle is None or count == 0:
                return np.zeros((count, self.channels), dtype="float32")
            self._handle.seek(start)
            return self._handle.read(count, dtype="float32", always_2d=True)

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def open_streamed_audio(path, min_bytes=None, max_channels=MAX_AUDIO_CHANNELS):
    if min_bytes is None:
        min_bytes = STREAMING_AUDIO_THRESHOLD_BYTES
    if os.path.getsize(path) <= min_bytes:
        return None
    try:
        stream = StreamedAudio(path)
    except (RuntimeError, OSError, ValueError):
        return None
    if not stream.seekable:
        stream.close()
        return None
    try:
        if stream.frames <= 0:
            raise AudioFormatError(
                f"Audio file header reports no audio frames ({stream.frames}): "
                f"{path}",
                recovery_hint="The file is empty or its header is damaged.",
            )
        _require_channel_count(path, stream.channels, max_channels)
    except Exception:
        stream.close()
        raise
    return stream


def check_audio_file_size(path, max_bytes=MAX_AUDIO_FILE_SIZE_BYTES):
    size = os.path.getsize(path)
    if size > max_bytes:
        size_gb = size / (1024 * 1024 * 1024)
        limit_gb = max_bytes / (1024 * 1024 * 1024)
        raise AudioFileTooLargeError(
            f"Audio file is too large: {size} bytes ({size_gb:.2f} GB); "
            f"limit is {max_bytes} bytes ({limit_gb:.2f} GB): {path}",
            recovery_hint=(
                "Trim or split the file so it is 2 GB or smaller before "
                "loading it."
            ),
        )
    return size
