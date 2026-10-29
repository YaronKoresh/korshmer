import os

import librosa
import numpy as np
import soundfile as sf

from korshmer.audio.loader import (
    check_audio_file_channels,
    check_audio_file_size,
    check_audio_file_type,
)

_SUBTYPE_MAP = {
    "wav": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT", 64: "DOUBLE"},
    "flac": {16: "PCM_16", 24: "PCM_24"},
    "aiff": {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"},
    "rf64": {16: "PCM_16", 32: "FLOAT", 64: "DOUBLE"},
}

_FORMAT_FROM_EXT = {
    ".wav": "WAV",
    ".flac": "FLAC",
    ".aiff": "AIFF",
    ".aif": "AIFF",
}


class AudioFile:
    __slots__ = ("_y", "_sr", "_bits", "_path")

    def __init__(self, y=None, sr=44100, bits=32, path=None):
        self._y = y
        self._sr = int(sr)
        self._bits = bits
        self._path = str(path) if path is not None else None

    @classmethod
    def load(cls, path, sr=None, mono=False, bits=32):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Path is not a file: {path}")
        check_audio_file_size(path)
        check_audio_file_type(path)
        if not mono:
            check_audio_file_channels(path)
        try:
            y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}") from e
        if y is None or len(y) == 0:
            raise ValueError("Loaded audio data is empty")
        return cls(y=y, sr=loaded_sr, bits=bits, path=path)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def sr(self):
        return self._sr

    @property
    def bits(self):
        return self._bits

    @property
    def path(self):
        return self._path

    @property
    def channels(self):
        if self._y is None:
            return 0
        if self._y.ndim == 1:
            return 1
        return self._y.shape[0]

    @property
    def samples(self):
        if self._y is None:
            return 0
        if self._y.ndim == 1:
            return self._y.shape[0]
        return self._y.shape[1]

    @property
    def duration(self):
        if self._y is None or self._sr == 0:
            return 0.0
        return self.samples / self._sr

    @property
    def is_mono(self):
        return self._y is not None and self._y.ndim == 1

    @property
    def is_stereo(self):
        return (
            self._y is not None and self._y.ndim == 2 and self._y.shape[0] == 2
        )

    def ensure_2d(self):
        if self._y is not None and self._y.ndim == 1:
            self._y = self._y[np.newaxis, :]
        return self

    def to_write_format(self):
        if self._y is None:
            return None
        if self._y.ndim == 2:
            return self._y.T
        return self._y

    def save(self, path, bits=None, fmt=None):
        if self._y is None:
            raise ValueError("No audio data to save")
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        ext = os.path.splitext(path)[1].lower()
        sf_format = fmt or _FORMAT_FROM_EXT.get(ext)
        b = bits or self._bits
        fmt_key = (sf_format or "WAV").lower()

        subtypes = _SUBTYPE_MAP.get(fmt_key, _SUBTYPE_MAP["wav"])
        subtype = subtypes.get(b)
        if subtype is None:
            b = max(subtypes.keys())
            subtype = subtypes[b]

        data = self.to_write_format()
        kwargs = {"subtype": subtype}
        if sf_format:
            kwargs["format"] = sf_format
        sf.write(path, data, self._sr, **kwargs)
        self._path = str(path)
        return path

    def save_rf64(self, path, bits=None):
        if self._y is None:
            raise ValueError("No audio data to save")
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        b = bits or self._bits
        y = np.clip(self._y, -1.0, 1.0)

        if b == 16:
            int16_max = np.iinfo(np.int16).max
            int16_min = np.iinfo(np.int16).min
            y_scaled = y * float(int16_max)
            tpdf = (
                np.random.uniform(-1.0, 1.0, y_scaled.shape)
                + np.random.uniform(-1.0, 1.0, y_scaled.shape)
            ) * 0.5
            y = np.clip(np.rint(y_scaled + tpdf), int16_min, int16_max).astype(
                np.int16
            )
        elif b == 32:
            y = y.astype(np.float32)
        elif b == 64:
            y = y.astype(np.float64)
        else:
            raise ValueError(f"Unsupported bit depth: {b}. Use 16, 32, or 64.")

        if y.ndim == 2 and y.shape[0] < y.shape[1]:
            y = y.T

        subtype_map = {16: "PCM_16", 32: "FLOAT", 64: "DOUBLE"}
        sf.write(path, y, self._sr, subtype=subtype_map[b], format="RF64")
        self._path = str(path)
        return path

    def copy(self):
        y_copy = self._y.copy() if self._y is not None else None
        return AudioFile(
            y=y_copy, sr=self._sr, bits=self._bits, path=self._path
        )

    def __repr__(self):
        ch = self.channels
        dur = f"{self.duration:.2f}s"
        return f"AudioFile(sr={self._sr}, channels={ch}, duration={dur}, bits={self._bits})"
