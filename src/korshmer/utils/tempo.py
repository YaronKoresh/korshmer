import logging
import math

import numpy as np

_logger = logging.getLogger(__name__)

MIN_BPM = 20.0
MAX_BPM = 300.0
DEFAULT_BPM = 120.0
BPM_DECIMALS = 2


def scalar_tempo(tempo) -> float:
    values = np.atleast_1d(tempo)
    return float(values[0]) if values.size else 0.0


def validate_bpm(value, default=DEFAULT_BPM):
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        _logger.warning(
            "Invalid BPM value %r; using default %s", value, default
        )
        return float(default)

    if not math.isfinite(bpm):
        _logger.warning(
            "Non-finite BPM value %r; using default %s", value, default
        )
        return float(default)

    if bpm <= 0:
        _logger.warning(
            "Non-positive BPM value %r; using default %s", value, default
        )
        return float(default)

    if bpm < MIN_BPM:
        _logger.warning("BPM %s below minimum; clamping to %s", bpm, MIN_BPM)
        return MIN_BPM

    if bpm > MAX_BPM:
        _logger.warning("BPM %s above maximum; clamping to %s", bpm, MAX_BPM)
        return MAX_BPM

    return bpm


TAP_RESET_SECONDS = 2.0
TAP_MAX_TAPS = 8


class TapTempo:
    def __init__(self, reset_seconds=TAP_RESET_SECONDS, max_taps=TAP_MAX_TAPS):
        self.reset_seconds = float(reset_seconds)
        self.max_taps = max(2, int(max_taps))
        self._taps = []

    def reset(self):
        self._taps = []

    def tap(self, timestamp):
        try:
            now = float(timestamp)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(now):
            return None
        if self._taps and (
            now <= self._taps[-1] or now - self._taps[-1] > self.reset_seconds
        ):
            self._taps = []
        self._taps.append(now)
        self._taps = self._taps[-self.max_taps :]
        if len(self._taps) < 2:
            return None
        interval = (self._taps[-1] - self._taps[0]) / (len(self._taps) - 1)
        return validate_bpm(60.0 / interval)


def midi_file_tempo(path):
    import mido

    try:
        midi = mido.MidiFile(str(path))
    except (OSError, ValueError, EOFError, KeyError, IndexError):
        return None
    first = None
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo" and message.tempo > 0:
                if first is None or tick < first[0]:
                    first = (tick, message.tempo)
                break
    if first is None:
        return None
    return validate_bpm(60_000_000.0 / first[1])
