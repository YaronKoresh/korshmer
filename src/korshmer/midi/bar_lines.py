import numpy as np

from korshmer.midi_generation.song_match import estimate_offset

BEATS_PER_BAR = 4


def _pickup(pm):
    try:
        if pm.time_signature_changes:
            return None
        tempos = pm.get_tempo_changes()[1]
        if len(tempos) != 1 or tempos[0] <= 0:
            return None
        bpm = float(tempos[0])
        offset = estimate_offset(pm, bpm)
    except (ValueError, TypeError, IndexError):
        return None
    if offset <= 0:
        return None
    return bpm, offset


def bar_lines(pm):
    found = _pickup(pm)
    if found is None:
        return np.asarray(pm.get_downbeats(), dtype=np.float64)
    bpm, offset = found
    bar = BEATS_PER_BAR * 60.0 / bpm
    end = max(float(pm.get_end_time()), offset)
    return offset + bar * np.arange(int((end - offset) // bar) + 1)


def beat_lines(pm):
    found = _pickup(pm)
    if found is None:
        return np.asarray(pm.get_beats(), dtype=np.float64)
    bpm, offset = found
    beat = 60.0 / bpm
    first = offset - int(offset // beat) * beat
    end = max(float(pm.get_end_time()), offset)
    return first + beat * np.arange(int((end - first) // beat) + 1)
