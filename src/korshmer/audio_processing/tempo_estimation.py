from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter

from korshmer.audio_processing.validation import validate_sample_rate

HOP_LENGTH = 512
MIN_BEATS_FOR_FIT = 8
MAX_FIT_DEVIATION = 0.04
REFINE_SPAN = 0.015
REFINE_STEP = 0.0002
MIN_ALIGNMENT = 0.3
MIN_BPM_ESTIMATE = 20.0
MAX_BPM_ESTIMATE = 300.0
DECIMALS = 2
LOCAL_WINDOW_S = 16.0
LOCAL_STEP_S = 8.0
LOCAL_BEATS = 4
LOCAL_SEARCH = 0.12
LOCAL_MIN_CORRELATION = 0.1
LOCAL_MIN_WINDOWS = 3
LOCAL_AGREEMENT = 0.015
PROFILE_WINDOW_S = 8.0
PROFILE_STEP_S = 2.0
PROFILE_SMOOTH = 3
STEADY_TOLERANCE = 0.006
STEADY_MIN_S = 8.0
CHANGE_MIN = 0.02


@dataclass(frozen=True)
class TempoProfile:
    main_bpm: float
    times: np.ndarray
    bpms: np.ndarray
    sections: tuple[tuple[float, float, float], ...]

    @property
    def changes(self) -> bool:
        tempos = [bpm for _, _, bpm in self.sections]
        return bool(tempos) and max(tempos) - min(tempos) >= CHANGE_MIN * min(
            tempos
        )


def _beat_line_tempo(envelope, sr, candidate):
    _, beats = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=candidate,
        tightness=100,
        units="time",
    )
    if len(beats) < MIN_BEATS_FOR_FIT:
        return None
    slope = np.polyfit(np.arange(len(beats)), beats, 1)[0]
    if slope <= 0:
        return None
    fitted = 60.0 / float(slope)
    if abs(fitted - candidate) / candidate > MAX_FIT_DEVIATION:
        return None
    return fitted


def _window_tempo(window, frames_per_second, candidate):
    centred = window - window.mean()
    spectrum = np.fft.rfft(centred, 2 * len(centred))
    auto = np.fft.irfft(spectrum * np.conj(spectrum))[: len(centred)]
    if auto[0] <= 0:
        return None
    auto = auto / auto[0]
    lag = LOCAL_BEATS * 60.0 / candidate * frames_per_second
    first = int(lag * (1 - LOCAL_SEARCH))
    last = int(np.ceil(lag * (1 + LOCAL_SEARCH))) + 1
    if first < 1 or last >= len(auto) // 2:
        return None
    lags = np.arange(first, last)
    support = auto[lags] + auto[lags // 2] + auto[lags // LOCAL_BEATS]
    near = first + int(np.argmax(support))
    low, high = max(first, near - 2), min(last, near + 3)
    peak = low + int(np.argmax(auto[low:high]))
    if peak in (first, last - 1) or auto[peak] < LOCAL_MIN_CORRELATION:
        return None
    before, here, after = auto[peak - 1 : peak + 2]
    bend = before - 2 * here + after
    shift = 0.5 * (before - after) / bend if bend < 0 else 0.0
    return LOCAL_BEATS * 60.0 * frames_per_second / (peak + shift)


def local_tempos(
    envelope, sr, candidate, window_s=LOCAL_WINDOW_S, step_s=LOCAL_STEP_S
):
    frames_per_second = sr / HOP_LENGTH
    size = int(window_s * frames_per_second)
    step = int(step_s * frames_per_second)
    found = []
    for first in range(0, len(envelope) - size + 1, max(1, step)):
        tempo = _window_tempo(
            envelope[first : first + size], frames_per_second, candidate
        )
        if tempo is not None:
            found.append((first / frames_per_second, float(tempo)))
    return found


def _dominant_tempo(envelope, sr, candidate):
    tempos = np.array([t for _, t in local_tempos(envelope, sr, candidate)])
    if len(tempos) < LOCAL_MIN_WINDOWS:
        return None
    middle = float(np.median(tempos))
    agreeing = tempos[np.abs(tempos - middle) <= LOCAL_AGREEMENT * middle]
    if len(agreeing) < LOCAL_MIN_WINDOWS or len(agreeing) == len(tempos):
        return None
    return float(agreeing.mean())


def _refine_by_alignment(onset_times, tempo):
    times = np.asarray(onset_times, dtype=float)
    if len(times) < MIN_BEATS_FOR_FIT:
        return tempo
    steps = int(round(REFINE_SPAN / REFINE_STEP))
    tempos = tempo * (1.0 + REFINE_STEP * np.arange(-steps, steps + 1))
    best_tempo, best_alignment = tempo, 0.0
    for divisions in (1, 2):
        freqs = divisions * tempos / 60.0
        phases = np.exp(2j * np.pi * np.outer(freqs, times))
        alignment = np.abs(phases.mean(axis=1))
        index = int(np.argmax(alignment))
        if alignment[index] > best_alignment:
            best_alignment, best_tempo = (
                float(alignment[index]),
                float(tempos[index]),
            )
    return best_tempo if best_alignment >= MIN_ALIGNMENT else tempo


def steady_sections(times, bpms, step_s=PROFILE_STEP_S):
    sections = []
    first = 0
    count = len(bpms)
    while first < count:
        last = first
        while last + 1 < count:
            middle = np.median(bpms[first : last + 2])
            run = bpms[first : last + 2]
            if np.abs(run - middle).max() > STEADY_TOLERANCE * middle:
                break
            last += 1
        begin = float(times[first] - step_s / 2)
        end = float(times[last] + step_s / 2)
        if end - begin >= STEADY_MIN_S:
            tempo = float(np.median(bpms[first : last + 1]))
            sections.append((max(0.0, begin), end, tempo))
        first = last + 1
    return tuple(sections)


def tempo_profile(y, sr):
    main = estimate_tempo(y, sr)
    if main is None:
        return None
    envelope = librosa.onset.onset_strength(
        y=np.asarray(y, dtype=np.float32), sr=sr, hop_length=HOP_LENGTH
    )
    found = local_tempos(envelope, sr, main, PROFILE_WINDOW_S, PROFILE_STEP_S)
    if len(found) < PROFILE_SMOOTH:
        return TempoProfile(main, np.zeros(0), np.zeros(0), ())
    times = np.array([t for t, _ in found]) + PROFILE_WINDOW_S / 2
    bpms = median_filter(
        np.array([b for _, b in found]), size=PROFILE_SMOOTH, mode="nearest"
    )
    return TempoProfile(main, times, bpms, steady_sections(times, bpms))


def tempo_profile_of_file(path):
    try:
        y, sr = librosa.load(path, sr=None)
        return tempo_profile(y, sr)
    except Exception as e:
        print(f"Tempo profile failed: {e}")
        return None


def estimate_tempo(y, sr):
    sr = validate_sample_rate(sr)
    audio = np.asarray(y, dtype=np.float32)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        return None
    envelope = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )
    candidate = float(
        librosa.feature.tempo(
            onset_envelope=envelope, sr=sr, hop_length=HOP_LENGTH
        )[0]
    )
    if not np.isfinite(candidate) or not (
        MIN_BPM_ESTIMATE <= candidate <= MAX_BPM_ESTIMATE
    ):
        return None
    tempo = (
        _dominant_tempo(envelope, sr, candidate)
        or _beat_line_tempo(envelope, sr, candidate)
        or candidate
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sr, hop_length=HOP_LENGTH, units="time"
    )
    tempo = _refine_by_alignment(onsets, tempo)
    return round(float(tempo), DECIMALS)
