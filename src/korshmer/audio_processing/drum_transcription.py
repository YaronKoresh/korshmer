import math

import librosa
import numpy as np
import pretty_midi

from korshmer.utils.tempo import validate_bpm

SAMPLE_RATE = 44100
MIN_SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

KICK, SNARE = 36, 38
HAT_CLOSED, HAT_OPEN, CRASH = 42, 46, 49
TOM_LOW, TOM_FLOOR_MID, TOM_MID, TOM_HIGH = 41, 45, 47, 50

LOW_BAND = (30.0, 150.0)
BODY_BAND = (150.0, 600.0)
NOISE_BAND = (2000.0, 8000.0)
HIGH_BAND = (7000.0, 16000.0)

MIN_GAP_SECONDS = {"low": 0.06, "mid": 0.05, "high": 0.035}
PEAK_DELTA = {"low": 0.12, "mid": 0.1, "high": 0.08}
PEAK_FLOOR = 0.06
SNARE_NOISE_RATIO = 0.25
SNARE_MAX_NOISE_RATIO = 6.0
KICK_LOW_TO_BODY = 1.5
PAD_FRAMES = 12
BLEED_TOLERANCE_SECONDS = 0.03
BLEED_MARGIN = 2.0
BLEED_QUANTILE = 25.0
NOISE_FLOOR_FACTOR = 5.0
OPEN_DECAY_SECONDS = 0.14
CRASH_DECAY_SECONDS = 0.6
DECAY_FRACTION = 0.25
MAX_DECAY_SECONDS = 1.2
MIN_VELOCITY = 85
MAX_VELOCITY = 127
HIT_LENGTH = {"low": 0.12, "mid": 0.1, "high": 0.06}
EPSILON = 1e-9


def _band_slice(freqs, band):
    low, high = band
    selected = np.where((freqs >= low) & (freqs < high))[0]
    return slice(int(selected[0]), int(selected[-1]) + 1)


def _flux(log_spec, band_slice):
    band = log_spec[band_slice]
    silence = np.zeros((band.shape[0], 1), dtype=band.dtype)
    rise = np.maximum(np.diff(band, axis=1, prepend=silence), 0.0)
    envelope = rise.sum(axis=0)
    peak = float(np.max(envelope)) if envelope.size else 0.0
    return envelope / peak if peak > 0 else envelope


def _pick(envelope, name, frame_rate):
    if envelope.size < 8 or not np.any(envelope > 0):
        return np.zeros(0, dtype=int)
    wait = max(1, int(round(MIN_GAP_SECONDS[name] * frame_rate)))
    padded = np.concatenate([np.zeros(PAD_FRAMES), envelope])
    picked = librosa.util.peak_pick(
        padded,
        pre_max=3,
        post_max=3,
        pre_avg=12,
        post_avg=12,
        delta=PEAK_DELTA[name],
        wait=wait,
    )
    frames = [int(f) - PAD_FRAMES for f in picked if f >= PAD_FRAMES]
    floor = max(PEAK_FLOOR, NOISE_FLOOR_FACTOR * float(np.median(envelope)))
    return np.array([f for f in frames if envelope[f] >= floor], dtype=int)


def _window_energy(magnitude, band_slice, frame, frames=6):
    stop = min(magnitude.shape[1], frame + frames)
    return float(np.mean(magnitude[band_slice, frame:stop]))


def _decay_seconds(magnitude, band_slice, frame, frame_rate):
    track = magnitude[band_slice].mean(axis=0)
    stop = min(len(track), frame + int(MAX_DECAY_SECONDS * frame_rate))
    segment = track[frame:stop]
    if segment.size == 0:
        return 0.0
    top = int(np.argmax(segment[: max(2, int(0.03 * frame_rate))]))
    peak = float(segment[top])
    if peak <= 0:
        return 0.0
    below = np.where(segment[top:] < DECAY_FRACTION * peak)[0]
    length = below[0] if below.size else segment.size - top
    return float(length) / frame_rate


def _tom_pitch(magnitude, freqs, frame):
    stop = min(magnitude.shape[1], frame + 6)
    body = _band_slice(freqs, (60.0, 400.0))
    profile = magnitude[body, frame:stop].mean(axis=1)
    if profile.sum() <= 0:
        return TOM_MID
    centroid = float(np.sum(freqs[body] * profile) / profile.sum())
    if centroid < 120.0:
        return TOM_LOW
    if centroid < 170.0:
        return TOM_FLOOR_MID
    if centroid < 230.0:
        return TOM_MID
    return TOM_HIGH


def _velocity(strength, reference):
    share = min(1.0, strength / max(reference, EPSILON))
    return int(round(MIN_VELOCITY + (MAX_VELOCITY - MIN_VELOCITY) * share))


def _near(frames, frame, tolerance):
    return frames.size > 0 and bool(np.any(np.abs(frames - frame) <= tolerance))


def _validate_sample_rate(sample_rate):
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float, np.integer, np.floating))
        or not math.isfinite(sample_rate)
        or sample_rate < MIN_SAMPLE_RATE
    ):
        raise ValueError(
            f"Sample rate must be a number of at least {MIN_SAMPLE_RATE} Hz, "
            f"got {sample_rate!r}"
        )
    return float(sample_rate)


def _mono(audio):
    y = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if y.ndim == 2:
        channel_axis = int(np.argmin(y.shape))
        return y.mean(axis=channel_axis)
    if y.ndim != 1:
        raise ValueError(f"Audio must be 1-D or 2-D, got {y.ndim} dimensions")
    return y


def find_hits(audio, sample_rate=SAMPLE_RATE):
    sample_rate = _validate_sample_rate(sample_rate)
    y = _mono(audio)
    if y.size < N_FFT * 2 or float(np.max(np.abs(y))) < 1e-4:
        return []
    raw = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    magnitude = np.maximum(raw - np.median(raw, axis=1, keepdims=True), 0.0)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    log_spec = np.log1p(10.0 * raw)
    frame_rate = sample_rate / HOP_LENGTH
    tolerance = max(1, int(round(BLEED_TOLERANCE_SECONDS * frame_rate)))

    low = _band_slice(freqs, LOW_BAND)
    body = _band_slice(freqs, BODY_BAND)
    noise = _band_slice(freqs, NOISE_BAND)
    high = _band_slice(freqs, HIGH_BAND)

    low_flux = _flux(log_spec, low)
    mid_flux = np.maximum(_flux(log_spec, body), _flux(log_spec, noise))
    high_flux = _flux(log_spec, high)
    low_frames = _pick(low_flux, "low", frame_rate)
    mid_frames = _pick(mid_flux, "mid", frame_rate)
    high_frames = _pick(high_flux, "high", frame_rate)

    hits = []
    kicks = []
    for frame in low_frames:
        low_energy = _window_energy(magnitude, low, frame)
        body_energy = _window_energy(magnitude, body, frame)
        if low_energy >= KICK_LOW_TO_BODY * body_energy:
            kicks.append(frame)
            hits.append((frame, KICK, float(low_flux[frame]), "low"))
    low_frames = np.array(kicks, dtype=int)
    kick_noise_ratio = 0.0
    kick_body_ratio = 0.0
    if kicks:
        lows = [max(_window_energy(magnitude, low, f), EPSILON) for f in kicks]
        noises = [_window_energy(magnitude, noise, f) for f in kicks]
        bodies = [_window_energy(magnitude, body, f) for f in kicks]
        kick_noise_ratio = float(
            np.percentile(np.array(noises) / lows, BLEED_QUANTILE)
        )
        kick_body_ratio = float(
            np.percentile(np.array(bodies) / lows, BLEED_QUANTILE)
        )

    for frame in mid_frames:
        body_energy = _window_energy(magnitude, body, frame)
        noise_energy = _window_energy(magnitude, noise, frame)
        coincides = _near(low_frames, frame, tolerance)
        if coincides:
            nearest = low_frames[np.argmin(np.abs(low_frames - frame))]
            kick_low = _window_energy(magnitude, low, nearest)
            if noise_energy <= BLEED_MARGIN * kick_noise_ratio * kick_low:
                continue
            body_energy = max(body_energy - kick_body_ratio * kick_low, EPSILON)
        ratio = noise_energy / max(body_energy, EPSILON)
        if ratio > SNARE_MAX_NOISE_RATIO:
            continue
        if ratio >= SNARE_NOISE_RATIO:
            hits.append((frame, SNARE, float(mid_flux[frame]), "mid"))
        else:
            hits.append(
                (
                    frame,
                    _tom_pitch(magnitude, freqs, frame),
                    float(mid_flux[frame]),
                    "mid",
                )
            )

    for frame in high_frames:
        decay = _decay_seconds(magnitude, high, frame, frame_rate)
        if decay >= CRASH_DECAY_SECONDS:
            pitch = CRASH
        elif decay >= OPEN_DECAY_SECONDS:
            pitch = HAT_OPEN
        else:
            pitch = HAT_CLOSED
        hits.append((frame, pitch, float(high_flux[frame]), "high"))

    references = {}
    for _, _, strength, kind in hits:
        references.setdefault(kind, []).append(strength)
    references = {
        kind: float(np.percentile(values, 95))
        for kind, values in references.items()
    }
    result = [
        (
            float(frame) / frame_rate,
            pitch,
            _velocity(strength, references[kind]),
            kind,
        )
        for frame, pitch, strength, kind in hits
    ]
    return sorted(result, key=lambda hit: (hit[0], hit[1]))


def drum_midi_from_audio(audio, sample_rate=SAMPLE_RATE, tempo=120.0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=validate_bpm(tempo))
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for seconds, pitch, velocity, kind in find_hits(audio, sample_rate):
        kit.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=seconds,
                end=seconds + HIT_LENGTH[kind],
            )
        )
    midi.instruments.append(kit)
    return midi


def transcribe_drums(audio_path, tempo=120.0):
    audio, rate = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return drum_midi_from_audio(audio, rate, tempo)
