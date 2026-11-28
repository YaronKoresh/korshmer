import random

import numpy as np
from scipy import signal

DRUMS, BASS, PADS, BAND, LEAD = "drums", "bass", "pads", "band", "lead"
BASS_PROGRAMS = frozenset({*range(32, 40), 43, 58})
DRIVEN_BASS = 87
DRIVEN_BASS_TOP = 55
PAD_PROGRAMS = frozenset({*range(16, 24), *range(40, 56), *range(88, 104)})
BAND_PROGRAMS = frozenset({*range(0, 8), *range(24, 32)})
KICKS = frozenset({35, 36})

TIMING_SPREAD = 0.006
TIMING_LIMIT = 0.014
VELOCITY_SPREAD = 6
DRUM_VELOCITY_SPREAD = 4
SHORTEST_NOTE = 0.02

ROOM_TAPS = (
    (0.007, 0.32, 0),
    (0.011, 0.27, 1),
    (0.017, 0.22, 0),
    (0.023, 0.18, 1),
    (0.031, 0.13, 0),
    (0.041, 0.09, 1),
)
ROOM_TONE_HZ = 6000.0
ROOM_MIX = {DRUMS: 0.16, BASS: 0.0, PADS: 0.2, BAND: 0.2, LEAD: 0.14}
DOUBLE_DELAYS = (0.011, 0.017)
DOUBLE_MIX = {PADS: 0.3, BAND: 0.28}
ECHO_BEATS = 0.75
ECHO_FEEDBACK = 0.35
ECHO_TONE_HZ = (300.0, 5000.0)
ECHO_MIX = 0.1
HIGHPASS_HZ = {BASS: 30.0, PADS: 120.0, BAND: 80.0, LEAD: 90.0}
MONO_BELOW_HZ = 120.0
PRESENCE = (3000.0, 2.0, 0.9)
BASS_DRIVE = (1.8, 0.3)
PUNCH = (-24.0, 4.0, 0.004, 0.35)
DUCK_DB = {BASS: 3.0, PADS: 2.0}
DUCK_RELEASE = 0.18
GLUE = (-16.0, 1.8, 0.08)
TAPE_DRIVE = (1.3, 0.4)
TAPE_TONE_HZ = 16000.0
TAPE_WOW = (0.00008, 0.6)
GAIN_SMOOTH = 0.03


def bus_of(instrument):
    if instrument.is_drum:
        return DRUMS
    program = instrument.program
    if program in BASS_PROGRAMS:
        return BASS
    if program == DRIVEN_BASS and instrument.notes:
        pitches = sorted(n.pitch for n in instrument.notes)
        if pitches[len(pitches) // 2] <= DRIVEN_BASS_TOP:
            return BASS
    if program in PAD_PROGRAMS:
        return PADS
    if program in BAND_PROGRAMS:
        return BAND
    return LEAD


def humanized_notes(instrument, seed, keep_time=False):
    rng = random.Random(f"humanize:{seed}")
    spread = DRUM_VELOCITY_SPREAD if instrument.is_drum else VELOCITY_SPREAD
    keep_time = keep_time or instrument.is_drum
    played = []
    for note in instrument.notes:
        start, end = note.start, note.end
        if not keep_time:
            shift = max(
                -TIMING_LIMIT, min(TIMING_LIMIT, rng.gauss(0.0, TIMING_SPREAD))
            )
            start = max(0.0, start + shift)
            end = max(start + SHORTEST_NOTE, end + shift)
        velocity = max(
            1, min(127, note.velocity + rng.randint(-spread, spread))
        )
        played.append((start, end, note.pitch, velocity))
    return played


def _stereo(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    return audio


def _one_pole(values, seconds, sr):
    if seconds <= 0:
        return values
    pole = float(np.exp(-1.0 / (sr * seconds)))
    return signal.lfilter([1.0 - pole], [1.0, -pole], values, axis=0)


def _filter(audio, sr, hz, kind, order=2):
    nyquist = sr / 2.0
    if not 0 < hz < nyquist:
        return audio
    sos = signal.butter(order, hz / nyquist, btype=kind, output="sos")
    return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def highpass(audio, sr, hz):
    return _filter(audio, sr, hz, "highpass")


def lowpass(audio, sr, hz, order=2):
    return _filter(audio, sr, hz, "lowpass", order)


def peaking(audio, sr, hz, gain_db, q):
    w0 = 2 * np.pi * hz / sr
    a = 10 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.lfilter(b, den, audio, axis=0).astype(np.float32)


def compress(audio, sr, threshold_db, ratio, seconds):
    power = np.mean(np.square(audio, dtype=np.float64), axis=1)
    level = np.sqrt(np.maximum(_one_pole(power, seconds, sr), 1e-12))
    over = 20 * np.log10(level) - threshold_db
    gain_db = -np.maximum(over, 0.0) * (1.0 - 1.0 / ratio)
    gain = 10 ** (_one_pole(gain_db, GAIN_SMOOTH, sr) / 20.0)
    return (audio * gain[:, None]).astype(np.float32)


def saturate(audio, drive, mix):
    shaped = np.tanh(drive * audio) / drive
    return ((1.0 - mix) * audio + mix * shaped).astype(np.float32)


def _delayed(values, samples):
    if samples <= 0:
        return values
    out = np.zeros_like(values)
    if samples < len(values):
        out[samples:] = values[:-samples]
    return out


def room(audio, sr, mix):
    if mix <= 0:
        return audio
    wet = np.zeros_like(audio)
    for seconds, gain, side in ROOM_TAPS:
        source = audio[:, 1 - side]
        wet[:, side] += gain * _delayed(source, int(seconds * sr))
    return audio + mix * lowpass(wet, sr, ROOM_TONE_HZ)


def double_track(audio, sr, mix):
    left = _delayed(audio[:, 1], int(DOUBLE_DELAYS[0] * sr))
    right = _delayed(audio[:, 0], int(DOUBLE_DELAYS[1] * sr))
    wide = audio + mix * np.stack([left, right], axis=1)
    return (wide / (1.0 + mix * 0.5)).astype(np.float32)


def mono_below(audio, sr, hz):
    low = lowpass(lowpass(audio, sr, hz), sr, hz)
    high = highpass(highpass(audio, sr, hz), sr, hz)
    centre = low.mean(axis=1, keepdims=True)
    return (high + centre).astype(np.float32)


def echo(audio, sr, bpm, mix):
    step = int(60.0 / max(bpm, 1.0) * ECHO_BEATS * sr)
    if step <= 0 or step >= len(audio):
        return audio
    source = lowpass(highpass(audio, sr, ECHO_TONE_HZ[0]), sr, ECHO_TONE_HZ[1])
    source = source.mean(axis=1)
    wet = np.zeros_like(audio)
    gain = 1.0
    for repeat in range(4):
        wet[:, repeat % 2] += gain * _delayed(source, step * (repeat + 1))
        gain *= ECHO_FEEDBACK
    return (audio + mix * wet).astype(np.float32)


def duck(audio, sr, kicks, depth_db):
    if depth_db <= 0 or not len(kicks):
        return audio
    hits = np.zeros(len(audio))
    for at in kicks:
        index = int(at * sr)
        if 0 <= index < len(hits):
            hits[index] = 1.0
    pole = float(np.exp(-1.0 / (sr * DUCK_RELEASE)))
    shape = np.minimum(signal.lfilter([1.0], [1.0, -pole], hits), 1.0)
    gain = 10 ** (-depth_db * shape / 20.0)
    return (audio * gain[:, None]).astype(np.float32)


def punch(audio, sr):
    threshold, ratio, seconds, mix = PUNCH
    squashed = compress(audio, sr, threshold, ratio, seconds)
    before = float(np.sqrt(np.mean(np.square(audio)))) + 1e-12
    after = float(np.sqrt(np.mean(np.square(squashed)))) + 1e-12
    return (audio + mix * squashed * (before / after)) / (1.0 + mix)


def tape(audio, sr):
    drive, mix = TAPE_DRIVE
    audio = saturate(audio, drive, mix)
    depth, rate = TAPE_WOW
    times = np.arange(len(audio)) / sr
    wobble = times - depth * (1.0 + np.sin(2 * np.pi * rate * times))
    wobbled = np.stack(
        [np.interp(wobble, times, audio[:, side]) for side in (0, 1)], axis=1
    )
    return lowpass(wobbled.astype(np.float32), sr, TAPE_TONE_HZ, order=1)


def kick_times(instruments):
    return sorted(
        note.start
        for inst in instruments
        if inst.is_drum
        for note in inst.notes
        if note.pitch in KICKS
    )


def polish_bus(bus, audio, sr, bpm=120.0, kicks=(), centred=True):
    audio = _stereo(audio)
    if not len(audio):
        return audio
    if bus in HIGHPASS_HZ:
        audio = highpass(audio, sr, HIGHPASS_HZ[bus])
    if bus == DRUMS:
        audio = punch(audio, sr)
    elif bus == BASS:
        audio = saturate(audio, *BASS_DRIVE)
        if centred:
            audio = mono_below(audio, sr, MONO_BELOW_HZ)
    elif bus == LEAD:
        audio = peaking(audio, sr, *PRESENCE)
    if centred:
        if bus in DOUBLE_MIX:
            audio = double_track(audio, sr, DOUBLE_MIX[bus])
        audio = room(audio, sr, ROOM_MIX.get(bus, 0.0))
        if bus == LEAD:
            audio = echo(audio, sr, bpm, ECHO_MIX)
    return duck(audio, sr, kicks, DUCK_DB.get(bus, 0.0)).astype(np.float32)


def polish_mix(audio, sr):
    audio = _stereo(audio)
    if not len(audio):
        return audio
    audio = compress(audio, sr, *GLUE)
    return tape(audio, sr).astype(np.float32)
