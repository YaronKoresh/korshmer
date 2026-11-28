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

TIMING_SPREAD = 0.009
TIMING_LIMIT = 0.022
LATE = {LEAD: 0.006, PADS: 0.004, BAND: 0.002}
VELOCITY_SPREAD = 7
DRUM_VELOCITY_SPREAD = 5
SHORTEST_NOTE = 0.02
ACCENTS = ((0.0, 9), (0.5, 3), (0.25, -6))
BEAT_ACCENT = 4

SWELL_PROGRAMS = frozenset(range(40, 96))
VIBRATO_PROGRAMS = frozenset(
    {40, 41, 42, 43, 52, 53, 54, *range(56, 80), 80, 81, 82, 83, 85, 86}
)
SWELL_FROM = 78
SWELL_TIME = 0.22
SWELL_STEPS = 6
VIBRATO_DELAY = 0.28
VIBRATO_RISE = 0.45
VIBRATO_DEPTH = 54
VIBRATO_STEPS = 5
VIBRATO_SHORTEST = 0.5
CHORD_GAP = 0.03
ENSEMBLE_PROGRAMS = frozenset({*range(40, 72), *range(88, 96), 80, 81, 85})
ENSEMBLE_CENTS = 6.0
ENSEMBLE_LAG = 0.012
ENSEMBLE_SPREAD = 22
ENSEMBLE_VOLUME = 92
BEND_SPAN_CENTS = 200.0

ROOM_TAPS = (
    (0.007, 0.32, 0),
    (0.011, 0.27, 1),
    (0.017, 0.22, 0),
    (0.023, 0.18, 1),
    (0.031, 0.13, 0),
    (0.041, 0.09, 1),
)
ROOM_TONE_HZ = 6000.0
ROOM_MIX = {DRUMS: 0.2, BASS: 0.0, PADS: 0.22, BAND: 0.24, LEAD: 0.18}
DOUBLE_DELAYS = (0.011, 0.017)
DOUBLE_MIX = {PADS: 0.3, BAND: 0.42}
HALL = {
    DRUMS: (1.0, 0.12),
    PADS: (2.6, 0.3),
    BAND: (1.2, 0.12),
    LEAD: (1.8, 0.18),
}
HALL_PREDELAY = 0.022
HALL_DAMP_HZ = 3500.0
HALL_LOW_HZ = 180.0
ECHO_BEATS = 0.75
ECHO_FEEDBACK = 0.35
ECHO_TONE_HZ = (300.0, 5000.0)
ECHO_MIX = 0.1
HIGHPASS_HZ = {BASS: 30.0, PADS: 120.0, BAND: 80.0, LEAD: 90.0}
MONO_BELOW_HZ = 120.0
PRESENCE = (3000.0, 2.0, 0.9)
BASS_DRIVE = (3.0, 0.5)
DRUM_DRIVE = (2.2, 0.35)
PUNCH = (-30.0, 6.0, 0.004, 0.7)
DUCK_DB = {BASS: 4.0, PADS: 3.0}
DUCK_RELEASE = 0.18
GLUE = (-18.0, 2.2, 0.06)
TAPE_DRIVE = (1.8, 0.5)
TAPE_TONE_HZ = 14000.0
TAPE_WOW = (0.00012, 0.6)
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


def accent(start, beat):
    if beat <= 0:
        return 0
    place = (start / beat) % 1.0
    for at, amount in ACCENTS:
        if min(abs(place - at), 1.0 - abs(place - at)) < 0.06:
            downbeat = (start / beat) % 4.0 < 0.06 and at == 0.0
            return amount + (BEAT_ACCENT if downbeat else 0)
    return ACCENTS[-1][1]


def humanized_notes(instrument, seed, keep_time=False, beat=0.0):
    rng = random.Random(f"humanize:{seed}")
    spread = DRUM_VELOCITY_SPREAD if instrument.is_drum else VELOCITY_SPREAD
    keep_time = keep_time or instrument.is_drum
    late = 0.0 if keep_time else LATE.get(bus_of(instrument), 0.0)
    played = []
    for note in instrument.notes:
        start, end = note.start, note.end
        if not keep_time:
            shift = late + max(
                -TIMING_LIMIT, min(TIMING_LIMIT, rng.gauss(0.0, TIMING_SPREAD))
            )
            start = max(0.0, start + shift)
            end = max(start + SHORTEST_NOTE, end + shift)
        velocity = (
            note.velocity
            + accent(note.start, beat)
            + rng.randint(-spread, spread)
        )
        played.append((start, end, note.pitch, max(1, min(127, velocity))))
    return played


def _onsets(played):
    starts = []
    for start, end, _pitch, _velocity in sorted(played):
        if starts and start - starts[-1][0] < CHORD_GAP:
            starts[-1][1] = max(starts[-1][1], end)
            continue
        starts.append([start, end])
    return starts


def expression_curves(program, played):
    swell, vibrato = [], []
    onsets = _onsets(played)
    if program in SWELL_PROGRAMS:
        for start, _end in onsets:
            for step in range(SWELL_STEPS + 1):
                value = SWELL_FROM + (127 - SWELL_FROM) * step / SWELL_STEPS
                at = start + SWELL_TIME * step / SWELL_STEPS
                swell.append((at, int(value)))
    if program in VIBRATO_PROGRAMS:
        for start, end in onsets:
            vibrato.append((start, 0))
            if end - start < VIBRATO_SHORTEST:
                continue
            rise = min(VIBRATO_RISE, end - start - VIBRATO_DELAY)
            for step in range(1, VIBRATO_STEPS + 1):
                at = start + VIBRATO_DELAY + rise * step / VIBRATO_STEPS
                vibrato.append((at, int(VIBRATO_DEPTH * step / VIBRATO_STEPS)))
    return swell, vibrato


def ensemble_bend(cents):
    return int(round(cents / BEND_SPAN_CENTS * 8192))


def wants_ensemble(instrument):
    return (
        not instrument.is_drum
        and instrument.program in ENSEMBLE_PROGRAMS
        and not instrument.pitch_bends
    )


def ensemble_twin(played):
    return [
        (start + ENSEMBLE_LAG, end + ENSEMBLE_LAG, pitch, max(1, velocity - 6))
        for start, end, pitch, velocity in played
    ]


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


def hall_response(sr, seconds, seed=7):
    rng = np.random.default_rng(seed)
    size = max(1, int(sr * seconds))
    times = np.arange(size) / sr
    noise = rng.standard_normal((size, 2)).astype(np.float32)
    low = lowpass(noise, sr, HALL_DAMP_HZ)
    high = noise - low
    decay = np.exp(-6.9 * times / seconds)[:, None]
    bright = np.exp(-6.9 * times / (seconds * 0.45))[:, None]
    response = highpass(
        (low * decay + 0.6 * high * bright).astype(np.float32), sr, HALL_LOW_HZ
    )
    gap = np.zeros((int(sr * HALL_PREDELAY), 2), np.float32)
    response = np.concatenate([gap, response])
    return (response / np.sqrt(np.sum(response**2) / 2.0)).astype(np.float32)


def hall(audio, sr, bus, centred=True):
    seconds, mix = HALL.get(bus, (0.0, 0.0))
    if mix <= 0 or not len(audio):
        return audio
    response = hall_response(sr, seconds, seed=len(bus))
    if centred:
        source = audio.mean(axis=1)
        wet = np.stack(
            [
                signal.oaconvolve(source, response[:, side])[: len(audio)]
                for side in (0, 1)
            ],
            axis=1,
        )
    else:
        wet = signal.oaconvolve(audio, response, axes=0)[: len(audio)]
    return (audio + mix * wet).astype(np.float32)


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
        audio = saturate(audio, *DRUM_DRIVE)
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
    audio = hall(audio, sr, bus, centred)
    return duck(audio, sr, kicks, DUCK_DB.get(bus, 0.0)).astype(np.float32)


def polish_mix(audio, sr):
    audio = _stereo(audio)
    if not len(audio):
        return audio
    audio = compress(audio, sr, *GLUE)
    return tape(audio, sr).astype(np.float32)
