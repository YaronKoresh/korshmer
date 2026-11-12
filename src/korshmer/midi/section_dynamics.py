import numpy as np
from scipy.ndimage import uniform_filter1d

from korshmer.audio_processing.validation import validate_effect_number

SAMPLE_RATE = 22050
HOP_LENGTH = 512
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
SILENCE_RANGE_DB = 60.0
LOUD_PERCENTILE = 95


DB_PER_VELOCITY_DECADE = 40.0


def loudness_curve(audio, sample_rate, window_s=3.0):
    import librosa

    validate_effect_number(window_s, "Dynamics window", 0.1, 60)
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=int(np.argmin(y.shape)))
    if sample_rate != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(y) < HOP_LENGTH:
        return np.zeros(0), np.zeros(0)
    power = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0] ** 2
    size = max(1, int(round(window_s * FRAMES_PER_SECOND)))
    smoothed = uniform_filter1d(power, size=size, mode="nearest")
    level = 10 * np.log10(smoothed + 1e-12)
    times = np.arange(len(level)) / FRAMES_PER_SECOND
    return times, level


def loud_level(level):
    audible = level[level > level.max() - SILENCE_RANGE_DB]
    return float(np.percentile(audible, LOUD_PERCENTILE))


def song_loud_level(stems, sample_rate, window_s=3.0):
    loudest = None
    for audio in stems:
        _, level = loudness_curve(audio, sample_rate, window_s)
        if len(level) == 0:
            continue
        stem_loud = loud_level(level)
        if loudest is None or stem_loud > loudest:
            loudest = stem_loud
    return loudest


def song_loud_level_from_files(paths, window_s=3.0):
    import librosa

    return song_loud_level(
        (librosa.load(str(p), sr=SAMPLE_RATE, mono=True)[0] for p in paths),
        SAMPLE_RATE,
        window_s,
    )


def apply_section_dynamics(
    midi_data,
    audio,
    sample_rate,
    *,
    amount=1.0,
    max_drop_db=18.0,
    window_s=3.0,
    min_velocity=1,
    reference_db=None,
):
    validate_effect_number(amount, "Dynamics amount", 0, 2)
    validate_effect_number(max_drop_db, "Dynamics range", 0, 60)
    if reference_db is not None:
        validate_effect_number(reference_db, "Dynamics reference", -200, 200)
    notes = [
        n
        for inst in midi_data.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    if not notes or amount == 0:
        return 0
    times, level = loudness_curve(audio, sample_rate, window_s)
    if len(level) == 0:
        return 0
    loud = loud_level(level) if reference_db is None else float(reference_db)
    changed = 0
    for note in notes:
        middle = 0.5 * (note.start + min(note.end, note.start + 0.5))
        at = float(np.interp(middle, times, level))
        offset = np.clip(at - loud, -max_drop_db, 0.0) * amount
        velocity = note.velocity * 10 ** (offset / DB_PER_VELOCITY_DECADE)
        velocity = int(np.clip(round(velocity), min_velocity, 127))
        if velocity != note.velocity:
            note.velocity = velocity
            changed += 1
    return changed


def apply_section_dynamics_from_file(midi_data, path, **options):
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return apply_section_dynamics(midi_data, audio, SAMPLE_RATE, **options)
