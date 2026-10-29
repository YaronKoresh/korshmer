import numpy as np
from scipy import signal

from korshmer.audio_processing import vocal_cleanup
from korshmer.audio_processing.vocal_cleanup import clean_vocal

SR = 22050
BLEED_DB = -55.0
PHRASES = (("loud", 0.0), ("medium", -20.0), ("soft", -34.0), ("loud", 0.0))


def _db(audio):
    return 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)


def _voice(seconds, level_db):
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 280 * k * t) / k for k in range(1, 6))
    envelope = np.minimum(1.0, np.minimum(t / 0.03, (seconds - t) / 0.08))
    return 0.2 * 10 ** (level_db / 20) * tone * envelope / 0.855


def _bleed(seconds, rng):
    return rng.standard_normal(int(seconds * SR)) * 10 ** (BLEED_DB / 20)


def _song(seed=0):
    rng = np.random.default_rng(seed)
    parts, spans, position = [], {}, 0
    for index, (_name, level) in enumerate(PHRASES):
        for kind, seconds in (("phrase", 1.2), ("pause", 0.7)):
            if kind == "phrase":
                part = _voice(seconds, level) + _bleed(seconds, rng)
            else:
                part = _bleed(seconds, rng)
            spans[(index, kind)] = slice(position, position + len(part))
            position += len(part)
            parts.append(part)
    return np.concatenate(parts).astype(np.float32), spans


def _inner(span):
    margin = int(0.15 * SR)
    return slice(span.start + margin, span.stop - margin)


def test_a_phrase_34_db_under_the_loudest_survives_the_gate():
    audio, spans = _song()
    cleaned = clean_vocal(audio, SR)
    for index, (name, _level) in enumerate(PHRASES):
        span = _inner(spans[(index, "phrase")])
        change = _db(cleaned[span]) - _db(audio[span])
        limit = 3.0 if name == "soft" else 1.5
        assert change > -limit, (name, change)


def test_pauses_with_bleed_still_drop_by_ten_db():
    audio, spans = _song()
    cleaned = clean_vocal(audio, SR)
    for index in range(len(PHRASES)):
        span = _inner(spans[(index, "pause")])
        assert _db(audio[span]) - _db(cleaned[span]) >= 10.0


def test_noise_inside_a_phrase_is_not_spectrally_subtracted():
    audio, spans = _song()
    cleaned = clean_vocal(audio, SR)
    span = _inner(spans[(0, "phrase")])

    def hiss_db(data):
        freqs, power = signal.welch(data[span], SR, nperseg=2048)
        band = (freqs >= 5000) & (freqs <= 9000)
        return 10 * np.log10(power[band].sum())

    assert abs(hiss_db(cleaned) - hiss_db(audio)) < 0.5


def test_both_channels_get_the_same_gain():
    audio, spans = _song()
    rng = np.random.default_rng(5)
    right = 0.5 * audio + _bleed(len(audio) / SR, rng)[: len(audio)] * 0.5
    stereo = np.stack([audio, right.astype(np.float32)], axis=1)
    cleaned = clean_vocal(stereo, SR)
    for key in ((0, "phrase"), (1, "pause"), (2, "phrase"), (3, "pause")):
        span = _inner(spans[key])
        change = [_db(cleaned[span, c]) - _db(stereo[span, c]) for c in (0, 1)]
        assert abs(change[0] - change[1]) < 0.5


def test_gate_range_is_separate_from_the_active_range():
    assert vocal_cleanup.ACTIVE_RANGE_DB == 30.0
    assert vocal_cleanup.GATE_RANGE_DB > vocal_cleanup.ACTIVE_RANGE_DB
