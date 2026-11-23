import librosa
import numpy as np
import pytest

from korshmer.audio_processing.vocal_cleanup import clean_vocal

RATE = 44100
VOWELS = (
    (700.0, 1200.0, 2600.0),
    (400.0, 2000.0, 2700.0),
    (300.0, 800.0, 2400.0),
)


def _sung_word(seconds, f0, formants, rng):
    time = np.arange(int(seconds * RATE)) / RATE
    vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 5.5 * time + rng.uniform(0, 6))
    phase = 2 * np.pi * np.cumsum(f0 * vibrato) / RATE
    word = np.zeros(time.shape[0])
    for harmonic in range(1, 40):
        freq = harmonic * f0
        if freq > RATE / 2 - 500:
            break
        weight = 0.02 + sum(
            np.exp(-(((freq - centre) / (90 + 0.06 * centre)) ** 2))
            for centre in formants
        )
        word += weight / harmonic**0.7 * np.sin(harmonic * phase)
    word /= np.sqrt(np.mean(word**2))
    edge = int(0.03 * RATE)
    word[:edge] *= np.linspace(0.0, 1.0, edge)
    word[-edge:] *= np.linspace(1.0, 0.0, edge)
    return word


def _steady_phone_take(seed, seconds=16.0, snr_db=30.0):
    rng = np.random.default_rng(seed)
    voice = np.zeros(int(seconds * RATE))
    level = 10 ** (-18.0 / 20.0)
    at = 0.3
    while at < seconds - 1.2:
        length = rng.uniform(0.3, 0.8)
        word = _sung_word(
            length, rng.uniform(180, 330), VOWELS[rng.integers(3)], rng
        )
        first = int(at * RATE)
        voice[first : first + word.shape[0]] += level * word
        at += length + rng.uniform(0.05, 0.12)
        if rng.random() < 0.2:
            at += rng.uniform(0.6, 1.0)
    hiss = rng.normal(0.0, 1.0, voice.shape[0])
    noise = hiss * 10 ** ((-18.0 - snr_db) / 20.0)
    return voice.astype(np.float32), (voice + noise).astype(np.float32)


def _frame_db(audio):
    hop = int(0.02 * RATE)
    levels = librosa.feature.rms(y=audio, frame_length=2 * hop, hop_length=hop)
    return 20.0 * np.log10(np.maximum(levels[0], 1e-9))


@pytest.mark.parametrize("seed", [1, 2])
def test_tone_balance_does_not_make_a_steady_vocal_pump(seed):
    voice, take = _steady_phone_take(seed)

    cleaned = clean_vocal(take, RATE)

    before, after = _frame_db(voice), _frame_db(cleaned)
    count = min(before.shape[0], after.shape[0])
    before, after = before[:count], after[:count]
    sung = before > np.percentile(before, 97) - 8.0
    change = after[sung] - before[sung]
    assert np.percentile(change, 95) - np.percentile(change, 5) <= 4.5


def test_tone_balance_keeps_the_level_of_each_phrase():
    voice, take = _steady_phone_take(3)

    cleaned = clean_vocal(take, RATE)

    window = 2 * RATE
    changes = []
    for first in range(0, voice.shape[0] - window, window):
        heard = voice[first : first + window]
        if np.sqrt(np.mean(heard**2)) < 10 ** (-30.0 / 20.0):
            continue
        out = cleaned[first : first + window]
        changes.append(10.0 * np.log10(np.mean(out**2) / np.mean(heard**2)))
    assert max(changes) - min(changes) <= 2.0
