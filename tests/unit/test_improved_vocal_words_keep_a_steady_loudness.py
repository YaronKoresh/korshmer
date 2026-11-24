import numpy as np

from korshmer.audio_processing import clip_level

RATE = 44100
SYLLABLE_S = 0.25


def _line_with_loud_and_soft_words(seconds=8.0, jump_db=12.0):
    time = np.arange(int(seconds * RATE)) / RATE
    voice = np.sin(2 * np.pi * 220.0 * time)
    voice += 0.5 * np.sin(2 * np.pi * 440.0 * time)
    voice += 0.25 * np.sin(2 * np.pi * 880.0 * time)
    voice += 0.12 * np.sin(2 * np.pi * 1760.0 * time)
    syllable = (time // SYLLABLE_S).astype(int)
    loudness = np.where(syllable % 2 == 0, 0.0, -jump_db)
    voice *= 10.0 ** (loudness / 20.0)
    rng = np.random.default_rng(5)
    voice += rng.normal(0.0, 10.0 ** (-80.0 / 20.0), voice.shape[0])
    return (0.1 * voice / np.abs(voice).max()).astype(np.float32)


def _syllable_levels_db(audio):
    size = int(SYLLABLE_S * RATE)
    edge = int(0.05 * RATE)
    levels = []
    for start in range(size, audio.shape[0] - size, size):
        middle = audio[start + edge : start + size - edge]
        levels.append(10.0 * np.log10(np.mean(middle.astype(np.float64) ** 2)))
    return np.array(levels)


def _word_to_word_jump_db(audio):
    return float(np.median(np.abs(np.diff(_syllable_levels_db(audio)))))


def test_loud_and_soft_words_are_brought_closer_together():
    recording = _line_with_loud_and_soft_words()

    improved, _readings = clip_level.improve(recording, RATE)

    assert _word_to_word_jump_db(recording) > 11.0
    assert _word_to_word_jump_db(improved) <= 9.5


def test_word_loudness_steadying_is_reported():
    _improved, readings = clip_level.improve(
        _line_with_loud_and_soft_words(), RATE
    )

    assert "word spread" in clip_level.summary(readings)
