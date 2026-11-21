import numpy as np
import pytest

from korshmer.audio_processing.vocal_cleanup import clean_vocal

RATE = 44100
BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
NATURAL = {
    125: -6.0,
    250: 0.0,
    500: 1.0,
    1000: 0.0,
    2000: -5.0,
    4000: -11.0,
    8000: -17.0,
}


def _interpolated(curve, freq):
    centres = np.log2(np.array(sorted(curve)))
    values = np.array([curve[c] for c in sorted(curve)])
    return np.interp(np.log2(max(freq, 1.0)), centres, values)


def _voice(seconds, shape=NATURAL, level_db=-20.0, f0=196.0, top_hz=12000.0):
    time = np.arange(int(seconds * RATE)) / RATE
    pitch = f0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.5 * time))
    phase = 2 * np.pi * np.cumsum(pitch) / RATE
    voice = np.zeros_like(time)
    for harmonic in range(1, int(top_hz / f0)):
        gain_db = _interpolated(shape, harmonic * f0)
        voice += 10 ** (gain_db / 20.0) * np.sin(harmonic * phase)
    voice *= 10 ** (level_db / 20.0) / np.sqrt(np.mean(voice**2))
    return voice


def _hiss(samples, level_db, low_hz=20.0, high_hz=20000.0, seed=1):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, samples))
    freqs = np.fft.rfftfreq(samples, 1 / RATE)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    noise = np.fft.irfft(spectrum, samples)
    return noise * 10 ** (level_db / 20.0) / np.sqrt(np.mean(noise**2))


def _band_db(audio, low_hz, high_hz):
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / RATE)
    chosen = (freqs >= low_hz) & (freqs < high_hz)
    return 10 * np.log10(np.mean(power[chosen]) + 1e-30)


def _octave_db(audio, centre):
    return _band_db(audio, centre / np.sqrt(2), centre * np.sqrt(2))


def _balance(audio):
    body = _octave_db(audio, 1000)
    return {centre: _octave_db(audio, centre) - body for centre in BANDS}


def _middle(audio, seconds=0.5):
    edge = int(seconds * RATE)
    return audio[edge:-edge]


def _frame_db(audio, frame=2048):
    count = audio.shape[0] // frame
    frames = audio[: count * frame].reshape(count, frame)
    return 10 * np.log10(np.mean(frames**2, axis=1) + 1e-30)


def test_a_dull_vocal_gets_its_presence_and_air_back():
    dull = dict(NATURAL)
    dull.update({2000: -14.0, 4000: -30.0, 8000: -45.0})
    voice = _voice(4.0, dull)

    before = _balance(_middle(voice))
    after = _balance(_middle(clean_vocal(voice, RATE)))

    assert after[4000] - before[4000] >= 4.0
    assert after[8000] - before[8000] >= 4.0
    assert after[4000] - before[4000] <= 9.5


def test_a_muddy_vocal_has_its_low_middle_turned_down():
    muddy = dict(NATURAL)
    muddy.update({125: 14.0, 250: 14.0, 500: 14.0})
    voice = _voice(4.0, muddy)

    before = _balance(_middle(voice))
    after = _balance(_middle(clean_vocal(voice, RATE)))

    assert before[250] - after[250] >= 3.0
    assert before[500] - after[500] >= 3.0


def test_a_well_balanced_vocal_keeps_its_tone():
    voice = _voice(4.0)

    before = _balance(_middle(voice))
    after = _balance(_middle(clean_vocal(voice, RATE)))

    for centre in (250, 500, 2000, 4000):
        assert after[centre] == pytest.approx(before[centre], abs=1.5)


def test_hiss_under_singing_that_never_pauses_is_turned_down():
    voice = _voice(4.0, top_hz=5000.0)
    hiss = _hiss(voice.shape[0], -50.0, 10000.0, 16000.0)

    cleaned = clean_vocal(voice + hiss, RATE)

    removed = _band_db(_middle(voice + hiss), 10000, 16000) - _band_db(
        _middle(cleaned), 10000, 16000
    )
    assert removed >= 8.0


def test_a_long_held_note_is_not_mistaken_for_noise():
    voice = _voice(6.0, top_hz=5000.0)

    cleaned = clean_vocal(voice + _hiss(voice.shape[0], -70.0), RATE)

    assert _band_db(_middle(cleaned), 150, 3000) == pytest.approx(
        _band_db(_middle(voice), 150, 3000), abs=1.0
    )


def test_a_fading_note_fades_smoothly_without_sudden_drops():
    seconds = 4.0
    voice = _voice(seconds, top_hz=5000.0)
    fade_db = np.linspace(0.0, -42.0, voice.shape[0])
    fading = voice * 10 ** (fade_db / 20.0)
    held = np.concatenate([_voice(1.0, top_hz=5000.0), fading])
    noisy = held + _hiss(held.shape[0], -80.0)

    cleaned = clean_vocal(noisy, RATE)

    start = int(1.2 * RATE)
    stop = start + int(2.6 * RATE)
    steps_in = np.diff(_frame_db(noisy[start:stop]))
    steps_out = np.diff(_frame_db(cleaned[start:stop]))
    assert np.max(np.abs(steps_out - steps_in)) <= 1.0


def test_soft_s_sounds_between_words_keep_their_brightness():
    vowel = _voice(0.4, top_hz=5000.0)
    hiss_s = _hiss(int(0.09 * RATE), -38.0, 4500.0, 9000.0, seed=3)
    rest = np.zeros(int(0.25 * RATE))
    pieces, s_spans, at = [], [], 0
    for _ in range(6):
        for piece, is_s in ((vowel, False), (hiss_s, True), (rest, False)):
            if is_s:
                s_spans.append((at, at + piece.shape[0]))
            pieces.append(piece)
            at += piece.shape[0]
    phrase = np.concatenate(pieces) + _hiss(at, -75.0)

    cleaned = clean_vocal(phrase, RATE)

    for start, stop in s_spans:
        kept = _band_db(cleaned[start:stop], 4500, 9000)
        assert kept == pytest.approx(
            _band_db(phrase[start:stop], 4500, 9000), abs=2.0
        )


def test_rumble_under_the_voice_is_still_removed():
    voice = _voice(3.0)
    time = np.arange(voice.shape[0]) / RATE
    rumble = 0.05 * np.sin(2 * np.pi * 40 * time)

    cleaned = clean_vocal(voice + rumble, RATE)

    removed = _band_db(_middle(voice + rumble), 35, 45) - _band_db(
        _middle(cleaned), 35, 45
    )
    assert removed >= 12.0


def test_the_voice_keeps_its_loudness():
    voice = _voice(4.0)

    cleaned = clean_vocal(voice, RATE)

    rms_in = 20 * np.log10(np.sqrt(np.mean(_middle(voice) ** 2)))
    rms_out = 20 * np.log10(np.sqrt(np.mean(_middle(cleaned) ** 2)))
    assert rms_out == pytest.approx(rms_in, abs=1.0)


def test_cleaning_a_cleaned_vocal_again_changes_little():
    dull = dict(NATURAL)
    dull.update({4000: -24.0, 8000: -36.0})
    voice = _voice(4.0, dull) + _hiss(int(4.0 * RATE), -60.0)

    once = clean_vocal(voice, RATE)
    twice = clean_vocal(once, RATE)

    first, second = _balance(_middle(once)), _balance(_middle(twice))
    for centre in (250, 500, 2000, 4000):
        assert second[centre] == pytest.approx(first[centre], abs=1.5)


def test_a_stereo_vocal_keeps_both_channels_and_its_width():
    left = _voice(3.0)
    right = 0.5 * _voice(3.0, f0=220.0)
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    cleaned = clean_vocal(stereo, RATE)

    assert cleaned.shape == stereo.shape
    ratio_in = np.std(stereo[:, 1]) / np.std(stereo[:, 0])
    ratio_out = np.std(cleaned[:, 1]) / np.std(cleaned[:, 0])
    assert ratio_out == pytest.approx(ratio_in, rel=0.15)


@pytest.mark.parametrize("bleed_db", [-80.0, -60.0])
def test_a_phrase_that_swells_and_fades_never_jumps_in_level(bleed_db):
    voice = _voice(5.0, top_hz=5000.0)
    time = np.arange(voice.shape[0]) / RATE
    swell_db = -12.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * time))
    fade_db = np.linspace(0.0, -20.0, voice.shape[0])
    sung = voice * 10 ** ((swell_db + fade_db) / 20.0)
    noisy = sung + _hiss(voice.shape[0], bleed_db, 100.0, 8000.0)

    cleaned = clean_vocal(noisy, RATE)

    levels_in, levels_out = _frame_db(noisy), _frame_db(cleaned)
    heard = levels_in > levels_in.max() - 50.0
    both = heard[1:] & heard[:-1]
    extra = np.abs(np.diff(levels_out) - np.diff(levels_in))[both]
    assert np.max(extra) <= 1.5


def _breathy_phrase(breath_db=-38.0, seed=5):
    vowel = _voice(0.8, top_hz=5000.0)
    breath = _hiss(int(0.35 * RATE), breath_db, 400.0, 3500.0, seed=seed)
    ramp = np.minimum(
        1.0,
        np.minimum(np.arange(breath.size), np.arange(breath.size)[::-1])
        / 600.0,
    )
    breath = breath * ramp
    pieces, breath_spans, vowel_spans, at = [], [], [], 0
    for _ in range(4):
        for piece, spans in ((vowel, vowel_spans), (breath, breath_spans)):
            spans.append((at, at + piece.shape[0]))
            pieces.append(piece)
            at += piece.shape[0]
    return np.concatenate(pieces) + _hiss(at, -80.0), breath_spans, vowel_spans


def _rms_db(audio):
    return 10 * np.log10(np.mean(audio**2) + 1e-30)


def test_breaths_between_phrases_are_turned_down():
    phrase, breaths, _vowels = _breathy_phrase()

    cleaned = clean_vocal(phrase, RATE)

    for start, stop in breaths:
        inner = slice(start + 2048, stop - 2048)
        assert _rms_db(phrase[inner]) - _rms_db(cleaned[inner]) >= 8.0


def test_turning_breaths_down_leaves_the_sung_vowels_alone():
    phrase, _breaths, vowels = _breathy_phrase()

    cleaned = clean_vocal(phrase, RATE)

    for start, stop in vowels:
        inner = slice(start + 4096, stop - 4096)
        assert _rms_db(cleaned[inner]) == pytest.approx(
            _rms_db(phrase[inner]), abs=1.0
        )


def test_a_short_h_sound_inside_a_word_is_not_taken_for_a_breath():
    vowel = _voice(0.5, top_hz=5000.0)
    h_sound = _hiss(int(0.07 * RATE), -34.0, 400.0, 3500.0, seed=8)
    word = np.concatenate([vowel, h_sound, vowel, h_sound, vowel])
    word = word + _hiss(word.shape[0], -80.0)
    start = vowel.shape[0]

    cleaned = clean_vocal(word, RATE)

    span = slice(start, start + h_sound.shape[0])
    assert _rms_db(cleaned[span]) == pytest.approx(_rms_db(word[span]), abs=3.0)


def _popped_phrase():
    vowel = _voice(1.5, f0=130.0, top_hz=5000.0)
    time = np.arange(int(0.08 * RATE)) / RATE
    pop = 0.5 * np.sin(2 * np.pi * 110 * time) * np.exp(-time / 0.02)
    start = int(0.7 * RATE)
    phrase = np.concatenate([_voice(0.7, f0=130.0, top_hz=5000.0), vowel])
    phrase = phrase + _hiss(phrase.shape[0], -80.0)
    phrase[start : start + pop.shape[0]] += pop
    return phrase, slice(start, start + pop.shape[0])


def test_a_pop_of_air_on_the_microphone_is_taken_out():
    phrase, pop = _popped_phrase()

    cleaned = clean_vocal(phrase, RATE)

    removed = _band_db(phrase[pop], 70, 250) - _band_db(cleaned[pop], 70, 250)
    assert removed >= 8.0


def test_the_low_notes_of_a_deep_voice_are_not_taken_for_pops():
    deep = _voice(4.0, f0=98.0, top_hz=5000.0)
    phrases = np.concatenate([deep, np.zeros(int(0.5 * RATE)), deep])
    phrases = phrases + _hiss(phrases.shape[0], -75.0)

    cleaned = clean_vocal(phrases, RATE)

    assert _band_db(_middle(cleaned), 70, 250) == pytest.approx(
        _band_db(_middle(phrases), 70, 250), abs=1.5
    )


def test_a_phrase_much_quieter_than_the_rest_is_brought_closer():
    loud = _voice(3.0, level_db=-14.0)
    soft = _voice(3.0, level_db=-28.0, f0=220.0)
    gap = np.zeros(int(0.6 * RATE))
    sung = np.concatenate([loud, gap, soft, gap, loud])
    sung = sung + _hiss(sung.shape[0], -80.0)
    loud_span = slice(int(0.5 * RATE), int(2.5 * RATE))
    soft_start = loud.shape[0] + gap.shape[0]
    soft_span = slice(
        soft_start + int(0.5 * RATE), soft_start + int(2.5 * RATE)
    )

    cleaned = clean_vocal(sung, RATE)

    before = _rms_db(sung[loud_span]) - _rms_db(sung[soft_span])
    after = _rms_db(cleaned[loud_span]) - _rms_db(cleaned[soft_span])
    assert before > 13.0
    assert after <= 6.0
    assert _rms_db(cleaned[loud_span]) == pytest.approx(
        _rms_db(sung[loud_span]), abs=1.0
    )


def test_a_duller_phrase_gets_the_same_brightness_as_the_rest():
    dull = dict(NATURAL)
    dull.update({2000: -10.0, 4000: -24.0, 8000: -36.0})
    bright = _voice(4.0)
    muffled = _voice(4.0, dull)
    sung = np.concatenate([bright, muffled, bright])

    cleaned = clean_vocal(sung, RATE)

    clear = cleaned[int(0.5 * RATE) : int(3.5 * RATE)]
    dim = cleaned[int(4.5 * RATE) : int(7.5 * RATE)]
    assert _balance(dim)[4000] == pytest.approx(_balance(clear)[4000], abs=4.0)
