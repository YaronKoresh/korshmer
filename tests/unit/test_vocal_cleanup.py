import numpy as np
import pytest

from korshmer.audio_processing.vocal_cleanup import clean_vocal

SR = 22050
SECONDS = 5
PHRASES = ((0.5, 1.8), (2.8, 4.2))


def _db(audio):
    return 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)


def _voice():
    t = np.arange(SR * SECONDS) / SR
    tone = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6)) * 0.15
    envelope = np.zeros(t.size)
    for start, end in PHRASES:
        envelope[int(start * SR) : int(end * SR)] = 1.0
    envelope = np.convolve(envelope, np.ones(400) / 400, "same")
    return (tone * envelope).astype(np.float32), t


def _bleed(t):
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(t.size) * 0.01
    return (noise + 0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def _window(start, end):
    return slice(int(start * SR), int(end * SR))


def test_pauses_get_quieter_and_phrases_keep_their_level():
    voice, t = _voice()
    noisy = voice + _bleed(t)
    cleaned = clean_vocal(noisy, SR)
    pause, phrase = _window(2.0, 2.6), _window(0.9, 1.6)
    assert _db(noisy[pause]) - _db(cleaned[pause]) >= 10.0
    assert abs(_db(noisy[phrase]) - _db(cleaned[phrase])) < 1.5


def test_pauses_are_lowered_not_silenced():
    voice, t = _voice()
    cleaned = clean_vocal(voice + _bleed(t), SR)
    assert _db(cleaned[_window(2.0, 2.6)]) > -90.0


def test_length_and_shape_are_kept():
    voice, t = _voice()
    stereo = np.stack([voice + _bleed(t), voice], axis=1)
    cleaned = clean_vocal(stereo, SR)
    assert cleaned.shape == stereo.shape
    assert cleaned.dtype == np.float32
    assert clean_vocal(voice, SR).shape == voice.shape


def test_continuous_singing_is_not_gated():
    t = np.arange(SR * 3) / SR
    sung = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sung *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t) ** 2).astype(np.float32)
    cleaned = clean_vocal(sung, SR)
    assert abs(_db(sung) - _db(cleaned)) < 1.5


def test_low_rumble_is_removed():
    t = np.arange(SR * 2) / SR
    rumble = (0.3 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    assert _db(clean_vocal(rumble, SR)) < _db(rumble) - 20.0


def test_empty_and_short_input():
    assert clean_vocal(np.zeros(0, dtype=np.float32), SR).size == 0
    short = np.full(100, 0.1, dtype=np.float32)
    assert clean_vocal(short, SR).shape == short.shape


def test_non_finite_samples_do_not_spread():
    voice, t = _voice()
    audio = voice + _bleed(t)
    audio[1000] = np.nan
    audio[2000] = np.inf
    cleaned = clean_vocal(audio, SR)
    assert np.all(np.isfinite(cleaned))


def test_output_never_clips():
    voice, t = _voice()
    loud = np.clip(voice * 30.0, -3.0, 3.0)
    assert float(np.max(np.abs(clean_vocal(loud, SR)))) <= 0.99 + 1e-6


def test_input_is_not_modified():
    voice, t = _voice()
    audio = voice + _bleed(t)
    before = audio.copy()
    clean_vocal(audio, SR)
    np.testing.assert_array_equal(audio, before)


@pytest.mark.parametrize("silence", [np.zeros(SR, dtype=np.float32)])
def test_silence_stays_silent(silence):
    assert float(np.max(np.abs(clean_vocal(silence, SR)))) == 0.0


def _sung(level_db, seconds=3):
    t = np.arange(SR * seconds) / SR
    tone = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    envelope = np.where((t % 1.5) < 1.0, 1.0, 0.0)
    envelope = np.convolve(envelope, np.ones(400) / 400, "same")
    audio = tone * envelope
    audio = audio / np.sqrt(np.mean(audio[envelope > 0.99] ** 2))
    return (audio * 10 ** (level_db / 20)).astype(np.float32)


def test_quiet_and_loud_voices_end_at_the_same_level():
    from korshmer.audio_processing.vocal_cleanup import (
        VOCAL_ABOVE_MIDI_DB,
        active_level_db,
        vocal_clip_gain,
    )

    target = -12.0 + VOCAL_ABOVE_MIDI_DB
    for level in (-20.0, -16.0, -12.0):
        voice = _sung(level)
        gain = vocal_clip_gain(voice, midi_level_db=-12.0)
        assert active_level_db(voice * gain) == pytest.approx(target, abs=0.6)


def test_gain_uses_the_midi_render_level_by_default():
    from korshmer.audio_processing.vocal_cleanup import vocal_clip_gain
    from korshmer.audio_processors import RENDER_TARGET_RMS_DB

    voice = _sung(-25.0)
    assert vocal_clip_gain(voice) == vocal_clip_gain(
        voice, midi_level_db=RENDER_TARGET_RMS_DB
    )


def test_gain_follows_the_requested_step_above_midi():
    from korshmer.audio_processing.vocal_cleanup import vocal_clip_gain

    voice = _sung(-15.0)
    ratio = vocal_clip_gain(
        voice, midi_level_db=-12.0, above_midi_db=6.0
    ) / vocal_clip_gain(voice, midi_level_db=-12.0, above_midi_db=0.0)
    assert ratio == pytest.approx(10 ** (6 / 20), rel=1e-3)


def test_gain_is_limited_to_the_clip_range():
    from korshmer.audio_processing.vocal_cleanup import (
        MAX_VOCAL_GAIN,
        MIN_VOCAL_GAIN,
        vocal_clip_gain,
    )
    from korshmer.ui.viewmodels.editor_vm import MAX_CLIP_GAIN

    assert MAX_VOCAL_GAIN <= MAX_CLIP_GAIN
    assert vocal_clip_gain(_sung(-70.0), midi_level_db=-12.0) == MAX_VOCAL_GAIN
    assert vocal_clip_gain(_sung(0.0), midi_level_db=-40.0) == MIN_VOCAL_GAIN


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros(SR * 2, dtype=np.float32),
        np.full(100, 0.5, dtype=np.float32),
        np.full(SR, np.nan, dtype=np.float32),
    ],
)
def test_unmeasurable_audio_keeps_gain_one(audio):
    from korshmer.audio_processing.vocal_cleanup import vocal_clip_gain

    assert vocal_clip_gain(audio, midi_level_db=-12.0) == 1.0


def test_stereo_voice_is_measured_like_mono():
    from korshmer.audio_processing.vocal_cleanup import vocal_clip_gain

    voice = _sung(-22.0)
    stereo = np.stack([voice, voice], axis=1)
    assert vocal_clip_gain(stereo, midi_level_db=-12.0) == pytest.approx(
        vocal_clip_gain(voice, midi_level_db=-12.0)
    )


@pytest.mark.parametrize("sr", [0, -1, float("nan"), float("inf"), "x", None])
def test_invalid_sample_rate_is_rejected(sr):
    audio = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    with pytest.raises(ValueError):
        clean_vocal(audio, sr)


def test_default_voice_level_sits_below_the_midi_render():
    from korshmer.audio_processing.vocal_cleanup import (
        VOCAL_ABOVE_MIDI_DB,
        active_level_db,
        vocal_clip_gain,
    )

    voice = _sung(-18.0)
    level = active_level_db(voice * vocal_clip_gain(voice, midi_level_db=-12.0))
    assert VOCAL_ABOVE_MIDI_DB < 0
    assert level < -12.0


@pytest.mark.parametrize("length", [1, 5, 15, 18, 19])
def test_clip_shorter_than_the_filter_padding_keeps_its_length(length):
    audio = (np.random.default_rng(0).standard_normal(length) * 0.1).astype(
        np.float32
    )
    cleaned = clean_vocal(audio, 44100)
    assert cleaned.shape == audio.shape
    assert np.all(np.isfinite(cleaned))
