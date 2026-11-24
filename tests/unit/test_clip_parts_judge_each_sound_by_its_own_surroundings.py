import numpy as np
import pytest

from korshmer.audio_processing import clip_polish

RATE = 44100
LOUD = 0.3


def _note(seconds, level, pitch=220.0):
    time = np.arange(int(RATE * seconds)) / RATE
    wobble = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time)
    return level * np.sin(2 * np.pi * pitch * time) * wobble


def _rest(seconds):
    return np.zeros(int(RATE * seconds))


def _with_hiss(audio, under_loud_db, seed=0):
    hiss = np.random.default_rng(seed).normal(0, 1, audio.shape)
    return (audio + LOUD * 10 ** (under_loud_db / 20.0) * hiss).astype(
        np.float32
    )


def _soft_phrases(level):
    return np.concatenate(
        [
            _rest(2.0),
            _note(4.0, level),
            _rest(4.0),
            _note(4.0, level),
            _rest(3.0),
        ]
    )


def _shifted(spans, seconds):
    return [(begin - seconds, end - seconds) for begin, end in spans]


@pytest.mark.parametrize("soft_under_loud_db", [20.0, 25.0, 30.0])
def test_soft_phrases_are_kept_alike_whether_or_not_the_clip_has_a_loud_part(
    soft_under_loud_db,
):
    soft = LOUD * 10 ** (-soft_under_loud_db / 20.0)
    alone = _with_hiss(_soft_phrases(soft), -60.0)
    lead_in = np.concatenate([_rest(1.0), _note(3.0, LOUD), _rest(4.0)])
    with_loud = _with_hiss(
        np.concatenate([lead_in, _soft_phrases(soft)]), -60.0
    )

    spans_alone = clip_polish.clip_parts(alone, RATE)
    spans_with_loud = clip_polish.clip_parts(with_loud, RATE)

    assert len(spans_alone) == 2
    assert len(spans_with_loud) == 3
    assert np.allclose(
        _shifted(spans_with_loud[1:], 8.0), spans_alone, atol=0.15
    )


def test_noise_between_soft_phrases_is_still_cut_when_the_clip_has_a_loud_part():
    soft = LOUD * 10 ** (-25.0 / 20.0)
    lead_in = np.concatenate([_rest(1.0), _note(3.0, LOUD), _rest(4.0)])
    clip = _with_hiss(np.concatenate([lead_in, _soft_phrases(soft)]), -45.0)

    spans = clip_polish.clip_parts(clip, RATE)

    gaps = [(0.95, 1.0), (4.05, 10.0), (14.05, 18.0), (22.05, 25.0)]
    for begin, end in gaps:
        assert not any(b < end - 0.2 and e > begin + 0.2 for b, e in spans)


def test_a_faint_sound_right_after_a_loud_one_still_counts_as_background():
    faint = LOUD * 10 ** (-28.0 / 20.0)
    clip = _with_hiss(
        np.concatenate(
            [_rest(1.0), _note(4.0, LOUD), _note(0.6, faint), _rest(3.0)]
        ),
        -60.0,
    )

    spans = clip_polish.clip_parts(clip, RATE)

    assert len(spans) == 1
    assert spans[0][1] < 5.3
