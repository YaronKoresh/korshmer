import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing.stem_session import mix_render_with_clips

RATE = 44100
LEVEL = 0.25
CEILING = 10 ** (-1.0 / 20.0)


def tone_file(tmp_path, level=LEVEL):
    t = np.arange(RATE // 2) / RATE
    path = tmp_path / "take.wav"
    sf.write(str(path), level * np.sin(2 * np.pi * 220 * t), RATE, "FLOAT")
    return str(path)


def mixed(tmp_path, pan, level=LEVEL):
    clip = {"path": tone_file(tmp_path, level), "start": 0.0, "pan": pan}
    out = mix_render_with_clips(None, [clip], str(tmp_path / "mix.wav"))
    data, _rate = sf.read(out, dtype="float32")
    return data[:, 0], data[:, 1]


def power_db(left, right):
    return 10 * np.log10(np.mean(left**2) + np.mean(right**2))


def peak(side):
    return float(np.max(np.abs(side)))


def test_a_clip_panned_hard_left_leaves_the_right_side_silent(tmp_path):
    left, right = mixed(tmp_path, -1.0)
    assert peak(right) == 0.0
    assert peak(left) > LEVEL


def test_a_clip_panned_hard_right_leaves_the_left_side_silent(tmp_path):
    left, right = mixed(tmp_path, 1.0)
    assert peak(left) == 0.0
    assert peak(right) > LEVEL


def test_hard_right_is_the_mirror_of_hard_left(tmp_path):
    left_side, left_rest = mixed(tmp_path, -1.0)
    right_rest, right_side = mixed(tmp_path, 1.0)
    assert peak(left_rest) == 0.0 and peak(right_rest) == 0.0
    assert np.allclose(left_side, right_side, atol=1e-6)


def test_the_centre_keeps_its_level_while_a_side_gets_louder(tmp_path):
    centre_left, centre_right = mixed(tmp_path, 0.0)
    side_left, _silent = mixed(tmp_path, -1.0)
    assert np.array_equal(centre_left, centre_right)
    assert peak(centre_left) == pytest.approx(LEVEL, abs=1e-3)
    assert peak(side_left) == pytest.approx(LEVEL * np.sqrt(2), abs=1e-3)


@pytest.mark.parametrize(
    "pan", [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]
)
def test_a_clip_is_as_loud_anywhere_between_the_speakers(tmp_path, pan):
    centre = power_db(*mixed(tmp_path, 0.0))
    left, right = mixed(tmp_path, pan)
    louder, quieter = (right, left) if pan > 0 else (left, right)
    assert peak(louder) > peak(quieter)
    assert power_db(left, right) == pytest.approx(centre, abs=0.1)


def test_the_further_a_clip_is_panned_the_quieter_the_other_side(tmp_path):
    lefts = [peak(mixed(tmp_path, pan)[0]) for pan in (0.0, 0.25, 0.5, 0.75)]
    assert lefts == sorted(lefts, reverse=True)
    assert len(set(lefts)) == 4


def test_a_full_scale_clip_panned_to_one_side_does_not_clip_the_mix(tmp_path):
    left, right = mixed(tmp_path, 1.0, level=1.0)
    assert peak(left) == 0.0
    assert peak(right) == pytest.approx(CEILING, abs=1e-3)


@pytest.mark.parametrize("bad", ["left", float("nan")])
def test_a_clip_with_an_unreadable_pan_plays_in_the_centre(tmp_path, bad):
    centre_left, centre_right = mixed(tmp_path, bad)
    panned_left, panned_right = mixed(tmp_path, 0.5)
    assert np.array_equal(centre_left, centre_right)
    assert peak(panned_right) > peak(panned_left)
