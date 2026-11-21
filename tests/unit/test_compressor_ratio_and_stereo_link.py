import numpy as np
import pytest

from korshmer.audio_processing.dynamics import Compressor

RATE = 44100
THRESHOLD_DB = -30.0


def _steady(level_db, seconds=1.0):
    return np.full(int(RATE * seconds), 10 ** (level_db / 20.0))


def _settled_db(audio):
    return 20 * np.log10(abs(audio[-1]))


def _compress(audio, **settings):
    options = {
        "threshold_db": THRESHOLD_DB,
        "ratio": 4.0,
        "attack_ms": 1,
        "release_ms": 10,
    } | settings
    return Compressor(RATE).process(audio, **options)


@pytest.mark.parametrize("over_db", [4.0, 8.0, 16.0, 24.0])
def test_a_four_to_one_ratio_lets_one_decibel_out_for_every_four_in(over_db):
    out = _compress(_steady(THRESHOLD_DB + over_db))

    assert _settled_db(out) == pytest.approx(
        THRESHOLD_DB + over_db / 4.0, abs=0.05
    )


def test_a_two_to_one_ratio_halves_the_decibels_over_the_threshold():
    out = _compress(_steady(THRESHOLD_DB + 12.0), ratio=2.0)

    assert _settled_db(out) == pytest.approx(THRESHOLD_DB + 6.0, abs=0.05)


def test_makeup_gain_is_added_after_the_gain_reduction():
    out = _compress(_steady(THRESHOLD_DB + 8.0), makeup_db=5.0)

    assert _settled_db(out) == pytest.approx(THRESHOLD_DB + 7.0, abs=0.05)


def test_a_loud_left_channel_turns_both_channels_down_by_the_same_amount():
    left = _steady(THRESHOLD_DB + 16.0)
    right = _steady(THRESHOLD_DB - 10.0)

    out = _compress(np.stack([left, right]))

    assert _settled_db(out[0]) == pytest.approx(THRESHOLD_DB + 4.0, abs=0.05)
    assert np.allclose(out[0] / left, out[1] / right)


def test_a_centred_sound_stays_centred_when_one_side_has_an_extra_hit():
    rng = np.random.default_rng(3)
    centre = rng.uniform(0.02, 0.1, RATE)
    hit = np.zeros(RATE)
    hit[RATE // 2 : RATE // 2 + 2000] = 0.8
    left = centre + hit

    out = _compress(np.stack([left, centre]))

    assert np.allclose(out[0] / left, out[1] / centre)


class TestBehaviourThatMustStayAsItWas:
    def test_sound_under_the_threshold_passes_through_untouched(self):
        quiet = _steady(THRESHOLD_DB - 6.0)

        assert np.allclose(_compress(quiet), quiet)

    def test_quiet_stereo_passes_through_untouched(self):
        quiet = np.stack([_steady(THRESHOLD_DB - 6.0)] * 2)

        assert np.allclose(_compress(quiet), quiet)
