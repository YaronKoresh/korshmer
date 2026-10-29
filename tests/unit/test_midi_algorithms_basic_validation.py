import io

import pretty_midi
import pytest

from korshmer import midi_algorithms as algo

NAN = float("nan")
INF = float("inf")


def make_midi():
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(100, 60, 0.0, 0.5))
    inst.notes.append(pretty_midi.Note(90, 64, 0.5, 1.0))
    midi.instruments.append(inst)
    return midi


def snapshot(midi):
    return [
        (n.velocity, n.pitch, n.start, n.end)
        for inst in midi.instruments
        for n in inst.notes
    ]


def assert_rejected_and_untouched(fn, *args, **kwargs):
    midi = make_midi()
    before = snapshot(midi)
    with pytest.raises(ValueError):
        fn(midi, *args, **kwargs)
    assert snapshot(midi) == before


@pytest.mark.parametrize(
    "feedback", [1.5, 2.0, -0.1, NAN, INF, "0.5", None, True]
)
def test_delay_feedback_must_be_between_zero_and_one(feedback):
    assert_rejected_and_untouched(algo.apply_midi_delay, 0.25, feedback, 3)


@pytest.mark.parametrize("delay", [-1.0, NAN, INF, "0.25", None, True, 61])
def test_delay_time_must_be_a_finite_number_of_seconds(delay):
    assert_rejected_and_untouched(algo.apply_midi_delay, delay, 0.5, 3)


@pytest.mark.parametrize("repeats", [2.5, -1, "3", None, True, NAN, 65])
def test_delay_repeats_must_be_a_whole_number(repeats):
    assert_rejected_and_untouched(algo.apply_midi_delay, 0.25, 0.5, repeats)


def test_valid_delay_makes_a_savable_midi():
    midi = algo.apply_midi_delay(make_midi(), 0.25, 1.0, 3)
    for inst in midi.instruments:
        for note in inst.notes:
            assert 1 <= note.velocity <= 127
    midi.write(io.BytesIO())


@pytest.mark.parametrize("amount", [-0.1, NAN, INF, "0.05", None, True])
def test_humanize_timing_must_be_a_finite_number_of_zero_or_more(amount):
    assert_rejected_and_untouched(algo.apply_humanize, amount, 10)


@pytest.mark.parametrize("amount", [-1, NAN, INF, "10", None, True])
def test_humanize_velocity_must_be_a_finite_number_of_zero_or_more(amount):
    assert_rejected_and_untouched(algo.apply_humanize, 0.05, amount)


@pytest.mark.parametrize("prob", [-0.1, 1.1, NAN, INF, "0.5", None, True])
def test_density_thin_probability_must_be_between_zero_and_one(prob):
    assert_rejected_and_untouched(algo.apply_density_thin, prob)


@pytest.mark.parametrize("mode", ["Bogus", "", None, 5, "fixed"])
def test_velocity_mode_must_be_known(mode):
    assert_rejected_and_untouched(algo.process_velocity, mode, 50)


@pytest.mark.parametrize("value", [NAN, INF, "x", None, True])
def test_velocity_value_must_be_a_finite_number(value):
    assert_rejected_and_untouched(algo.process_velocity, "Fixed", value)


def test_velocity_original_ignores_the_value():
    midi = make_midi()
    assert algo.process_velocity(midi, "Original", "anything") is midi


@pytest.mark.parametrize("mode", ["Bogus", "", None, 5, "legato"])
def test_legato_mode_must_be_known(mode):
    assert_rejected_and_untouched(algo.apply_legato_staccato, mode, 0.5)


@pytest.mark.parametrize(
    "percentage", [NAN, INF, -1, 0, 1.5, "0.5", None, True]
)
def test_staccato_percentage_must_be_above_zero_and_at_most_one(percentage):
    assert_rejected_and_untouched(
        algo.apply_legato_staccato, "Staccato", percentage
    )


@pytest.mark.parametrize("octave", ["x", 1.5, NAN, INF, 10**9, -3, True])
def test_octave_must_be_a_whole_number_in_range(octave):
    assert_rejected_and_untouched(algo.apply_octave_transpose, octave)


def test_octave_none_changes_nothing_and_valid_octave_moves_notes():
    midi = make_midi()
    assert algo.apply_octave_transpose(midi, None) is midi
    algo.apply_octave_transpose(midi, 5)
    assert [n.pitch for n in midi.instruments[0].notes] == [72, 76]
