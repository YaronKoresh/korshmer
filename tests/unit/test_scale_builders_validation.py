import pytest

from korshmer.midi_processing.scale_analysis import (
    MAX_DIVISIONS,
    CustomScaleBuilder,
    MicrotonalScale,
    Scale,
    ScaleDegreeHighlighter,
    ScaleDetector,
    ScaleType,
)

NAN = float("nan")
INF = float("inf")


def _micro():
    scale = MicrotonalScale(60.0)
    scale.add_interval(0)
    scale.add_interval(12)
    return scale


@pytest.mark.parametrize("root", [NAN, INF, "C", None, True])
def test_microtonal_root_must_be_a_finite_number(root):
    with pytest.raises(ValueError, match="Root"):
        MicrotonalScale(root)


@pytest.mark.parametrize(
    "divisions", [0, -1, 2.5, "12", None, True, MAX_DIVISIONS + 1]
)
def test_microtonal_divisions_must_be_a_sensible_whole_number(divisions):
    with pytest.raises(ValueError, match="Divisions per octave"):
        MicrotonalScale(60.0, divisions)


@pytest.mark.parametrize("interval", [NAN, INF, "x", None, True])
def test_microtonal_interval_must_be_a_finite_number(interval):
    with pytest.raises(ValueError, match="Interval"):
        MicrotonalScale(60.0).add_interval(interval)


@pytest.mark.parametrize("index", [-1, 1.5, "0", None, True])
def test_frequency_rejects_invalid_note_index(index):
    with pytest.raises(ValueError, match="Note index"):
        _micro().get_frequency(index)


@pytest.mark.parametrize("base", [NAN, INF, 0, -440.0, "440", None])
def test_frequency_rejects_invalid_base_frequency(base):
    with pytest.raises(ValueError, match="Base frequency"):
        _micro().get_frequency(1, base)


def test_frequency_values():
    scale = _micro()
    assert scale.get_frequency(0) == pytest.approx(440.0)
    assert scale.get_frequency(1) == pytest.approx(880.0)
    assert scale.get_frequency(5) == 440.0


@pytest.mark.parametrize(
    "divisions", [0, -3, 2.5, "24", None, True, MAX_DIVISIONS + 1]
)
def test_equal_temperament_rejects_invalid_divisions(divisions):
    with pytest.raises(ValueError, match="Divisions"):
        _micro().to_equal_temperament(divisions)


def test_equal_temperament_steps():
    assert _micro().to_equal_temperament(4) == [0.0, 3.0, 6.0, 9.0]


@pytest.mark.parametrize("root", [-1, 128, 60.5, "C", None, True])
def test_custom_scale_rejects_invalid_root(root):
    with pytest.raises(ValueError, match="root"):
        CustomScaleBuilder().create_scale("mine", root, (0, 2, 4))


@pytest.mark.parametrize(
    "intervals", [(), [], None, "024", (0, -1), (0, 24), (0, 2.5), (0, 0, 4)]
)
def test_custom_scale_rejects_invalid_intervals(intervals):
    with pytest.raises(ValueError, match="interval"):
        CustomScaleBuilder().create_scale("mine", 0, intervals)


def test_custom_scale_sorts_its_intervals_and_degrees_follow():
    scale = CustomScaleBuilder().create_scale("mine", 0, (7, 0, 4))
    assert scale.intervals == (0, 4, 7)
    assert scale.degrees == (1, 2, 3)
    assert scale.scale_type is ScaleType.CUSTOM


def test_degree_names_continue_past_seven():
    scale = Scale(
        0, ScaleType.DIMINISHED, (0, 2, 3, 5, 6, 8, 9, 11), tuple(range(1, 9))
    )
    highlighter = ScaleDegreeHighlighter(scale)
    names = [highlighter.get_degree_name(n) for n in (0, 2, 3, 5, 6, 8, 9, 11)]
    assert names == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert highlighter.get_degree_name(1) == "X"


def test_degree_names_of_a_major_scale_are_unchanged():
    scale = Scale(
        0, ScaleType.MAJOR, (0, 2, 4, 5, 7, 9, 11), tuple(range(1, 8))
    )
    highlighter = ScaleDegreeHighlighter(scale)
    assert [highlighter.get_degree_name(n) for n in (0, 4, 7, 11)] == [
        "1",
        "3",
        "5",
        "7",
    ]


@pytest.mark.parametrize(
    "notes",
    [[60, 200, 64], [60, -1, 64], [60, "x", 64], [60, NAN, 64], [60, 60.5, 64]],
)
def test_detection_rejects_non_notes(notes):
    with pytest.raises(ValueError, match="MIDI note"):
        ScaleDetector().detect_scale_from_notes(notes)


@pytest.mark.parametrize("notes", [None, "cde", 5])
def test_detection_rejects_non_lists(notes):
    with pytest.raises(ValueError, match="Notes"):
        ScaleDetector().detect_scale_from_notes(notes)


def test_detection_still_finds_c_major():
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    best = ScaleDetector().detect_scale_from_notes(notes)
    assert best
    assert ScaleDetector().detect_scale_from_notes([60, 62]) == []
