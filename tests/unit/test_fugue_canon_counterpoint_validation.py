import pytest

from korshmer.midi_processing.voice_leading import (
    MAX_CANON_OFFSET,
    CanonDetector,
    CounterpointChecker,
    FugueAnalyzer,
)

NAN = float("nan")
MELODY = [(i * 0.5, 60 + i) for i in range(8)]


@pytest.mark.parametrize("length", [0, -2, 2.5, "4", None, True, 5000])
def test_subject_length_must_be_a_sensible_whole_number(length):
    with pytest.raises(ValueError, match="Subject length"):
        FugueAnalyzer().detect_subject(MELODY, length)


@pytest.mark.parametrize(
    "melody",
    [[1, 2, 3, 4], [(0.0,)] * 4, [(NAN, 60)] * 4, [(0.0, 200)] * 4, None],
)
def test_subject_melody_must_be_time_pitch_pairs(melody):
    with pytest.raises(ValueError):
        FugueAnalyzer().detect_subject(melody, 4)


def test_subject_of_a_valid_melody():
    subject = FugueAnalyzer().detect_subject(MELODY, 4)
    assert subject.notes == [60, 61, 62, 63]
    assert subject.start_time == 0.0
    assert subject.duration == 1.5
    assert FugueAnalyzer().detect_subject(MELODY[:2], 4) is None


def test_subject_entries_reject_a_bad_melody():
    analyzer = FugueAnalyzer()
    subject = analyzer.detect_subject(MELODY, 3)
    with pytest.raises(ValueError):
        analyzer.find_subject_entries(subject, [(0.0, "x")])
    assert analyzer.find_subject_entries(subject, MELODY)[0] == 0.0


@pytest.mark.parametrize(
    "offset",
    [NAN, float("inf"), -1.0, MAX_CANON_OFFSET + 1, "2", None, True],
)
def test_canon_offset_must_be_a_bounded_finite_number(offset):
    with pytest.raises(ValueError, match="time offset"):
        CanonDetector().detect_canon([MELODY, MELODY], offset)


@pytest.mark.parametrize("voices", [None, "ab", [MELODY, [1, 2]], [[(0, 300)]]])
def test_canon_voices_must_be_melodies(voices):
    with pytest.raises(ValueError):
        CanonDetector().detect_canon(voices)


def test_canon_is_found_for_a_shifted_copy():
    voice = [(i * 0.5, 60 + (i * 3) % 7) for i in range(8)]
    shifted = [(t + 1.0, p + 5) for t, p in voice]
    found = CanonDetector().detect_canon([voice, shifted], 2.0)
    assert found
    assert found[0][:2] == (0, 1)
    assert found[0][2] == pytest.approx(1.0)
    assert found[0][3] == 5


@pytest.mark.parametrize(
    "notes", [["x", 60], [NAN, 60], [60.5, 60], [200, 60], [-1, 60], None]
)
def test_counterpoint_rejects_non_notes_in_either_voice(notes):
    checker = CounterpointChecker()
    with pytest.raises(ValueError):
        checker.check_counterpoint(notes, [60, 62])
    with pytest.raises(ValueError):
        checker.check_counterpoint([60, 62], notes)


def test_counterpoint_still_reports_rule_violations():
    checker = CounterpointChecker()
    result = checker.check_counterpoint([60, 62, 64], [67, 68, 60])
    assert any("Dissonant" in line for line in result)
    assert checker.check_counterpoint([], []) == []
