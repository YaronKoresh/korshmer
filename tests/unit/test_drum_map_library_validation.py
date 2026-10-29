import pytest

from korshmer.midi_processing.drum_map import (
    MAX_PATTERN_VARIATIONS,
    DrumHumanizer,
    DrumMapEditor,
    DrumNote,
    DrumPatternLibrary,
    DrumSample,
)

NAN = float("nan")


@pytest.mark.parametrize("note", [-1, 128, 36.5, "36", None, True])
def test_assign_sample_rejects_invalid_drum_notes(note):
    with pytest.raises(ValueError, match="drum note"):
        DrumMapEditor().assign_sample("kit", note, DrumSample("x", 36))


@pytest.mark.parametrize("sample", [None, "kick", 36, {}])
def test_assign_sample_rejects_non_samples(sample):
    with pytest.raises(ValueError, match="DrumSample"):
        DrumMapEditor().assign_sample("kit", 36, sample)


@pytest.mark.parametrize("name", ["", "  ", None, 5, "GM"])
def test_custom_map_names_must_be_text_and_not_gm(name):
    editor = DrumMapEditor()
    with pytest.raises(ValueError, match="map"):
        editor.create_custom_map(name)
    with pytest.raises(ValueError, match="map"):
        editor.assign_sample(name, 36, DrumSample("x", 36))


def test_assigned_sample_is_found_and_gm_is_untouched():
    editor = DrumMapEditor()
    sample = DrumSample("Mine", 36)
    editor.assign_sample("kit", 36, sample)
    assert editor.get_sample("kit", 36) is sample
    assert editor.get_sample("GM", 36).name == "Bass Drum 1"
    assert editor.get_sample("nope", 36) is None


def test_get_pattern_returns_a_copy_of_the_built_in_pattern():
    library = DrumPatternLibrary()
    size = len(DrumPatternLibrary.ROCK_BASIC)
    library.get_pattern("rock_basic").append((38, 9.0, 100))
    assert len(DrumPatternLibrary.ROCK_BASIC) == size
    assert len(library.get_pattern("rock_basic")) == size
    assert len(DrumPatternLibrary().get_pattern("rock_basic")) == size


def test_built_in_patterns_are_not_the_class_lists():
    library = DrumPatternLibrary()
    assert library.patterns["rock_basic"] is not DrumPatternLibrary.ROCK_BASIC
    assert library.patterns["jazz_swing"] is not DrumPatternLibrary.JAZZ_SWING


def test_added_pattern_is_stored_as_a_copy():
    library = DrumPatternLibrary()
    pattern = [(36, 0.0, 100)]
    library.add_pattern("mine", pattern)
    pattern.append((38, 1.0, 100))
    assert library.get_pattern("mine") == [(36, 0.0, 100)]


@pytest.mark.parametrize(
    "pattern",
    [
        [("a", 0.0, 100)],
        [(200, 0.0, 100)],
        [(36, -1.0, 100)],
        [(36, NAN, 100)],
        [(36, 0.0, 0)],
        [(36, 0.0, 128)],
        [(36, 0.0)],
        [36],
        "x",
        None,
    ],
)
def test_add_pattern_rejects_malformed_patterns(pattern):
    with pytest.raises(ValueError, match="Drum pattern|MIDI"):
        DrumPatternLibrary().add_pattern("mine", pattern)


@pytest.mark.parametrize("name", ["", None, 5])
def test_add_pattern_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="name"):
        DrumPatternLibrary().add_pattern(name, [(36, 0.0, 100)])


@pytest.mark.parametrize(
    "count", [-1, 2.5, "2", None, True, MAX_PATTERN_VARIATIONS + 1]
)
def test_variations_reject_invalid_counts(count):
    library = DrumPatternLibrary()
    with pytest.raises(ValueError, match="variations"):
        library.generate_pattern_variations(library.ROCK_BASIC, count)


def test_variations_reject_a_malformed_base_pattern():
    with pytest.raises(ValueError, match="base pattern"):
        DrumPatternLibrary().generate_pattern_variations([(1, 2)], 1)


def test_variations_keep_working_for_valid_input():
    library = DrumPatternLibrary()
    result = library.generate_pattern_variations(library.ROCK_BASIC, 3)
    assert len(result) == 3
    assert library.generate_pattern_variations(library.ROCK_BASIC, 0) == []
    for variation in result:
        assert all(1 <= velocity <= 127 for _p, _t, velocity in variation)


@pytest.mark.parametrize("timing", [NAN, float("inf"), -0.5, "0.1", None])
def test_humanizer_rejects_invalid_timing_variation(timing):
    humanizer = DrumHumanizer()
    humanizer.timing_variation = timing
    with pytest.raises(ValueError, match="Timing variation"):
        humanizer.humanize_pattern([DrumNote(38, 1.0, 100)])


@pytest.mark.parametrize("spread", [2.5, -1, "5", None, True])
def test_humanizer_rejects_invalid_velocity_variation(spread):
    humanizer = DrumHumanizer()
    humanizer.velocity_variation = spread
    with pytest.raises(ValueError, match="Velocity variation"):
        humanizer.humanize_pattern([DrumNote(38, 1.0, 100)])


def test_humanizer_keeps_notes_in_range_for_valid_settings():
    humanizer = DrumHumanizer()
    humanizer.timing_variation = 0.0
    humanizer.velocity_variation = 0
    result = humanizer.humanize_pattern(
        [DrumNote(36, 1.0, 100)], add_ghost_notes=False
    )
    assert [(n.pitch, n.start_time, n.velocity) for n in result] == [
        (36, 1.0, 100)
    ]
