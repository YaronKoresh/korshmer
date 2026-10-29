from korshmer.constants import (
    GM_DRUM_MAP,
    GRID_RESOLUTIONS,
    INSTRUMENT_DB,
    MICROTONAL_DIVISIONS,
    NOTE_NAMES,
    RESERVED_WINDOWS_NAMES,
    SCALES,
)


class TestNoteNames:
    def test_note_names_length(self):
        assert len(NOTE_NAMES) == 12

    def test_note_names_start_with_c(self):
        assert NOTE_NAMES[0] == "C"

    def test_note_names_chromatic_order(self):
        expected = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        assert NOTE_NAMES == expected


class TestScales:
    def test_scales_is_dict(self):
        assert isinstance(SCALES, dict)

    def test_major_scale(self):
        assert SCALES["Major"] == [0, 2, 4, 5, 7, 9, 11]

    def test_minor_scale(self):
        assert SCALES["Minor"] == [0, 2, 3, 5, 7, 8, 10]

    def test_chromatic_scale(self):
        assert SCALES["Chromatic"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def test_all_scales_start_with_zero(self):
        for scale_name, intervals in SCALES.items():
            assert intervals[0] == 0, f"Scale {scale_name} should start with 0"

    def test_all_scale_intervals_in_range(self):
        for scale_name, intervals in SCALES.items():
            for interval in intervals:
                assert 0 <= interval <= 11, (
                    f"Interval {interval} in {scale_name} out of range"
                )


class TestGridResolutions:
    def test_grid_resolutions_is_dict(self):
        assert isinstance(GRID_RESOLUTIONS, dict)

    def test_quarter_note_resolution(self):
        assert GRID_RESOLUTIONS["1/4"] == 1

    def test_sixteenth_note_resolution(self):
        assert GRID_RESOLUTIONS["1/16"] == 4


class TestMicrotonalDivisions:
    def test_microtonal_divisions_is_dict(self):
        assert isinstance(MICROTONAL_DIVISIONS, dict)

    def test_standard_semitones(self):
        assert MICROTONAL_DIVISIONS["Semitones (12-TET)"] == 12

    def test_quartertones(self):
        assert MICROTONAL_DIVISIONS["Quartertones (24-TET)"] == 24

    def test_all_divisions_positive(self):
        for name, divisions in MICROTONAL_DIVISIONS.items():
            assert divisions > 0, f"Division {name} should be positive"
            assert isinstance(divisions, int), (
                f"Division {name} should be integer"
            )


class TestDrumMap:
    def test_drum_map_is_dict(self):
        assert isinstance(GM_DRUM_MAP, dict)

    def test_kick_drum_mapping(self):
        assert GM_DRUM_MAP["Kick"] == 36

    def test_snare_drum_mapping(self):
        assert GM_DRUM_MAP["Snare"] == 38

    def test_all_drum_notes_in_midi_range(self):
        for drum_name, note in GM_DRUM_MAP.items():
            assert 0 <= note <= 127, (
                f"Drum {drum_name} note {note} out of MIDI range"
            )


class TestInstrumentDB:
    def test_instrument_db_is_dict(self):
        assert isinstance(INSTRUMENT_DB, dict)

    def test_instrument_db_has_categories(self):
        expected_categories = [
            "keys",
            "chromatic",
            "organ",
            "guitar",
            "bass",
            "strings",
            "vocals",
            "brass",
            "woodwind",
            "synth_lead",
            "synth_pad",
            "ethnic",
        ]
        for category in expected_categories:
            assert category in INSTRUMENT_DB, f"Category {category} missing"

    def test_instrument_tuples_format(self):
        for category, instruments in INSTRUMENT_DB.items():
            for instrument in instruments:
                assert isinstance(instrument, tuple), (
                    f"Instrument in {category} should be tuple"
                )
                assert len(instrument) == 4, (
                    f"Instrument {instrument} should have 4 elements"
                )
                name, low_note, high_note, program = instrument
                assert isinstance(name, str), "Instrument name should be string"
                assert isinstance(low_note, int), "Low note should be integer"
                assert isinstance(high_note, int), "High note should be integer"
                assert isinstance(program, int), (
                    "Program number should be integer"
                )
                assert 0 <= low_note <= 127, f"Low note {low_note} out of range"
                assert 0 <= high_note <= 127, (
                    f"High note {high_note} out of range"
                )
                assert 0 <= program <= 127, f"Program {program} out of range"
                assert low_note <= high_note, (
                    f"Low note should be <= high note for {name}"
                )


class TestReservedWindowsNames:
    def test_reserved_names_is_frozenset(self):
        assert isinstance(RESERVED_WINDOWS_NAMES, frozenset)

    def test_contains_con(self):
        assert "CON" in RESERVED_WINDOWS_NAMES

    def test_contains_prn(self):
        assert "PRN" in RESERVED_WINDOWS_NAMES

    def test_all_uppercase(self):
        for name in RESERVED_WINDOWS_NAMES:
            assert name.isupper(), f"Reserved name {name} should be uppercase"
