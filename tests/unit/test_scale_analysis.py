from korshmer.midi_processing.scale_analysis import (
    Scale,
    ScaleDetector,
    ScaleType,
)


class TestScaleCreation:
    def test_scale_creation(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(1, 2, 3, 4, 5, 6, 7),
        )
        assert scale.root == 0
        assert scale.scale_type == ScaleType.MAJOR

    def test_get_notes_c_major(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        notes = scale.get_notes()
        assert notes == [0, 2, 4, 5, 7, 9, 11]

    def test_get_notes_d_major(self):
        scale = Scale(
            root=2,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        notes = scale.get_notes()
        assert notes == [2, 4, 6, 7, 9, 11, 13]

    def test_get_name_c_major(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        assert scale.get_name() == "C Major"

    def test_get_name_f_sharp_minor(self):
        scale = Scale(
            root=6,
            scale_type=ScaleType.NATURAL_MINOR,
            intervals=(0, 2, 3, 5, 7, 8, 10),
            degrees=(),
        )
        assert scale.get_name() == "F# Natural Minor"

    def test_get_formula(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        formula = scale.get_formula()
        assert "1" in formula
        assert "3" in formula
        assert "5" in formula

    def test_contains_note_in_scale(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        assert scale.contains_note(0)
        assert scale.contains_note(4)
        assert scale.contains_note(7)

    def test_contains_note_not_in_scale(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        assert not scale.contains_note(1)
        assert not scale.contains_note(3)
        assert not scale.contains_note(6)

    def test_contains_note_different_octaves(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        assert scale.contains_note(12)
        assert scale.contains_note(24)
        assert scale.contains_note(60)


class TestScaleTypes:
    def test_major_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.MAJOR]
        assert intervals == (0, 2, 4, 5, 7, 9, 11)

    def test_natural_minor_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.NATURAL_MINOR]
        assert intervals == (0, 2, 3, 5, 7, 8, 10)

    def test_harmonic_minor_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.HARMONIC_MINOR]
        assert intervals == (0, 2, 3, 5, 7, 8, 11)

    def test_pentatonic_major_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.PENTATONIC_MAJOR]
        assert intervals == (0, 2, 4, 7, 9)

    def test_pentatonic_minor_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.PENTATONIC_MINOR]
        assert intervals == (0, 3, 5, 7, 10)

    def test_blues_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.BLUES]
        assert intervals == (0, 3, 5, 6, 7, 10)

    def test_whole_tone_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.WHOLE_TONE]
        assert intervals == (0, 2, 4, 6, 8, 10)

    def test_chromatic_scale_intervals(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.CHROMATIC]
        assert intervals == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


class TestExoticScales:
    def test_phrygian_dominant_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.PHRYGIAN_DOMINANT]
        assert intervals == (0, 1, 4, 5, 7, 8, 10)

    def test_hungarian_minor_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.HUNGARIAN_MINOR]
        assert intervals == (0, 2, 3, 6, 7, 8, 11)

    def test_neapolitan_major_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.NEAPOLITAN_MAJOR]
        assert intervals == (0, 1, 3, 5, 7, 9, 11)

    def test_neapolitan_minor_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.NEAPOLITAN_MINOR]
        assert intervals == (0, 1, 3, 5, 7, 8, 11)

    def test_arabic_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.ARABIC]
        assert intervals == (0, 1, 4, 5, 7, 8, 11)

    def test_japanese_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.JAPANESE]
        assert intervals == (0, 1, 5, 7, 8)

    def test_chinese_scale(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.CHINESE]
        assert intervals == (0, 4, 6, 7, 11)


class TestModalScales:
    def test_dorian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.DORIAN]
        assert intervals == (0, 2, 3, 5, 7, 9, 10)

    def test_phrygian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.PHRYGIAN]
        assert intervals == (0, 1, 3, 5, 7, 8, 10)

    def test_lydian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.LYDIAN]
        assert intervals == (0, 2, 4, 6, 7, 9, 11)

    def test_mixolydian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.MIXOLYDIAN]
        assert intervals == (0, 2, 4, 5, 7, 9, 10)

    def test_aeolian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.AEOLIAN]
        assert intervals == (0, 2, 3, 5, 7, 8, 10)

        assert (
            intervals == ScaleDetector.SCALE_TEMPLATES[ScaleType.NATURAL_MINOR]
        )

    def test_locrian_mode(self):
        intervals = ScaleDetector.SCALE_TEMPLATES[ScaleType.LOCRIAN]
        assert intervals == (0, 1, 3, 5, 6, 8, 10)


class TestScaleDetector:
    def test_scale_detector_has_all_templates(self):
        expected_scales = [
            ScaleType.MAJOR,
            ScaleType.NATURAL_MINOR,
            ScaleType.HARMONIC_MINOR,
            ScaleType.MELODIC_MINOR,
            ScaleType.DORIAN,
            ScaleType.PHRYGIAN,
            ScaleType.LYDIAN,
            ScaleType.MIXOLYDIAN,
            ScaleType.AEOLIAN,
            ScaleType.LOCRIAN,
            ScaleType.PHRYGIAN_DOMINANT,
            ScaleType.HUNGARIAN_MINOR,
            ScaleType.NEAPOLITAN_MAJOR,
            ScaleType.NEAPOLITAN_MINOR,
            ScaleType.WHOLE_TONE,
            ScaleType.DIMINISHED,
            ScaleType.CHROMATIC,
            ScaleType.PENTATONIC_MAJOR,
            ScaleType.PENTATONIC_MINOR,
            ScaleType.BLUES,
            ScaleType.ARABIC,
            ScaleType.JAPANESE,
            ScaleType.CHINESE,
        ]

        for scale_type in expected_scales:
            assert scale_type in ScaleDetector.SCALE_TEMPLATES


class TestNoteTransposition:
    def test_transpose_within_c_major(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )

        notes = scale.get_notes()
        assert 0 in notes
        assert 4 in notes

    def test_scale_snapping_c_sharp_to_c_major(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )

        note = 1
        notes = scale.get_notes()

        closest = min(notes, key=lambda x: abs(x - note))
        assert closest in [0, 2]

    def test_scale_snapping_preserves_octave(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )

        note = 61
        scale_notes = scale.get_notes()

        octave = note // 12
        notes_in_octave = [n + octave * 12 for n in scale_notes]

        closest = min(notes_in_octave, key=lambda x: abs(x - note))
        assert closest // 12 == octave


class TestScaleFormula:
    def test_major_scale_formula(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        formula = scale.get_formula()

        assert "1" in formula
        assert "2" in formula
        assert "3" in formula

    def test_minor_scale_formula(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.NATURAL_MINOR,
            intervals=(0, 2, 3, 5, 7, 8, 10),
            degrees=(),
        )
        formula = scale.get_formula()

        assert "1" in formula
        assert "b3" in formula


class TestEdgeCases:
    def test_scale_with_negative_root(self):
        scale = Scale(
            root=-1,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        name = scale.get_name()

        assert isinstance(name, str)

    def test_scale_with_high_root(self):
        scale = Scale(
            root=14,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        name = scale.get_name()

        assert "D" in name

    def test_contains_note_with_negative_note(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )

        result = scale.contains_note(-12)
        assert result == scale.contains_note(0)

    def test_empty_intervals(self):
        scale = Scale(
            root=0, scale_type=ScaleType.CUSTOM, intervals=(), degrees=()
        )
        notes = scale.get_notes()
        assert notes == []

    def test_single_note_scale(self):
        scale = Scale(
            root=0, scale_type=ScaleType.CUSTOM, intervals=(0,), degrees=()
        )
        notes = scale.get_notes()
        assert notes == [0]

    def test_large_interval_scale(self):
        scale = Scale(
            root=0,
            scale_type=ScaleType.CUSTOM,
            intervals=(0, 5, 10, 15, 20),
            degrees=(),
        )
        notes = scale.get_notes()
        assert len(notes) == 5
        assert notes == [0, 5, 10, 15, 20]


class TestScaleComparison:
    def test_major_vs_natural_minor_notes(self):
        major = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        minor = Scale(
            root=0,
            scale_type=ScaleType.NATURAL_MINOR,
            intervals=(0, 2, 3, 5, 7, 8, 10),
            degrees=(),
        )

        major_notes = set(major.get_notes())
        minor_notes = set(minor.get_notes())

        assert major_notes != minor_notes

    def test_relative_major_minor(self):

        c_major = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        a_minor = Scale(
            root=9,
            scale_type=ScaleType.NATURAL_MINOR,
            intervals=(0, 2, 3, 5, 7, 8, 10),
            degrees=(),
        )

        c_notes = {n % 12 for n in c_major.get_notes()}
        a_notes = {n % 12 for n in a_minor.get_notes()}

        assert c_notes == a_notes

    def test_parallel_major_minor(self):

        c_major = Scale(
            root=0,
            scale_type=ScaleType.MAJOR,
            intervals=(0, 2, 4, 5, 7, 9, 11),
            degrees=(),
        )
        c_minor = Scale(
            root=0,
            scale_type=ScaleType.NATURAL_MINOR,
            intervals=(0, 2, 3, 5, 7, 8, 10),
            degrees=(),
        )

        major_notes = set(c_major.get_notes())
        minor_notes = set(c_minor.get_notes())

        assert major_notes != minor_notes

        assert c_major.root == c_minor.root
