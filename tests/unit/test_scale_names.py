import pytest

from korshmer.audio_processing.stem_session import (
    DEFAULT_SCALE,
    normalize_scale_name,
)
from korshmer.constants import SCALES
from korshmer.midi_processing.scale_analysis import (
    CustomScaleBuilder,
    ScaleType,
)
from korshmer.utils.scale_names import (
    MAX_SCALE_NAME_LENGTH,
    sanitize_scale_name,
)


class TestSanitizeScaleName:
    def test_none_returns_empty(self):
        assert sanitize_scale_name(None) == ""

    def test_non_string_int_returns_empty(self):
        assert sanitize_scale_name(5) == ""

    def test_removes_null_and_control_chars(self):
        assert sanitize_scale_name("My\x00Scale\x07") == "MyScale"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_scale_name("  Dorian  ") == "Dorian"

    def test_truncates_to_max_length(self):
        name = "A" * (MAX_SCALE_NAME_LENGTH + 50)
        result = sanitize_scale_name(name)
        assert len(result) == MAX_SCALE_NAME_LENGTH

    def test_does_not_html_escape_angle_brackets(self):
        assert sanitize_scale_name("<b>X</b> & Y") == "<b>X</b> & Y"

    def test_keeps_ampersand_unescaped(self):
        assert sanitize_scale_name("Bass & Treble") == "Bass & Treble"

    def test_keeps_non_ascii_hebrew(self):
        assert sanitize_scale_name("מז'ור שלי") == "מז'ור שלי"

    def test_removes_lone_surrogate(self):
        assert sanitize_scale_name("A\ud800B") == "AB"

    def test_removes_del_character(self):
        assert sanitize_scale_name("Scale\x7fName") == "ScaleName"

    def test_empty_string_stays_empty(self):
        assert sanitize_scale_name("") == ""

    def test_whitespace_only_becomes_empty(self):
        assert sanitize_scale_name("   ") == ""


class TestScalesKeysSanitization:
    def test_all_scales_keys_pass_unchanged(self):
        for key in SCALES:
            assert sanitize_scale_name(key) == key, (
                f"SCALES key {key!r} was altered by sanitize_scale_name"
            )


class TestCustomScaleBuilderSanitization:
    def test_create_scale_strips_control_chars_from_name(self):
        builder = CustomScaleBuilder()
        builder.create_scale("A\x00B", 0, (0, 2, 4, 5, 7, 9, 11))
        assert "AB" in builder.list_scales()
        assert "A\x00B" not in builder.list_scales()

    def test_create_scale_raises_on_whitespace_only_name(self):
        builder = CustomScaleBuilder()
        with pytest.raises(ValueError, match="scale name must not be empty"):
            builder.create_scale("   ", 0, (0, 2, 4))

    def test_create_scale_raises_on_control_char_only_name(self):
        builder = CustomScaleBuilder()
        with pytest.raises(ValueError, match="scale name must not be empty"):
            builder.create_scale("\x00\x01", 0, (0, 2, 4))

    def test_load_scale_finds_entry_with_control_chars(self):
        builder = CustomScaleBuilder()
        builder.create_scale("My Scale", 0, (0, 2, 4, 5, 7, 9, 11))

        result = builder.load_scale("My Scale")
        assert result is not None
        assert result.scale_type == ScaleType.CUSTOM

    def test_delete_scale_works_after_sanitization(self):
        builder = CustomScaleBuilder()
        builder.create_scale("TestScale", 0, (0, 2, 4))

        deleted = builder.delete_scale("TestScale\x00")
        assert deleted is True
        assert "TestScale" not in builder.list_scales()

    def test_save_scale_raises_on_empty_name(self):
        from korshmer.midi_processing.scale_analysis import Scale

        builder = CustomScaleBuilder()
        scale = Scale(
            root=0,
            scale_type=ScaleType.CUSTOM,
            intervals=(0, 2, 4),
            degrees=(1, 2, 3),
        )
        with pytest.raises(ValueError, match="scale name must not be empty"):
            builder.save_scale("", scale)


class TestNormalizeScaleNameSanitization:
    def test_minor_string_returns_minor(self):
        assert normalize_scale_name("minor") == "Minor"

    def test_major_string_returns_major(self):
        assert normalize_scale_name("major") == "Major"

    def test_control_chars_only_returns_default(self):
        assert normalize_scale_name("\x00\x01") == DEFAULT_SCALE

    def test_none_returns_default(self):
        assert normalize_scale_name(None) == DEFAULT_SCALE

    def test_empty_string_returns_default(self):
        assert normalize_scale_name("") == DEFAULT_SCALE

    def test_known_scale_key_passes_through(self):
        assert (
            normalize_scale_name("Spanish (Phrygian Dominant)")
            == "Spanish (Phrygian Dominant)"
        )

    def test_control_chars_stripped_before_lookup(self):

        assert normalize_scale_name("\x00Major") == "Major"

    def test_whitespace_stripped_before_lookup(self):
        assert normalize_scale_name("  Minor  ") == "Minor"
