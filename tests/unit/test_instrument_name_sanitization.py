import html

from korshmer.utils.instrument_names import (
    MAX_INSTRUMENT_NAME_LENGTH,
    sanitize_instrument_name,
)


class TestSanitizeInstrumentName:
    def test_removes_control_characters(self):
        assert sanitize_instrument_name("Piano\x00\x07") == "Piano"

    def test_none_returns_empty(self):
        assert sanitize_instrument_name(None) == ""

    def test_non_string_returns_empty(self):
        assert sanitize_instrument_name(123) == ""

    def test_keeps_non_ascii(self):
        assert sanitize_instrument_name("פסנתר") == "פסנתר"

    def test_does_not_escape_html(self):
        assert sanitize_instrument_name("<b>Bass</b>") == "<b>Bass</b>"

    def test_keeps_ampersand_unescaped(self):
        assert sanitize_instrument_name("Bass & Drums") == "Bass & Drums"

    def test_truncates_to_max_length(self):
        name = "a" * (MAX_INSTRUMENT_NAME_LENGTH + 50)
        assert len(sanitize_instrument_name(name)) == MAX_INSTRUMENT_NAME_LENGTH

    def test_strips_whitespace(self):
        assert sanitize_instrument_name("  Lead  ") == "Lead"

    def test_removes_lone_surrogate(self):
        assert sanitize_instrument_name("A\ud800B") == "AB"


class TestEscapeInReport:
    def test_escape_produces_entities(self):
        name = sanitize_instrument_name("<h1>X</h1><img src='a.png'>")
        escaped = html.escape(str(name))
        assert "&lt;h1&gt;" in escaped
        assert "<h1>" not in escaped
