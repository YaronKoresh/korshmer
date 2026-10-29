from korshmer.midi_processing.midi_file_enhancement import (
    MIDIFileEnhancer,
    sanitize_lyrics_text,
)


class TestLyricTextSanitization:
    def test_lone_surrogate_is_removed_and_text_encodes_as_utf8(self):
        e = MIDIFileEnhancer()
        e.add_lyric(0.5, "a\ud800b")
        assert e.lyrics[0].text == "ab"
        e.lyrics[0].text.encode("utf-8")

    def test_control_characters_are_removed(self):
        e = MIDIFileEnhancer()
        e.add_lyric(0.5, "a\x00b\x1fc\x7fd")
        assert e.lyrics[0].text == "abcd"

    def test_non_string_becomes_empty_string(self):
        e = MIDIFileEnhancer()
        e.add_lyric(0.5, None)
        e.add_lyric(0.6, 123)
        assert [x.text for x in e.lyrics] == ["", ""]

    def test_valid_text_is_unchanged(self):
        assert sanitize_lyrics_text("la la  לה") == "la la  לה"

    def test_invalid_time_still_rejected(self):
        e = MIDIFileEnhancer()
        e.add_lyric(float("nan"), "la")
        assert e.lyrics == []
