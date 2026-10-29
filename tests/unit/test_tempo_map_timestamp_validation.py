import pytest

from korshmer.midi_processing.midi_file_enhancement import MIDIFileEnhancer


def _enhancer():
    return MIDIFileEnhancer()


class TestAddTempoChange:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_invalid_time_not_added(self, bad):
        e = _enhancer()
        e.add_tempo_change(bad, 120.0)
        assert len(e.tempo_map) == 0

    def test_negative_time_clamped(self):
        e = _enhancer()
        e.add_tempo_change(-5.0, 120.0)
        assert len(e.tempo_map) == 1
        assert e.tempo_map[0].time == 0.0

    def test_valid_time_added(self):
        e = _enhancer()
        e.add_tempo_change(10.0, 120.0)
        assert len(e.tempo_map) == 1
        assert e.tempo_map[0].time == 10.0


class TestOtherTimelineMethods:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_time_signature_invalid_time(self, bad):
        e = _enhancer()
        e.add_time_signature(bad, 4, 4)
        assert len(e.time_signatures) == 0

    def test_time_signature_negative_clamped(self):
        e = _enhancer()
        e.add_time_signature(-2.0, 3, 8)
        assert e.time_signatures[0].time == 0.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_key_signature_invalid_time(self, bad):
        e = _enhancer()
        e.add_key_signature(bad, 0)
        assert len(e.key_signatures) == 0

    def test_key_signature_valid(self):
        e = _enhancer()
        e.add_key_signature(4.0, 0)
        assert e.key_signatures[0].time == 4.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_lyric_invalid_time(self, bad):
        e = _enhancer()
        e.add_lyric(bad, "la")
        assert len(e.lyrics) == 0

    def test_lyric_negative_clamped(self):
        e = _enhancer()
        e.add_lyric(-1.0, "la")
        assert e.lyrics[0].time == 0.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_text_event_invalid_time(self, bad):
        e = _enhancer()
        e.add_text_event(bad, "note")
        assert len(e.text_events) == 0

    def test_text_event_valid(self):
        e = _enhancer()
        e.add_text_event(2.5, "note")
        assert e.text_events[0].time == 2.5


class TestMarkerUnchanged:
    def test_marker_still_works(self):
        e = _enhancer()
        e.add_marker(5.0, "Intro")
        assert e.markers[0].time == 5.0
        e.add_marker(float("nan"), "X")
        assert len(e.markers) == 1
