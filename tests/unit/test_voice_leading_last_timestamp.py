from korshmer.midi_processing.voice_leading import Voice, VoiceLeadingAnalyzer


def _types(violations):
    return sorted((v.type, v.time) for v in violations)


def test_last_timestamp_range_and_crossing_are_reported():
    voices = [
        Voice("bass", [(0.0, 48), (1.0, 50)], range_low=40, range_high=60),
        Voice("tenor", [(0.0, 60), (1.0, 45)], range_low=50, range_high=70),
    ]
    found = _types(VoiceLeadingAnalyzer().analyze_voices(voices))
    assert ("out_of_range", 1.0) in found
    assert ("voice_crossing", 1.0) in found


def test_single_timestamp_out_of_range_is_reported():
    voices = [Voice("bass", [(0.0, 100)], range_low=40, range_high=60)]
    found = _types(VoiceLeadingAnalyzer().analyze_voices(voices))
    assert found == [("out_of_range", 0.0)]


def test_valid_voices_report_nothing():
    voices = [
        Voice("bass", [(0.0, 48), (1.0, 50)], range_low=40, range_high=60),
        Voice("tenor", [(0.0, 60), (1.0, 59)], range_low=50, range_high=70),
    ]
    assert VoiceLeadingAnalyzer().analyze_voices(voices) == []
