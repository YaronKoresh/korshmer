import pytest

from korshmer.midi_processing.voice_leading import VoiceLeadingAnalyzer


def _classes(voicing):
    return {pitch % 12 for pitch in voicing}


@pytest.mark.parametrize(
    "progression",
    [
        [[0, 4, 7], [5, 9, 0]],
        [[0, 4, 7], [7, 11, 2], [0, 4, 7]],
        [[0, 4, 7], [9, 0, 4], [5, 9, 0], [7, 11, 2], [0, 4, 7]],
    ],
)
def test_every_chord_keeps_all_its_tones(progression):
    voiced = VoiceLeadingAnalyzer().optimize_voice_leading(progression, 4)

    for chord, voicing in zip(progression, voiced, strict=True):
        assert len(voicing) == 4
        assert _classes(voicing) == {note % 12 for note in chord}
        assert all(0 <= pitch <= 127 for pitch in voicing)


def test_movement_stays_small():
    voiced = VoiceLeadingAnalyzer().optimize_voice_leading(
        [[0, 4, 7], [5, 9, 0]], 4
    )

    movement = sum(
        abs(a - b) for a, b in zip(voiced[0], voiced[1], strict=True)
    )
    assert movement <= 5


def test_more_chord_tones_than_voices_covers_as_many_as_possible():
    voiced = VoiceLeadingAnalyzer().optimize_voice_leading(
        [[0, 4, 7], [0, 3, 6, 9]], 3
    )

    assert len(_classes(voiced[1])) == 3


def test_single_voice_progression():
    voiced = VoiceLeadingAnalyzer().optimize_voice_leading(
        [[0, 4, 7], [5, 9, 0]], 1
    )

    assert [len(v) for v in voiced] == [1, 1]


def test_empty_chord_is_rejected():
    with pytest.raises(ValueError):
        VoiceLeadingAnalyzer().optimize_voice_leading([[0, 4, 7], []], 4)
    with pytest.raises(ValueError):
        VoiceLeadingAnalyzer().optimize_voice_leading([[]], 4)


@pytest.mark.parametrize("voices", [0, -2, 2.5, True, None, 17])
def test_invalid_voice_count_is_rejected(voices):
    with pytest.raises(ValueError):
        VoiceLeadingAnalyzer().optimize_voice_leading([[0, 4, 7]], voices)
