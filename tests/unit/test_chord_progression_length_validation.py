import pytest

from korshmer.ai.copilot import MusicalContext, MusicCopilot
from korshmer.audio_processing.music_generation import MusicGenerator
from korshmer.midi_processing.note_validation import (
    MAX_CHORD_PROGRESSION_LENGTH,
    validate_chord_progression,
)
from korshmer.midi_processing.voice_leading import VoiceLeadingAnalyzer

CHORD = [0, 4, 7]


class TestValidateChordProgression:
    def test_accepts_normal(self):
        chords = [CHORD, [5, 9, 0]]
        assert validate_chord_progression(chords) is chords

    def test_accepts_exactly_max(self):
        chords = [CHORD] * MAX_CHORD_PROGRESSION_LENGTH
        assert validate_chord_progression(chords) is chords

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_chord_progression([])

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            validate_chord_progression(
                [CHORD] * (MAX_CHORD_PROGRESSION_LENGTH + 1)
            )

    def test_rejects_non_sequence(self):
        with pytest.raises(ValueError):
            validate_chord_progression("not a list")


class TestMusicGenerator:
    def test_melody_normal(self):
        g = MusicGenerator()
        prog = [[0, 4, 7], [5, 9, 0], [7, 11, 2], [0, 4, 7]]
        midi = g.generate_melody_from_chords(prog)
        assert len(midi.instruments[0].notes) > 0

    def test_melody_exactly_max(self):
        g = MusicGenerator()
        g.generate_melody_from_chords(
            [[0, 4, 7]] * MAX_CHORD_PROGRESSION_LENGTH
        )

    def test_melody_too_long(self):
        g = MusicGenerator()
        with pytest.raises(ValueError) as exc:
            g.generate_melody_from_chords(
                [[0, 4, 7]] * (MAX_CHORD_PROGRESSION_LENGTH + 1)
            )
        assert "256" in str(exc.value) and "257" in str(exc.value)

    def test_melody_empty_raises_valueerror_not_zerodivision(self):
        g = MusicGenerator()
        with pytest.raises(ValueError):
            g.generate_melody_from_chords([])

    def test_bass_normal(self):
        g = MusicGenerator()
        prog = [[0, 4, 7], [5, 9, 0]]
        midi = g.generate_bass_line(prog)
        assert len(midi.instruments[0].notes) > 0

    def test_bass_too_long(self):
        g = MusicGenerator()
        with pytest.raises(ValueError):
            g.generate_bass_line(
                [[0, 4, 7]] * (MAX_CHORD_PROGRESSION_LENGTH + 1)
            )

    def test_bass_empty(self):
        g = MusicGenerator()
        with pytest.raises(ValueError):
            g.generate_bass_line([])


class TestVoiceLeading:
    def test_empty_still_returns_empty(self):
        v = VoiceLeadingAnalyzer()
        assert v.optimize_voice_leading([]) == []

    def test_too_long_raises(self):
        v = VoiceLeadingAnalyzer()
        with pytest.raises(ValueError):
            v.optimize_voice_leading(
                [[0, 4, 7]] * (MAX_CHORD_PROGRESSION_LENGTH + 1)
            )


class TestCopilotPrompt:
    def test_too_long_rejected(self):
        c = MusicCopilot()
        ctx = MusicalContext(
            chord_progression=["C"] * (MAX_CHORD_PROGRESSION_LENGTH + 1)
        )
        with pytest.raises(ValueError):
            c._build_prompt(ctx, "help")

    def test_empty_allowed(self):
        c = MusicCopilot()
        ctx = MusicalContext(chord_progression=[])
        assert c._build_prompt(ctx, "help")
