import numpy as np
import pytest

from korshmer.midi_processing.drum_map import DrumNote
from korshmer.midi_processing.mpe_support import (
    MPEGlissandoGenerator,
    MPENote,
)
from korshmer.midi_processing.note_validation import (
    MIDI_NOTE_MAX,
    MIDI_NOTE_MIN,
    fold_into_midi_range,
    validate_midi_note,
)
from korshmer.midi_processing.pattern_generation import (
    Arpeggiator,
    ArpNote,
    ArpPattern,
)
from korshmer.midi_processing.quantization import Note
from korshmer.midi_processing.score_notation import ScoreNote
from korshmer.midi_processing.voice_leading import Voice, VoiceLeadingAnalyzer


class TestValidateMidiNote:
    @pytest.mark.parametrize("value", [0, 60, 127, np.int64(64), 60.0])
    def test_valid(self, value):
        assert validate_midi_note(value) == int(value)
        assert isinstance(validate_midi_note(value), int)

    @pytest.mark.parametrize(
        "value",
        [-1, 128, 200, 60.5, float("nan"), None, True, "60", [60]],
    )
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="MIDI pitch"):
            validate_midi_note(value)

    def test_bounds(self):
        assert (MIDI_NOTE_MIN, MIDI_NOTE_MAX) == (0, 127)

    @pytest.mark.parametrize(
        "pitch,expected", [(60, 60), (130, 118), (-1, 11), (127, 127)]
    )
    def test_fold(self, pitch, expected):
        assert fold_into_midi_range(pitch) == expected


def _score(pitch):
    return ScoreNote(pitch=pitch, duration=1.0, start_time=0.0, velocity=90)


class TestNoteDataclasses:
    @pytest.mark.parametrize("pitch", [128, -1])
    def test_score_note_rejects(self, pitch):
        with pytest.raises(ValueError):
            _score(pitch)

    def test_score_note_octave_valid(self):
        note = _score(60)
        assert note.get_octave() == 4

    def test_drum_note_rejects(self):
        with pytest.raises(ValueError):
            DrumNote(pitch=200, start_time=0.0, velocity=100)
        assert DrumNote(pitch=36, start_time=0.0, velocity=100).pitch == 36

    def test_other_note_types_reject(self):
        with pytest.raises(ValueError):
            ArpNote(pitch=128, start_time=0, duration=1, velocity=100)
        with pytest.raises(ValueError):
            Note(start_time=0, end_time=1, pitch=-5, velocity=100)
        with pytest.raises(ValueError):
            MPENote(128, 0, 1, 100, 2, [], [], [], [])
        with pytest.raises(ValueError):
            Voice(name="s", notes=[(0.0, 300)])

    def test_glissando_end_pitch_checked(self):
        with pytest.raises(ValueError):
            MPEGlissandoGenerator().generate_glissando(60, 200, 0.0, 1.0)


class TestArpeggiator:
    @pytest.mark.parametrize("pattern", list(ArpPattern))
    def test_high_notes_stay_in_range(self, pattern):
        arp = Arpeggiator()
        arp.set_octave_range(4)
        arp.pattern = pattern
        notes = arp.generate_arpeggio([120, 124, 127], duration=8.0)
        assert notes
        assert all(0 <= n.pitch <= 127 for n in notes)

    def test_input_notes_validated(self):
        with pytest.raises(ValueError):
            Arpeggiator().generate_arpeggio([60, 130], duration=1.0)

    def test_regular_range_unchanged(self):
        arp = Arpeggiator()
        arp.set_octave_range(2)
        arp.pattern = ArpPattern.UP_DOWN
        seq = arp._generate_sequence([60, 64, 67])
        assert seq == [60, 64, 67, 64, 60, 72, 76, 79, 76, 72]


class TestVoiceLeading:
    def test_voicings_in_range(self):
        optimizer = VoiceLeadingAnalyzer()
        progression = [[120, 124, 127], [0, 4, 7], [100, 103, 107]]
        voiced = optimizer.optimize_voice_leading(progression, num_voices=8)
        for chord in voiced:
            assert all(0 <= p <= 127 for p in chord)

    def test_regular_voicing_unchanged(self):
        optimizer = VoiceLeadingAnalyzer()
        voiced = optimizer.optimize_voice_leading([[0, 4, 7], [5, 9, 0]])
        assert voiced[0] == [48, 52, 55, 60]

    def test_invalid_chord_note_rejected(self):
        with pytest.raises(ValueError):
            VoiceLeadingAnalyzer().optimize_voice_leading([[0, 4, 200]])
