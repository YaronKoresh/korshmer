import sys

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

if not QApplication.instance():
    app = QApplication(sys.argv)

from korshmer.audio_processing.vocal_cleanup import (
    VOCAL_ABOVE_MIDI_DB,
    vocal_clip_gain,
)
from korshmer.ui.viewmodels.editor_vm import (
    RHYTHM_BOOST,
    EditorModel,
    rhythm_boost,
)

SR = 22050


def _played(model):
    return {
        (inst.program, inst.is_drum): [n.velocity for n in inst.notes]
        for inst in model.to_pretty_midi().instruments
    }


def _model():
    model = EditorModel(tempo=120.0)
    piano = model.add_track("Keys", program=0)
    bass = model.add_track("Bass", program=33)
    drums = model.add_track("Drums", is_drum=True)
    for track in (piano, bass, drums):
        model.add_note(track.id, 40, 0.0, 0.5, 80)
    return model, piano, bass, drums


def test_bass_and_drums_play_a_quarter_stronger_at_every_level():
    model, piano, bass, drums = _model()
    assert RHYTHM_BOOST == 1.25
    played = _played(model)
    assert played[(0, False)] == [80]
    assert played[(33, False)] == [100]
    assert played[(0, True)] == [100]
    for track in (piano, bass, drums):
        model.set_track_gain(track.id, 0.5)
    played = _played(model)
    assert played[(0, False)] == [40]
    assert played[(33, False)] == [50]
    assert played[(0, True)] == [50]


def test_the_lift_never_changes_the_stored_notes_or_the_shown_level():
    model, _piano, bass, _drums = _model()
    _played(model)
    _played(model)
    assert [n.velocity for n in model.notes] == [80, 80, 80]
    assert bass.gain == 1.0
    loud = model.add_note(bass.id, 41, 1.0, 0.5, 120)
    assert max(_played(model)[(33, False)]) == 127
    assert loud.velocity == 120


@pytest.mark.parametrize(
    "program, is_drum, expected",
    [(32, False, 1.25), (39, False, 1.25), (40, False, 1.0), (0, True, 1.25)],
)
def test_only_bass_programs_and_drums_are_lifted(program, is_drum, expected):
    assert rhythm_boost(program, is_drum) == expected


def _sung(level_db, seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t)
    return (tone * 10 ** (level_db / 20.0) * np.sqrt(2)).astype(np.float32)


def test_a_separated_vocal_starts_a_third_louder_than_before():
    assert VOCAL_ABOVE_MIDI_DB == -1.5
    gain = vocal_clip_gain(_sung(-20.0), midi_level_db=-12.0)
    before = vocal_clip_gain(
        _sung(-20.0), midi_level_db=-12.0, above_midi_db=-4.0
    )
    assert gain / before == pytest.approx(10 ** (2.5 / 20.0), rel=1e-3)
    assert 1.3 < gain / before < 1.37
