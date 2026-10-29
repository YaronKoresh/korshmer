import sys

import pytest

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="memray supports only Linux and macOS",
    ),
]

pytest.importorskip("pytest_memray")


TRACKS = 50
NOTES_PER_TRACK = 20


@pytest.fixture(scope="module")
def EditorModel():
    from korshmer.ui.viewmodels.editor_vm import EditorModel

    return EditorModel


def _build_model(EditorModel):
    model = EditorModel()
    for index in range(TRACKS):
        track = model.add_track(f"Track {index}")
        for n in range(NOTES_PER_TRACK):
            model.add_note(track.id, 40 + n % 40, n * 0.5, 0.25)
    return model


@pytest.mark.limit_memory("4 MB")
def test_add_track_and_note_memory_bounded(EditorModel):
    model = _build_model(EditorModel)
    assert len(model.tracks) == TRACKS
    assert model.note_count() == TRACKS * NOTES_PER_TRACK


@pytest.mark.limit_memory("4 MB")
def test_repeated_save_and_load_memory_bounded(EditorModel):
    model = _build_model(EditorModel)
    for _ in range(20):
        restored = EditorModel()
        restored.load_state(model.to_dict())
    assert restored.note_count() == TRACKS * NOTES_PER_TRACK


@pytest.mark.limit_memory("4 MB")
def test_repeated_add_remove_track_memory_bounded(EditorModel):
    model = EditorModel()
    for index in range(500):
        track = model.add_track(f"Track {index}")
        model.add_note(track.id, 60, 0.0, 0.5)
        model.remove_track(track.id)
    assert len(model.tracks) <= 1
