import time

import pytest

TRACK_COUNT = 120
NOTES_PER_TRACK = 20
MAX_BUILD_SECONDS = 30.0
MAX_SERIALIZE_SECONDS = 30.0
MAX_LOAD_SECONDS = 30.0


def _build_model():
    from korshmer.ui.viewmodels.editor_vm import EditorModel

    model = EditorModel()
    for index in range(TRACK_COUNT):
        track = model.add_track(f"Track {index}")
        for n in range(NOTES_PER_TRACK):
            model.add_note(track.id, 40 + n, n * 0.5, 0.25)
    return model


@pytest.mark.performance
@pytest.mark.slow
class TestLargeProjects:
    def test_build_large_project(self):
        start = time.time()
        model = _build_model()
        elapsed = time.time() - start

        assert len(model.tracks) == TRACK_COUNT
        assert model.note_count() == TRACK_COUNT * NOTES_PER_TRACK
        assert elapsed < MAX_BUILD_SECONDS, f"Build took {elapsed:.3f}s"

    def test_round_trip_preserves_tracks_and_notes(self):
        from korshmer.ui.viewmodels.editor_vm import EditorModel

        model = _build_model()

        start = time.time()
        data = model.to_dict()
        serialize_elapsed = time.time() - start

        restored = EditorModel()
        start = time.time()
        restored.load_state(data)
        load_elapsed = time.time() - start

        assert len(restored.tracks) == TRACK_COUNT
        assert restored.note_count() == TRACK_COUNT * NOTES_PER_TRACK
        assert serialize_elapsed < MAX_SERIALIZE_SECONDS
        assert load_elapsed < MAX_LOAD_SECONDS
        for track_id in model.tracks:
            assert len(restored.notes_for_track(track_id)) == NOTES_PER_TRACK

    def test_snapshot_undo_with_many_tracks(self):
        model = _build_model()

        start = time.time()
        model.snapshot()
        snapshot_elapsed = time.time() - start

        model.add_track("Extra")
        assert len(model.tracks) == TRACK_COUNT + 1
        assert model.undo()
        assert len(model.tracks) == TRACK_COUNT
        assert model.note_count() == TRACK_COUNT * NOTES_PER_TRACK
        assert snapshot_elapsed < MAX_SERIALIZE_SECONDS
