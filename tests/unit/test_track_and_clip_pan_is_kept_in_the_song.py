import pretty_midi
import pytest

from korshmer.ui.viewmodels.editor_vm import (
    ClipModel,
    EditorModel,
    TrackModel,
)

PAN_CONTROLLER = 10


def song_with_one_track_and_clip():
    model = EditorModel()
    track = model.add_track("Keys", program=4)
    model.add_note(track.id, 60, 0.0, 0.5)
    clip = model.add_clip("take.wav", duration=1.0)
    return model, track, clip


def pan_messages(instrument):
    return [
        change.value
        for change in instrument.control_changes
        if change.number == PAN_CONTROLLER
    ]


def test_a_new_track_and_a_new_clip_sit_in_the_centre():
    _model, track, clip = song_with_one_track_and_clip()
    assert track.pan == 0.0
    assert clip.pan == 0.0


def test_track_pan_is_limited_to_hard_left_and_hard_right():
    model, track, _clip = song_with_one_track_and_clip()
    model.set_track_pan(track.id, -3.0)
    assert track.pan == -1.0
    model.set_track_pan(track.id, 7.5)
    assert track.pan == 1.0


def test_clip_pan_is_limited_to_hard_left_and_hard_right():
    model, _track, clip = song_with_one_track_and_clip()
    model.set_clip_pan(clip.id, -2.0)
    assert clip.pan == -1.0
    model.set_clip_pan(clip.id, 2.0)
    assert clip.pan == 1.0


@pytest.mark.parametrize("bad", [None, "left", float("nan"), float("inf")])
def test_a_pan_that_is_not_a_number_falls_back_to_centre(bad):
    assert TrackModel(1, "Keys", pan=bad).pan == 0.0
    assert ClipModel(1, "take.wav", "take", pan=bad).pan == 0.0


def test_undo_and_redo_move_the_track_pan_back_and_forth():
    model, track, _clip = song_with_one_track_and_clip()
    model.snapshot()
    model.set_track_pan(track.id, 0.5)
    model.undo()
    assert model.tracks[track.id].pan == 0.0
    model.redo()
    assert model.tracks[track.id].pan == 0.5


def test_undo_and_redo_move_the_clip_pan_back_and_forth():
    model, _track, clip = song_with_one_track_and_clip()
    model.snapshot()
    model.set_clip_pan(clip.id, -0.25)
    model.undo()
    assert model.clip_by_id(clip.id).pan == 0.0
    model.redo()
    assert model.clip_by_id(clip.id).pan == -0.25


def test_pan_survives_saving_and_opening_the_song():
    model, track, clip = song_with_one_track_and_clip()
    model.set_track_pan(track.id, -0.4)
    model.set_clip_pan(clip.id, 0.8)
    opened = EditorModel.from_dict(model.to_dict())
    assert opened.tracks[track.id].pan == pytest.approx(-0.4)
    assert opened.clip_by_id(clip.id).pan == pytest.approx(0.8)


def test_a_song_saved_before_pan_existed_opens_in_the_centre():
    saved = {
        "tracks": [{"id": 1, "name": "Keys"}],
        "clips": [{"id": 1, "path": "take.wav", "duration": 1.0}],
    }
    opened = EditorModel.from_dict(saved)
    assert opened.tracks[1].pan == 0.0
    assert opened.clip_by_id(1).pan == 0.0


def test_a_copied_track_and_clip_keep_their_pan():
    model, track, clip = song_with_one_track_and_clip()
    model.set_track_pan(track.id, 0.3)
    model.set_clip_pan(clip.id, -0.6)
    assert track.copy().pan == pytest.approx(0.3)
    assert clip.copy().pan == pytest.approx(-0.6)


def test_a_clip_can_be_added_already_panned():
    clip = EditorModel().add_clip("take.wav", duration=1.0, pan=-1.0)
    assert clip.pan == -1.0


def test_playing_clips_carry_their_pan_to_the_mixer():
    model, _track, clip = song_with_one_track_and_clip()
    model.set_clip_pan(clip.id, 1.0)
    assert model.active_clips()[0]["pan"] == 1.0


def test_only_a_panned_track_tells_the_synth_where_it_sits():
    model, track, _clip = song_with_one_track_and_clip()
    centred = model.add_track("Bass", program=33)
    model.add_note(centred.id, 40, 0.0, 0.5)
    model.set_track_pan(track.id, -1.0)
    keys, bass = model.to_pretty_midi().instruments
    assert pan_messages(keys) == [0]
    assert pan_messages(bass) == []


def test_two_tracks_of_one_instrument_keep_their_own_sides():
    model = EditorModel()
    left = model.add_track("Left piano", program=0)
    right = model.add_track("Right piano", program=0)
    model.add_note(left.id, 60, 0.0, 0.5)
    model.add_note(right.id, 64, 0.0, 0.5)
    model.set_track_pan(left.id, -1.0)
    model.set_track_pan(right.id, 1.0)
    instruments = model.to_pretty_midi().instruments
    assert sorted(pan_messages(i)[0] for i in instruments) == [0, 127]


def test_an_exported_midi_file_opens_with_the_same_pan(tmp_path):
    model, track, _clip = song_with_one_track_and_clip()
    model.set_track_pan(track.id, 1.0)
    path = tmp_path / "song.mid"
    model.to_pretty_midi().write(str(path))
    (created,) = EditorModel().load_midi(pretty_midi.PrettyMIDI(str(path)))
    assert created.pan == 1.0


def test_a_midi_file_that_moves_its_pan_opens_at_the_first_position():
    song = pretty_midi.PrettyMIDI()
    keys = pretty_midi.Instrument(program=4)
    keys.notes.append(pretty_midi.Note(100, 60, 0.0, 2.0))
    keys.control_changes += [
        pretty_midi.ControlChange(PAN_CONTROLLER, 127, 1.0),
        pretty_midi.ControlChange(PAN_CONTROLLER, 0, 0.0),
    ]
    song.instruments.append(keys)
    (created,) = EditorModel().load_midi(song)
    assert created.pan == -1.0


def test_the_history_mark_changes_with_every_undo_step():
    model, track, _clip = song_with_one_track_and_clip()
    start = model.history_mark()
    model.snapshot()
    saved = model.history_mark()
    model.undo()
    assert len({start, saved, model.history_mark()}) == 3
