import numpy as np
import pretty_midi
import soundfile as sf

from korshmer.ui.viewmodels.editor_vm import (
    TRACK_PALETTE,
    ClipModel,
    EditorModel,
    NoteModel,
    TrackModel,
    compute_clip_waveform,
    format_clock,
)


def build_midi():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=4, is_drum=False, name="Keys")
    piano.notes.append(pretty_midi.Note(90, 60, 0.0, 0.5))
    piano.notes.append(pretty_midi.Note(80, 64, 0.5, 1.0))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drums.notes.append(pretty_midi.Note(100, 36, 0.0, 0.1))
    drums.notes.append(pretty_midi.Note(100, 38, 0.5, 0.6))
    pm.instruments.extend([piano, drums])
    return pm


def write_wav(path, seconds=2.0, rate=22050):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    envelope = np.linspace(0.1, 1.0, t.size)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 220 * t) * envelope), rate)
    return str(path)


class TestPaletteAndHelpers:
    def test_palette_has_distinct_colors(self):
        assert len(TRACK_PALETTE) >= 10
        assert len({c.lower() for c in TRACK_PALETTE}) == len(TRACK_PALETTE)
        assert all(c.startswith("#") and len(c) == 7 for c in TRACK_PALETTE)

    def test_format_clock(self):
        assert format_clock(0) == "00:00"
        assert format_clock(65.4) == "01:05"
        assert format_clock(-3) == "00:00"


class TestTracks:
    def test_add_track_assigns_distinct_colors(self):
        model = EditorModel()
        tracks = [model.add_track(f"T{i}") for i in range(4)]
        assert [t.id for t in tracks] == [1, 2, 3, 4]
        assert len({t.color for t in tracks}) == 4
        assert list(model.tracks) == [1, 2, 3, 4]

    def test_remove_track_removes_notes(self):
        model = EditorModel()
        keep = model.add_track("Keep")
        drop = model.add_track("Drop")
        model.add_note(keep.id, 60, 0.0, 0.5)
        model.add_note(drop.id, 62, 0.0, 0.5)
        model.remove_track(drop.id)
        assert drop.id not in model.tracks
        assert [n.track_id for n in model.notes] == [keep.id]

    def test_track_setters(self):
        model = EditorModel()
        track = model.add_track("Lead", program=5)
        model.rename_track(track.id, "  Melody ")
        model.set_track_program(track.id, 200)
        model.set_track_mute(track.id, True)
        model.set_track_solo(track.id, True)
        model.set_track_visible(track.id, False)
        model.set_track_color(track.id, "#123456")
        assert track.name == "Melody"
        assert track.program == 127
        assert track.muted and track.solo and not track.visible
        assert track.color == "#123456"
        model.rename_track(track.id, "   ")
        assert track.name == "Melody"
        model.set_track_drum(track.id, True)
        assert track.instrument_label() == "Drums"

    def test_next_palette_color_cycles(self):
        model = EditorModel()
        assert model.next_palette_color(TRACK_PALETTE[0]) == TRACK_PALETTE[1]
        assert model.next_palette_color(TRACK_PALETTE[-1]) == TRACK_PALETTE[0]


class TestNotes:
    def test_add_note_clamps_values(self):
        model = EditorModel()
        track = model.add_track("Lead")
        note = model.add_note(track.id, 200, -1.0, 0.0, velocity=0)
        assert note.pitch == 127
        assert note.start == 0.0
        assert note.duration > 0
        assert note.end > note.start
        assert note.velocity == 1
        low = model.add_note(track.id, -5, 1.0, 0.25, velocity=400)
        assert low.pitch == 0
        assert low.velocity == 127

    def test_add_note_without_tracks_creates_default(self):
        model = EditorModel()
        note = model.add_note(99, 60, 0.0, 0.5)
        assert len(model.tracks) == 1
        assert model.tracks[note.track_id].name == "Lead"

    def test_remove_and_lookup(self):
        model = EditorModel()
        track = model.add_track("Lead")
        a = model.add_note(track.id, 60, 0.0, 0.5)
        b = model.add_note(track.id, 62, 0.5, 0.5)
        model.remove_notes([a.id])
        assert model.notes_by_id() == {b.id: b}
        assert model.notes_for_track(track.id) == [b]
        assert model.note_count() == 1

    def test_transpose_shift_velocity(self):
        model = EditorModel()
        track = model.add_track("Lead")
        note = model.add_note(track.id, 125, 1.0, 0.5, velocity=120)
        model.transpose_notes([note.id], 5)
        assert note.pitch == 127
        model.transpose_notes([note.id], -200)
        assert note.pitch == 0
        model.shift_notes([note.id], -5.0)
        assert note.start == 0.0
        model.set_notes_velocity([note.id], 300)
        assert note.velocity == 127
        model.adjust_notes_velocity([note.id], -500)
        assert note.velocity == 1

    def test_quantize_strength(self):
        model = EditorModel()
        track = model.add_track("Lead")
        note = model.add_note(track.id, 60, 0.13, 0.5)
        model.quantize_notes([note.id], 0.125, strength=0.5)
        assert abs(note.start - 0.1275) < 1e-9
        model.quantize_notes([note.id], 0.125, strength=1.0)
        assert abs(note.start - 0.125) < 1e-9
        model.quantize_notes([note.id], 0.0)
        assert abs(note.start - 0.125) < 1e-9

    def test_humanize_is_deterministic_with_seed(self):
        first = EditorModel()
        second = EditorModel()
        for model in (first, second):
            track = model.add_track("Lead")
            for i in range(8):
                model.add_note(track.id, 60 + i, i * 0.25, 0.2, velocity=90)
            model.humanize_notes([n.id for n in model.notes], 0.02, 10, seed=7)
        assert [n.start for n in first.notes] == [n.start for n in second.notes]
        assert [n.velocity for n in first.notes] == [
            n.velocity for n in second.notes
        ]
        assert all(n.start >= 0 for n in first.notes)
        assert all(1 <= n.velocity <= 127 for n in first.notes)

    def test_duplicate_aligns_to_beats(self):
        model = EditorModel()
        track = model.add_track("Lead")
        a = model.add_note(track.id, 60, 0.0, 0.3)
        b = model.add_note(track.id, 64, 0.5, 0.3)
        created = model.duplicate_notes([a.id, b.id], beat_seconds=0.5)
        assert [n.start for n in created] == [1.0, 1.5]
        assert [n.pitch for n in created] == [60, 64]
        assert len(model.notes) == 4

    def test_copy_paste_preserves_shape(self):
        model = EditorModel()
        lead = model.add_track("Lead")
        bass = model.add_track("Bass")
        a = model.add_note(lead.id, 60, 1.0, 0.5, velocity=70)
        b = model.add_note(bass.id, 40, 1.5, 0.25, velocity=80)
        clipboard = model.copy_notes([a.id, b.id])
        assert [c["offset"] for c in clipboard] == [0.0, 0.5]
        model.remove_track(bass.id)
        created = model.paste_notes(clipboard, 4.0, track_id=lead.id)
        assert [n.start for n in created] == [4.0, 4.5]
        assert all(n.track_id == lead.id for n in created)
        assert created[1].velocity == 80

    def test_scale_note_times(self):
        model = EditorModel()
        track = model.add_track("Lead")
        note = model.add_note(track.id, 60, 1.0, 0.5)
        model.scale_note_times(2.0)
        assert note.start == 2.0 and note.duration == 1.0
        model.scale_note_times(0.0)
        assert note.start == 2.0


class TestClips:
    def test_add_clip_defaults(self):
        model = EditorModel()
        clip = model.add_clip(
            "/tmp/vocals_take.wav", kind="vocals", duration=3.0
        )
        assert clip.name == "vocals_take"
        assert clip.color.lower() == "#ea4335"
        assert clip.duration == 3.0
        assert model.clip_by_id(clip.id) is clip
        assert model.clip_by_id(999) is None

    def test_missing_file_duration_is_zero(self):
        model = EditorModel()
        clip = model.add_clip("/definitely/missing.wav")
        assert clip.duration == 0.0

    def test_move_mute_gain_remove(self):
        model = EditorModel()
        clip = model.add_clip("/tmp/a.wav", duration=1.0)
        model.move_clip(clip.id, -2.0)
        assert clip.start == 0.0
        model.move_clip(clip.id, 2.5)
        assert clip.end == 3.5
        model.set_clip_mute(clip.id, True)
        model.set_clip_gain(clip.id, 99.0)
        assert clip.muted and clip.gain == 4.0
        model.set_clip_waveform(clip.id, [0.1, 0.2], 7.0)
        assert clip.waveform == [0.1, 0.2] and clip.duration == 7.0
        model.remove_clip(clip.id)
        assert model.clips == []

    def test_active_clips_honour_mute_and_solo(self):
        model = EditorModel()
        track = model.add_track("Lead")
        vocals = model.add_clip("/tmp/v.wav", kind="vocals", duration=1.0)
        guitar = model.add_clip("/tmp/g.wav", kind="guitar", duration=1.0)
        assert [c["id"] for c in model.active_clips()] == [vocals.id, guitar.id]
        model.set_clip_mute(guitar.id, True)
        assert [c["id"] for c in model.active_clips()] == [vocals.id]
        model.set_clip_mute(guitar.id, False)
        model.set_track_solo(track.id, True)
        assert model.active_clips() == []
        assert [t.id for t in model.active_tracks()] == [track.id]
        model.set_clip_solo(vocals.id, True)
        assert [c["id"] for c in model.active_clips()] == [vocals.id]
        model.set_track_solo(track.id, False)
        assert model.active_tracks() == []
        active = model.active_clips()[0]
        assert set(active) >= {
            "id",
            "path",
            "name",
            "start",
            "duration",
            "gain",
            "muted",
            "kind",
        }


class TestMidiRoundtrip:
    def test_load_midi_creates_track_per_instrument(self):
        model = EditorModel()
        created = model.load_midi(build_midi())
        assert [t.name for t in created] == ["Keys", "Drums"]
        assert created[0].program == 4 and not created[0].is_drum
        assert created[1].is_drum
        assert len(model.notes) == 4
        assert len({t.color for t in created}) == 2

    def test_load_midi_merge_with_prefix(self):
        model = EditorModel()
        model.load_midi(build_midi())
        merged = model.load_midi(
            build_midi(), replace=False, name_prefix="Stem"
        )
        assert len(model.tracks) == 4
        assert merged[0].name == "Stem Keys"
        assert len(model.notes) == 8

    def test_unnamed_instruments_get_program_names(self):
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=33)
        inst.notes.append(pretty_midi.Note(100, 40, 0.0, 0.5))
        pm.instruments.append(inst)
        model = EditorModel()
        track = model.load_midi(pm)[0]
        assert track.name == "Bass Guitar"
        assert track.instrument_label() == "Bass Guitar"

    def test_classic_general_midi_names_are_replaced_on_load(self):
        pm = pretty_midi.PrettyMIDI()
        for program, name in ((33, "Electric Bass (finger)"), (40, "Violin")):
            inst = pretty_midi.Instrument(program=program, name=name)
            inst.notes.append(pretty_midi.Note(100, 40, 0.0, 0.5))
            pm.instruments.append(inst)
        pm.instruments.append(
            pretty_midi.Instrument(program=0, name="Lead Vocal")
        )
        pm.instruments[2].notes.append(pretty_midi.Note(100, 60, 0.0, 0.5))
        model = EditorModel()
        tracks = model.load_midi(pm)
        assert [t.name for t in tracks] == [
            "Bass Guitar",
            "Violin",
            "Lead Vocal",
        ]

    def test_export_keeps_overlapping_notes_of_a_single_track(self):
        model = EditorModel()
        track = model.add_track("Lead", program=0)
        model.add_note(track.id, 60, 0.0, 1.0)
        model.add_note(track.id, 60, 0.002, 1.0)
        notes = model.to_pretty_midi().instruments[0].notes
        assert [(n.start, n.end) for n in notes] == [(0.0, 1.0), (0.002, 1.002)]

    def test_merge_instruments_reuses_the_track_of_the_same_instrument(self):
        model = EditorModel()
        first = model.load_midi(build_midi())
        again = model.load_midi(
            build_midi(), replace=False, merge_instruments=True
        )
        assert len(model.tracks) == 2
        assert [t.id for t in again] == [t.id for t in first]
        assert model.note_count() == 8

    def test_merge_instruments_skips_muted_and_hidden_tracks(self):
        model = EditorModel()
        first = model.load_midi(build_midi())
        model.set_track_mute(first[0].id, True)
        model.set_track_visible(first[1].id, False)
        model.load_midi(build_midi(), replace=False, merge_instruments=True)
        assert len(model.tracks) == 4

    def test_merge_instruments_keeps_other_instruments_apart(self):
        model = EditorModel()
        model.load_midi(build_midi())
        pm = pretty_midi.PrettyMIDI()
        violin = pretty_midi.Instrument(program=40)
        violin.notes.append(pretty_midi.Note(80, 60, 0.0, 0.5))
        pm.instruments.append(violin)
        created = model.load_midi(pm, replace=False, merge_instruments=True)
        assert len(model.tracks) == 3
        assert created[0].name == "Violin"

    def test_export_has_one_track_per_instrument(self):
        model = EditorModel()
        model.load_midi(build_midi())
        second = build_midi()
        for inst in second.instruments:
            for note in inst.notes:
                note.pitch += 12
        model.load_midi(second, replace=False, name_prefix="Take 2")
        assert len(model.tracks) == 4
        pm = model.to_pretty_midi()
        assert sorted(i.name for i in pm.instruments) == [
            "Drums",
            "Mellow Keys",
        ]
        assert sum(len(i.notes) for i in pm.instruments) == 8

    def test_export_keeps_the_name_of_a_lone_track(self):
        model = EditorModel()
        model.load_midi(build_midi())
        pm = model.to_pretty_midi()
        assert [i.name for i in pm.instruments] == ["Keys", "Drums"]

    def test_to_pretty_midi_roundtrip(self):
        model = EditorModel()
        model.load_midi(build_midi())
        pm = model.to_pretty_midi(tempo=98.0)
        assert [i.name for i in pm.instruments] == ["Keys", "Drums"]
        assert pm.instruments[1].is_drum
        assert pm.instruments[0].program == 4
        assert len(pm.instruments[0].notes) == 2
        assert all(n.end > n.start for i in pm.instruments for n in i.notes)
        assert abs(pm.get_tempo_changes()[1][0] - 98.0) < 0.5

    def test_to_pretty_midi_respects_mute_solo(self):
        model = EditorModel()
        keys, drums = model.load_midi(build_midi())
        model.set_track_mute(drums.id, True)
        assert [i.name for i in model.to_pretty_midi().instruments] == ["Keys"]
        model.set_track_mute(drums.id, False)
        model.set_track_solo(drums.id, True)
        assert [i.name for i in model.to_pretty_midi().instruments] == ["Drums"]
        names = [
            i.name
            for i in model.to_pretty_midi(respect_mute_solo=False).instruments
        ]
        assert names == ["Keys", "Drums"]

    def test_empty_midi_does_not_crash(self):
        model = EditorModel()
        assert model.load_midi(pretty_midi.PrettyMIDI()) == []
        assert model.to_pretty_midi().instruments == []
        assert model.end_time() == 0.0
        assert model.load_midi(None) == []


class TestHistory:
    def test_undo_redo_cycle(self):
        model = EditorModel()
        track = model.add_track("Lead")
        assert not model.can_undo()
        model.snapshot()
        model.add_note(track.id, 60, 0.0, 0.5)
        assert model.can_undo()
        assert model.undo()
        assert model.notes == []
        assert model.can_redo()
        assert model.redo()
        assert len(model.notes) == 1
        assert model.undo()
        assert not model.undo()
        assert model.redo()
        assert len(model.notes) == 1
        assert not model.redo()

    def test_snapshot_clears_redo_and_discard(self):
        model = EditorModel()
        track = model.add_track("Lead")
        model.snapshot()
        model.add_note(track.id, 60, 0.0, 0.5)
        model.undo()
        model.snapshot()
        assert not model.can_redo()
        model.discard_last_snapshot()
        assert not model.can_undo()
        model.discard_last_snapshot()

    def test_undo_restores_clips_and_tracks(self):
        model = EditorModel()
        track = model.add_track("Lead")
        clip = model.add_clip("/tmp/v.wav", kind="vocals", duration=1.0)
        model.snapshot()
        model.remove_track(track.id)
        model.remove_clip(clip.id)
        model.undo()
        assert track.id in model.tracks
        assert model.clips[0].path == "/tmp/v.wav"
        model.add_note(track.id, 60, 0.0, 0.5)
        assert model.notes[0].id > 0

    def test_history_limit(self):
        model = EditorModel()
        model.max_undo_steps = 3
        for _ in range(6):
            model.snapshot()
        assert len(model._undo_stack) == 3


class TestSerialization:
    def test_dict_roundtrip(self):
        model = EditorModel(tempo=95.0)
        keys, drums = model.load_midi(build_midi())
        model.set_track_mute(drums.id, True)
        clip = model.add_clip(
            "/tmp/v.wav", start=1.5, kind="vocals", duration=4.0
        )
        model.set_clip_gain(clip.id, 1.5)
        data = model.to_dict()
        restored = EditorModel.from_dict(data)
        assert restored.tempo == 95.0
        assert [t.name for t in restored.tracks.values()] == ["Keys", "Drums"]
        assert restored.tracks[drums.id].muted
        assert len(restored.notes) == 4
        assert restored.clips[0].path == "/tmp/v.wav"
        assert restored.clips[0].start == 1.5
        assert restored.clips[0].gain == 1.5
        new_track = restored.add_track("Extra")
        assert new_track.id == drums.id + 1
        new_note = restored.add_note(new_track.id, 60, 0.0, 0.5)
        assert new_note.id > max(n.id for n in model.notes)

    def test_from_dict_tolerates_garbage(self):
        restored = EditorModel.from_dict(
            {"tempo": "abc", "notes": [{"id": 1, "track_id": 9}], "clips": [{}]}
        )
        assert restored.tempo == 120.0
        assert restored.notes == []
        assert restored.clips == []
        assert EditorModel.from_dict(None).tracks == {}

    def test_model_dicts(self):
        track = TrackModel.from_dict(
            TrackModel(3, "A", 9, True, "#000000").to_dict()
        )
        assert (track.id, track.name, track.program, track.is_drum) == (
            3,
            "A",
            9,
            True,
        )
        note = NoteModel.from_dict(NoteModel(2, 3, 60, 1.0, 0.5, 90).to_dict())
        assert (note.id, note.track_id, note.pitch, note.velocity) == (
            2,
            3,
            60,
            90,
        )
        clip = ClipModel.from_dict({"id": 4, "path": "/x/vox.wav"})
        assert clip.name == "vox" and clip.kind == "audio"
        assert clip.copy().to_dict() == clip.to_dict()

    def test_summary_and_end_time(self):
        model = EditorModel()
        track = model.add_track("Lead")
        model.add_note(track.id, 60, 10.0, 2.0)
        model.add_clip("/tmp/v.wav", start=60.0, duration=5.5)
        assert model.end_time() == 65.5
        assert (
            model.summary_text() == "1 tracks · 1 notes · 1 clips · end 01:06"
        )


class TestWaveform:
    def test_waveform_from_generated_wav(self, tmp_path):
        path = write_wav(tmp_path / "clip.wav", seconds=2.0, rate=22050)
        waveform, duration = compute_clip_waveform(path, points=512)
        assert len(waveform) == 512
        assert abs(duration - 2.0) < 1e-3
        assert max(waveform) == 1.0
        assert min(waveform) >= 0.0
        assert waveform[-1] > waveform[0]

    def test_waveform_short_file_and_odd_rate(self, tmp_path):
        path = write_wav(tmp_path / "short.wav", seconds=0.01, rate=11025)
        waveform, duration = compute_clip_waveform(path, points=64)
        assert len(waveform) == 64
        assert abs(duration - 0.01) < 1e-3

    def test_waveform_missing_or_silent(self, tmp_path):
        assert compute_clip_waveform(str(tmp_path / "nope.wav")) == ([], 0.0)
        silent = tmp_path / "silent.wav"
        sf.write(str(silent), np.zeros(4410), 44100)
        waveform, duration = compute_clip_waveform(str(silent), points=16)
        assert waveform == [0.0] * 16
        assert abs(duration - 0.1) < 1e-6

    def test_add_clip_probes_duration(self, tmp_path):
        path = write_wav(tmp_path / "probe.wav", seconds=1.5)
        model = EditorModel()
        clip = model.add_clip(path)
        assert abs(clip.duration - 1.5) < 1e-3


class TestVelocityValidation:
    def test_clamp_velocity_handles_non_numeric(self):
        from korshmer.ui.viewmodels.editor_vm import clamp_velocity

        for bad in (float("nan"), float("inf"), float("-inf"), None, "abc"):
            result = clamp_velocity(bad)
            assert isinstance(result, int)
            assert 1 <= result <= 127

    def test_clamp_velocity_clamps_range(self):
        from korshmer.ui.viewmodels.editor_vm import clamp_velocity

        assert clamp_velocity(-5) == 1
        assert clamp_velocity(200) == 127
        assert clamp_velocity(127.6) == 127
        assert clamp_velocity("90") == 90

    def test_clamp_pitch_handles_non_numeric(self):
        from korshmer.ui.viewmodels.editor_vm import clamp_pitch

        for bad in (float("nan"), float("inf"), None, "abc"):
            result = clamp_pitch(bad)
            assert isinstance(result, int)
            assert 0 <= result <= 127

    def test_load_state_with_invalid_velocity_does_not_crash(self):
        model = EditorModel()
        state = {
            "tempo": 120.0,
            "tracks": [
                {"id": 1, "name": "Lead", "color": "#1A73E8", "muted": False}
            ],
            "notes": [
                {
                    "id": 1,
                    "track_id": 1,
                    "pitch": 60,
                    "start": 0.0,
                    "duration": 0.5,
                    "velocity": None,
                },
                {
                    "id": 2,
                    "track_id": 1,
                    "pitch": 64,
                    "start": 0.5,
                    "duration": 0.5,
                    "velocity": float("nan"),
                },
            ],
            "clips": [],
        }
        model.load_state(state)
        velocities = [n.velocity for n in model.notes]
        assert len(velocities) == 2
        for v in velocities:
            assert isinstance(v, int)
            assert 1 <= v <= 127

    def test_paste_notes_with_none_velocity(self):
        model = EditorModel()
        track = model.add_track("Lead")
        clipboard = [
            {
                "pitch": 60,
                "start": 0.0,
                "duration": 0.5,
                "velocity": None,
            }
        ]
        model.paste_notes(clipboard, at_time=0.0, track_id=track.id)
        assert all(
            isinstance(n.velocity, int) and 1 <= n.velocity <= 127
            for n in model.notes
        )
