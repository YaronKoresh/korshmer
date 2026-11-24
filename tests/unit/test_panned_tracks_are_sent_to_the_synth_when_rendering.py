import logging

import numpy as np
import pretty_midi

from korshmer import audio_processors
from korshmer.audio_processing import guitar_amp
from korshmer.ui.viewmodels.editor_vm import EditorModel

PAN_CONTROLLER = 10
DRUM_BANK = 128
DRUM_CHANNEL = 9
RATE = 44100


class RecordingSynth:
    made = []

    def __init__(self, *args, **kwargs):
        self.pans = []
        self.sounds = []
        self.rendered = 0
        type(self).made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sf_load(self, path):
        return 1

    def set_reverb(self, *args):
        pass

    def set_chorus(self, *args):
        pass

    def program_select(self, channel, sfid, bank, program):
        self.sounds.append((channel, bank, program))

    def cc(self, channel, controller, value):
        if controller == PAN_CONTROLLER:
            self.pans.append((channel, value, self.rendered))

    def pitch_wheel_sens(self, *args):
        pass

    def pitch_bend(self, *args):
        pass

    def note_on(self, *args):
        pass

    def note_off(self, *args):
        pass

    def get_samples(self, count):
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(count) / RATE)
        self.rendered += count
        return tone, tone.copy()


def rendered_file(midi_path, tmp_path, monkeypatch):
    RecordingSynth.made = []
    saved = {}
    monkeypatch.setattr(audio_processors, "FluidSynth", RecordingSynth)
    monkeypatch.setattr(
        audio_processors,
        "save_normalized_audio",
        lambda path, audio, rate: saved.update(audio=audio),
    )
    audio_processors.render_midi_to_wav(
        str(midi_path), str(tmp_path / "song.wav"), "fluid", "font.sf3"
    )
    return saved.get("audio")


def rendered(model, tmp_path, monkeypatch):
    midi = tmp_path / "song.mid"
    model.to_pretty_midi().write(str(midi))
    return rendered_file(midi, tmp_path, monkeypatch)


def track_with_a_note(model, name, pan=0.0, **sound):
    track = model.add_track(name, **sound)
    model.add_note(track.id, 60, 0.0, 0.5)
    model.set_track_pan(track.id, pan)
    return track


def energy(side):
    return float(np.sum(np.square(side)))


def test_only_the_panned_track_is_panned_in_the_synth(tmp_path, monkeypatch):
    model = EditorModel()
    track_with_a_note(model, "Keys", pan=-1.0, program=4)
    track_with_a_note(model, "Bass", program=33)
    rendered(model, tmp_path, monkeypatch)
    (synth,) = RecordingSynth.made
    keys_channel = next(c for c, _bank, program in synth.sounds if program == 4)
    assert synth.pans == [(keys_channel, 0, 0)]


def test_one_panned_drum_track_is_panned_on_the_drum_channel(
    tmp_path, monkeypatch
):
    model = EditorModel()
    track_with_a_note(model, "Kit", pan=1.0, is_drum=True)
    rendered(model, tmp_path, monkeypatch)
    (synth,) = RecordingSynth.made
    assert synth.pans == [(DRUM_CHANNEL, 127, 0)]


def test_drum_tracks_panned_apart_cost_no_instrument_its_voice(
    tmp_path, monkeypatch
):
    model = EditorModel()
    track_with_a_note(model, "Kick", is_drum=True)
    track_with_a_note(model, "Hats", pan=1.0, is_drum=True)
    for program in range(15):
        track_with_a_note(model, f"Part {program}", program=program)
    audio = rendered(model, tmp_path, monkeypatch)
    sounds = [sound for synth in RecordingSynth.made for sound in synth.sounds]
    melodic = {program for _c, bank, program in sounds if bank != DRUM_BANK}
    drum_channels = {c for c, bank, _p in sounds if bank == DRUM_BANK}
    assert melodic == set(range(15))
    assert drum_channels == {DRUM_CHANNEL}
    assert energy(audio[:, 1]) > energy(audio[:, 0]) > 0.0


def test_an_instrument_that_cannot_sound_is_named_in_the_log(
    tmp_path, monkeypatch, caplog
):
    model = EditorModel()
    for program in range(16):
        track_with_a_note(model, f"Part {program + 1}", program=program)
    with caplog.at_level(logging.WARNING):
        rendered(model, tmp_path, monkeypatch)
    assert "Part 16" in caplog.text
    assert "Part 15" not in caplog.text


def test_a_midi_file_that_moves_its_pan_moves_it_in_the_synth(
    tmp_path, monkeypatch
):
    song = pretty_midi.PrettyMIDI()
    keys = pretty_midi.Instrument(program=4)
    keys.notes.append(pretty_midi.Note(100, 60, 0.0, 2.0))
    keys.control_changes += [
        pretty_midi.ControlChange(PAN_CONTROLLER, 0, 0.0),
        pretty_midi.ControlChange(PAN_CONTROLLER, 127, 1.0),
    ]
    song.instruments.append(keys)
    midi = tmp_path / "moving.mid"
    song.write(str(midi))
    rendered_file(midi, tmp_path, monkeypatch)
    (synth,) = RecordingSynth.made
    assert [(value, at) for _c, value, at in synth.pans] == [
        (0, 0),
        (127, RATE),
    ]


def test_an_amplified_guitar_panned_left_stays_out_of_the_right_side(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(guitar_amp, "amp", lambda audio, rate, settings: audio)
    model = EditorModel()
    track_with_a_note(model, "Guitar", pan=-1.0, program=30)
    audio = rendered(model, tmp_path, monkeypatch)
    assert float(np.max(np.abs(audio[:, 1]))) == 0.0
    assert float(np.max(np.abs(audio[:, 0]))) > 0.0
    assert [synth.pans for synth in RecordingSynth.made] == [[]]
