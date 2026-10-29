import numpy as np
import pretty_midi
import pytest

import korshmer.audio_processors as ap


class FakeFluidSynth:
    instances = []

    def __init__(self, *args, **kwargs):
        self.sounding = 0
        self.peak = 0
        self.note_on_calls = 0
        self.note_off_calls = 0
        FakeFluidSynth.instances.append(self)

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

    def program_select(self, *args):
        pass

    def note_on(self, chan, note, vel):
        self.note_on_calls += 1
        self.sounding += 1
        self.peak = max(self.peak, self.sounding)

    def note_off(self, chan, note):
        self.note_off_calls += 1
        self.sounding -= 1

    def get_samples(self, n):
        return np.zeros(n), np.zeros(n)


def _write_midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for start, end, pitch in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


@pytest.fixture
def render(tmp_path, monkeypatch, capsys):
    FakeFluidSynth.instances = []
    monkeypatch.setattr(ap, "FluidSynth", FakeFluidSynth)
    saved = {}
    monkeypatch.setattr(
        ap,
        "save_normalized_audio",
        lambda path, audio, sr: saved.update(path=path, audio=audio),
    )

    def _render(notes):
        midi = tmp_path / "in.mid"
        _write_midi(midi, notes)
        result = ap.render_midi_to_wav(
            str(midi), str(tmp_path / "out.wav"), str(tmp_path), "x.sf3"
        )
        out = capsys.readouterr().out
        return result, FakeFluidSynth.instances[0], out

    return _render


def test_129_simultaneous_notes_are_capped_at_128(render):
    notes = [(0.0, 1.0, 60) for _ in range(129)]
    result, synth, out = render(notes)

    assert result is not None
    assert synth.peak == 128
    assert synth.note_on_calls == 128
    assert synth.note_off_calls == 128
    assert synth.sounding == 0
    assert out.count("Warning") == 1
    assert "skipped 1 notes" in out


def test_exactly_128_simultaneous_notes_are_not_dropped(render):
    notes = [(0.0, 1.0, 60) for _ in range(128)]
    _, synth, out = render(notes)

    assert synth.note_on_calls == 128
    assert synth.peak == 128
    assert "Warning" not in out


def test_sequential_notes_are_never_dropped(render):
    notes = [(i * 0.5, i * 0.5 + 0.25, 60 + (i % 12)) for i in range(200)]
    _, synth, out = render(notes)

    assert synth.note_on_calls == 200
    assert synth.note_off_calls == 200
    assert synth.peak == 1
    assert "Warning" not in out
