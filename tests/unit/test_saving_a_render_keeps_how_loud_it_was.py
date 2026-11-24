import importlib

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import korshmer.audio
import korshmer.audio_processing
from korshmer import audio_processors
from korshmer.audio.io import save_normalized_audio

RATE = 44100
WAYS_IN = [
    lambda: importlib.import_module("korshmer.audio.io").save_normalized_audio,
    lambda: (
        importlib.import_module(
            "korshmer.audio_processing.audio_io"
        ).save_normalized_audio
    ),
    lambda: korshmer.audio.save_normalized_audio,
    lambda: korshmer.audio_processing.save_normalized_audio,
]


@pytest.fixture(params=range(len(WAYS_IN)))
def save(request):
    return WAYS_IN[request.param]()


def saved_peak(save_audio, tmp_path, audio, **options):
    out = tmp_path / "render.wav"
    save_audio(out, np.asarray(audio, np.float32), RATE, **options)
    data, _rate = sf.read(out, dtype="float32")
    return float(np.max(np.abs(data)))


def test_every_way_to_reach_the_saver_gives_the_same_saver():
    assert all(way() is save_normalized_audio for way in WAYS_IN)


def test_a_quiet_render_is_saved_as_quiet_as_it_was(save, tmp_path):
    assert saved_peak(save, tmp_path, [0.1, -0.2, 0.1]) == pytest.approx(
        0.2, abs=1e-3
    )


def test_a_render_that_would_clip_is_turned_down_to_the_ceiling(save, tmp_path):
    quiet = saved_peak(save, tmp_path, [0.05, -0.05])
    loud = saved_peak(save, tmp_path, [0.5, -1.9, 0.3])
    assert loud == pytest.approx(0.95, abs=1e-3)
    assert quiet == pytest.approx(0.05, abs=1e-3)


def test_asking_for_full_level_still_raises_a_quiet_render(save, tmp_path):
    assert saved_peak(
        save, tmp_path, [0.1, -0.2, 0.1], normalize=True
    ) == pytest.approx(0.95, abs=1e-3)


class SteadySynth:
    def __init__(self, *args, **kwargs):
        pass

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

    def cc(self, *args):
        pass

    def note_on(self, *args):
        pass

    def note_off(self, *args):
        pass

    def get_samples(self, count):
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(count) / RATE)
        return tone, tone.copy()


def test_a_rendered_song_is_not_pushed_up_to_full_level(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_processors, "FluidSynth", SteadySynth)
    song = pretty_midi.PrettyMIDI()
    keys = pretty_midi.Instrument(program=0)
    keys.notes.append(pretty_midi.Note(100, 60, 0.0, 0.5))
    song.instruments.append(keys)
    midi = tmp_path / "song.mid"
    song.write(str(midi))
    out = audio_processors.render_midi_to_wav(
        str(midi), str(tmp_path / "song.wav"), "fluid", "font.sf3"
    )
    data, _rate = sf.read(out, dtype="float32")
    assert 0.1 < float(np.max(np.abs(data))) < 0.5
