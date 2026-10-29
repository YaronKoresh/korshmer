import os

import numpy as np
import pretty_midi
import soundfile as sf

from korshmer import audio_processors
from korshmer.constants import HELP_TEXTS
from korshmer.ui.managers.keyboard_shortcuts import KeyboardShortcutManager


class TestShortcutDefaults:
    def test_default_shortcuts_have_no_conflicts(self):
        manager = KeyboardShortcutManager()
        assert manager.detect_conflicts() == []

    def test_editor_shortcuts_present(self):
        manager = KeyboardShortcutManager()
        for key in (
            "duplicate",
            "quantize",
            "humanize",
            "toggle_snap",
            "build_song",
            "octave_up",
            "octave_down",
        ):
            assert manager.get(key)


class TestGeneratorHelpTexts:
    def test_generator_help_keys_exist(self):
        for key in (
            "midi_generator",
            "generator_mode",
            "generator_minutes",
            "generator_roles",
            "seed",
            "stems_editor",
            "separate",
            "clip",
            "lane",
            "snap_grid",
        ):
            assert key in HELP_TEXTS
            assert HELP_TEXTS[key]

    def test_old_builder_help_keys_are_gone(self):
        for key in ("song_builder", "layer", "parts"):
            assert key not in HELP_TEXTS

    def test_engine_managed_option_help_keys_are_gone(self):
        for key in ("generator_match", "generator_texture", "structure"):
            assert key not in HELP_TEXTS


def _write_wav(path, seconds=0.5, sr=22050, value=0.25):
    data = np.full((int(sr * seconds), 2), value, dtype=np.float32)
    sf.write(path, data, sr)
    return str(path)


class TestRenderEditorMix:
    def test_without_clips_delegates_to_midi_render(
        self, monkeypatch, tmp_path
    ):
        calls = []

        def fake_render(midi_path, save_path, fluid_dir, sf3_path, **kwargs):
            calls.append((midi_path, save_path, kwargs.get("bpm")))
            return save_path

        monkeypatch.setattr(audio_processors, "render_midi_to_wav", fake_render)
        out = audio_processors.render_editor_mix(
            "in.mid", str(tmp_path / "out.wav"), "fluid", "sf3", bpm=95.0
        )
        assert out == str(tmp_path / "out.wav")
        assert calls == [("in.mid", str(tmp_path / "out.wav"), 95.0)]

    def test_muted_clips_are_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            audio_processors,
            "render_midi_to_wav",
            lambda *a, **k: a[1],
        )
        out = audio_processors.render_editor_mix(
            "in.mid",
            str(tmp_path / "out.wav"),
            "fluid",
            "sf3",
            clips=[{"path": "x.wav", "start": 0.0, "muted": True}],
        )
        assert out == str(tmp_path / "out.wav")

    def test_clips_only_mix_without_soundfont(self, monkeypatch, tmp_path):
        clip = _write_wav(tmp_path / "vocals.wav")
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        midi_path = str(tmp_path / "empty.mid")
        midi.write(midi_path)
        out_path = str(tmp_path / "mix.wav")

        def fail_render(*_args, **_kwargs):
            raise AssertionError("MIDI render must not run without SoundFont")

        monkeypatch.setattr(audio_processors, "render_midi_to_wav", fail_render)
        result = audio_processors.render_editor_mix(
            midi_path,
            out_path,
            "fluid",
            str(tmp_path / "missing.sf3"),
            clips=[{"path": clip, "start": 0.5, "gain": 1.0, "muted": False}],
        )
        assert result == out_path
        assert os.path.exists(out_path)
        data, sr = sf.read(out_path)
        assert len(data) / sr >= 0.95
        assert np.max(np.abs(data[: int(sr * 0.4)])) < 1e-6
        assert np.max(np.abs(data[int(sr * 0.6) : int(sr * 0.9)])) > 0.1
