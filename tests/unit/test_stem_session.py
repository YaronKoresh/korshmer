import importlib
import json
import os
import sys

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import korshmer.audio_processing as audio_processing_package
from korshmer.audio_processing import core_analysis, stem_session as ss
from korshmer.event_bus import Event, EventBus, subscribe
from korshmer.mixins.actions import stems as stems_actions
from korshmer.mixins.actions.stems import StemsActionsMixin
from korshmer.workers.stem_separation import StemSeparationThread

SR = 22050


def write_tone(path, seconds=0.5, sample_rate=SR, channels=1, level=0.25):
    frames = int(seconds * sample_rate)
    data = np.full((frames, channels), level, dtype=np.float32)
    if channels == 1:
        data = data[:, 0]
    sf.write(str(path), data, sample_rate)
    return str(path)


def make_session(tmp_path, with_files=True):
    out_dir = tmp_path / "song" / "2026_01_01"
    stems_dir = out_dir / "Stems"
    midi_dir = out_dir / "MIDI"
    stems = {}
    midi_files = []
    if with_files:
        stems_dir.mkdir(parents=True)
        midi_dir.mkdir(parents=True)
        for name in ("vocals", "drums", "bass"):
            stems[name] = write_tone(stems_dir / f"{name}.wav")
        pm = pretty_midi.PrettyMIDI(initial_tempo=100)
        inst = pretty_midi.Instrument(program=33, name="Bass")
        inst.notes.append(pretty_midi.Note(90, 40, 0.0, 0.5))
        pm.instruments.append(inst)
        midi_path = midi_dir / "song.mid"
        pm.write(str(midi_path))
        midi_files.append(str(midi_path))
    session = ss.StemSession(
        source_path=str(tmp_path / "song.wav"),
        output_dir=str(out_dir),
        tempo=98.0,
        root="A",
        scale="Minor",
        stems=stems,
        midi_files=midi_files,
    )
    return session


class TestStemSessionManifest:
    def test_save_load_roundtrip(self, tmp_path):
        session = make_session(tmp_path)
        manifest = session.save()
        assert os.path.basename(manifest) == ss.MANIFEST_NAME
        assert manifest == session.manifest_path()
        with open(manifest, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["tempo"] == 98.0
        assert set(payload["stems"]) == {"vocals", "drums", "bass"}
        loaded = ss.StemSession.load(manifest)
        assert loaded.tempo == 98.0
        assert loaded.root == "A"
        assert loaded.scale == "Minor"
        assert loaded.stems == session.stems
        assert loaded.midi_files == session.midi_files
        assert loaded.output_dir == session.output_dir
        assert loaded.vocals_path() == session.stems["vocals"]
        assert loaded.created == session.created

    def test_find_walks_parent_chain(self, tmp_path):
        session = make_session(tmp_path)
        session.save()
        from_root = ss.StemSession.find(session.output_dir)
        from_stems = ss.StemSession.find(
            os.path.join(session.output_dir, "Stems")
        )
        from_file = ss.StemSession.find(session.stems["drums"])
        assert from_root is not None
        assert from_stems is not None
        assert from_file is not None
        assert from_file.stems["drums"] == session.stems["drums"]
        deep = os.path.join(session.output_dir, "a", "b", "c", "d")
        os.makedirs(deep)
        assert ss.StemSession.find(deep) is None
        assert ss.StemSession.find(str(tmp_path / "elsewhere")) is None
        assert ss.StemSession.find("") is None

    def test_load_relocates_moved_stems(self, tmp_path):
        session = make_session(tmp_path)
        session.stems["vocals"] = "/moved/away/vocals.wav"
        manifest = session.save()
        loaded = ss.StemSession.load(manifest)
        expected = os.path.join(session.output_dir, "Stems", "vocals.wav")
        assert loaded.stems["vocals"] == expected
        assert loaded.vocals_path() == expected

    def test_load_tolerates_missing_fields(self, tmp_path):
        folder = tmp_path / "bare"
        folder.mkdir()
        manifest = folder / ss.MANIFEST_NAME
        manifest.write_text(json.dumps({"source_path": "x.wav"}))
        loaded = ss.StemSession.load(str(manifest))
        assert loaded.tempo == ss.DEFAULT_TEMPO
        assert loaded.root == "C"
        assert loaded.scale == "Major"
        assert loaded.stems == {}
        assert loaded.vocals_path() is None

    def test_ordered_stems_follows_stem_order(self, tmp_path):
        session = make_session(tmp_path, with_files=False)
        session.stems = {
            "other": "o",
            "zither": "z",
            "vocals": "v",
            "bass": "b",
        }
        assert [name for name, _ in session.ordered_stems()] == [
            "vocals",
            "bass",
            "other",
            "zither",
        ]

    def test_build_session_from_processing_normalizes(self, tmp_path):
        session = ss.build_session_from_processing(
            "song.wav",
            str(tmp_path),
            None,
            9,
            "Minor (Natural)",
            {"vocals": tmp_path / "v.wav"},
            [tmp_path / "m.mid"],
        )
        assert session.tempo == 120.0
        assert session.root == "A"
        assert session.scale == "Minor"
        assert session.stems == {"vocals": str(tmp_path / "v.wav")}
        assert session.midi_files == [str(tmp_path / "m.mid")]
        assert session.source_path == os.path.abspath("song.wav")


class TestStemHelpers:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("vocals.wav", "vocals"),
            ("Vocal_Take2.mp3", "vocals"),
            ("drums.wav", "drums"),
            ("bass.wav", "bass"),
            ("Guitar_01.wav", "guitar"),
            ("piano.wav", "piano"),
            ("other.wav", "other"),
            ("no_vocals.wav", "other"),
            ("temp_trimmed.wav", None),
            ("mix.wav", None),
        ],
    )
    def test_classify_stem_name(self, name, expected):
        assert ss.classify_stem_name(name) == expected

    def test_find_stem_files(self, tmp_path):
        nested = tmp_path / "htdemucs_6s" / "song"
        nested.mkdir(parents=True)
        for name in ("other", "vocals", "drums"):
            (nested / f"{name}.wav").write_bytes(b"")
        (nested / "bass.mp3").write_bytes(b"")
        (nested / "temp_trimmed.wav").write_bytes(b"")
        (nested / "notes.txt").write_bytes(b"")
        (tmp_path / "mixdown.wav").write_bytes(b"")
        found = ss.find_stem_files(str(tmp_path))
        assert list(found) == ["vocals", "drums", "bass", "other"]
        assert found["bass"].endswith("bass.mp3")
        assert all("trimmed" not in path for path in found.values())

    def test_find_stem_files_prefers_wav_and_handles_missing_dir(
        self, tmp_path
    ):
        (tmp_path / "bass.mp3").write_bytes(b"")
        (tmp_path / "bass.wav").write_bytes(b"")
        found = ss.find_stem_files(str(tmp_path))
        assert found["bass"].endswith("bass.wav")
        assert ss.find_stem_files(str(tmp_path / "missing")) == {}
        assert ss.find_stem_files("") == {}

    def test_collect_stems_converts_subtype(self, tmp_path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        write_tone(source_dir / "vocals.wav", channels=2)
        write_tone(source_dir / "drums.wav")
        (source_dir / "broken_bass.wav").write_bytes(b"not audio")
        out_dir = tmp_path / "Stems"
        collected = ss.collect_stems(str(source_dir), str(out_dir))
        assert set(collected) == {"vocals", "drums"}
        assert collected["vocals"] == str(out_dir / "vocals.wav")
        info = sf.info(collected["vocals"])
        assert info.subtype == "PCM_24"
        assert info.channels == 2
        assert info.samplerate == SR
        data, _ = sf.read(collected["drums"], dtype="float32")
        assert np.allclose(data, 0.25, atol=1e-4)
        collected32 = ss.collect_stems(
            str(source_dir), str(tmp_path / "Stems32"), float32=True
        )
        assert sf.info(collected32["drums"]).subtype == "FLOAT"

    def test_collect_stems_single_file_and_empty(self, tmp_path):
        single = write_tone(tmp_path / "piano.wav")
        collected = ss.collect_stems(single, str(tmp_path / "out"))
        assert collected == {"piano": str(tmp_path / "out" / "piano.wav")}
        assert ss.collect_stems(str(tmp_path / "nothing"), str(tmp_path)) == {}

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, 120.0),
            (0, 120.0),
            (-5, 120.0),
            ("abc", 120.0),
            (float("nan"), 120.0),
            (200, 100.0),
            (60, 120.0),
            (128, 128.0),
            (185, 185.0),
            (75, 75.0),
        ],
    )
    def test_normalize_detected_tempo(self, value, expected):
        assert ss.normalize_detected_tempo(value) == expected

    def test_normalize_names(self):
        assert ss.normalize_scale_name("Minor (Natural)") == "Minor"
        assert ss.normalize_scale_name("Dorian") == "Dorian"
        assert ss.normalize_scale_name("") == "Major"
        assert ss.normalize_root_name(14) == "D"
        assert ss.normalize_root_name("G#") == "G#"
        assert ss.normalize_root_name("H") == "C"

    def test_package_lazy_exports(self):
        assert audio_processing_package.StemSession is ss.StemSession
        assert audio_processing_package.separate_stems is ss.separate_stems
        assert (
            audio_processing_package.mix_render_with_clips
            is ss.mix_render_with_clips
        )
        assert audio_processing_package.find_stem_files is ss.find_stem_files


def install_fake_demucs(monkeypatch, stem_names, model_name="htdemucs_6s"):
    import korshmer.patches as patches

    calls = {"patched": 0, "run": []}

    def fake_patch():
        calls["patched"] += 1

    def fake_run_demucs(audio_file, output_dir, model=model_name):
        calls["run"].append((audio_file, output_dir, model))
        song = os.path.splitext(os.path.basename(audio_file))[0]
        target = os.path.join(output_dir, model, song)
        os.makedirs(target, exist_ok=True)
        for name in stem_names:
            write_tone(os.path.join(target, f"{name}.wav"), channels=2)

    monkeypatch.setattr(patches, "_apply_demucs_patches", fake_patch)
    monkeypatch.setattr(patches, "_apply_torchaudio_patches", fake_patch)
    monkeypatch.setattr(core_analysis, "run_demucs", fake_run_demucs)
    return calls


class TestSeparateStems:
    def test_separate_stems_with_fake_demucs(self, tmp_path, monkeypatch):
        import korshmer.analysis as analysis

        data_dir = tmp_path / "data"
        monkeypatch.setattr(ss, "DATA_DIR", data_dir)
        calls = install_fake_demucs(
            monkeypatch, ["vocals", "drums", "bass", "other"]
        )
        monkeypatch.setattr(core_analysis, "analyse_tempo", lambda _p: None)
        monkeypatch.setattr(
            analysis, "detect_key_and_mode", lambda _p: (9, "Minor (Natural)")
        )
        song = write_tone(tmp_path / "MySong.wav", seconds=1.0, channels=2)
        progress = []
        session = ss.separate_stems(
            song,
            progress_callback=lambda p, m: progress.append((p, m)),
            check_stop_callback=lambda: False,
        )
        assert calls["patched"] == 2
        assert calls["run"][0][2] == "htdemucs_6s"
        assert session.tempo == 120.0
        assert session.root == "A"
        assert session.scale == "Minor"
        assert set(session.stems) == {"vocals", "drums", "bass", "other"}
        assert session.vocals_path() is not None
        assert os.path.isfile(session.vocals_path())
        assert sf.info(session.vocals_path()).subtype == "PCM_24"
        assert os.path.isfile(session.manifest_path())
        assert session.midi_files == []
        assert session.output_dir.startswith(str(data_dir / "MySong"))
        assert not os.path.exists(data_dir / "_temp_work" / "MySong")
        percents = [p for p, _ in progress]
        assert percents == sorted(percents)
        for required in (5, 15, 85, 100):
            assert required in percents
        reloaded = ss.StemSession.find(session.output_dir)
        assert reloaded is not None
        assert reloaded.stems == session.stems

    def test_separate_stems_custom_root_and_float32(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
        install_fake_demucs(monkeypatch, ["vocals", "other"], "mdx_extra")
        monkeypatch.setattr(core_analysis, "analyse_tempo", lambda _p: 190)
        song = write_tone(tmp_path / "song.wav")
        out_root = tmp_path / "custom_out"
        session = ss.separate_stems(
            song,
            output_root=str(out_root),
            model_name="mdx_extra",
            float32=True,
        )
        assert session.output_dir == str(out_root)
        assert session.tempo == 95.0
        assert sf.info(session.stems["vocals"]).subtype == "FLOAT"
        assert os.path.dirname(session.stems["vocals"]) == str(
            out_root / "Stems"
        )

    def test_separate_stems_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ss.separate_stems(str(tmp_path / "nope.wav"))

    def test_separate_stems_stop_requested(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
        calls = install_fake_demucs(monkeypatch, ["vocals"])
        song = write_tone(tmp_path / "song.wav")
        with pytest.raises(InterruptedError):
            ss.separate_stems(song, check_stop_callback=lambda: True)
        assert calls["run"] == []
        assert not os.path.exists(tmp_path / "data" / "_temp_work" / "song")

    def test_separate_stems_without_demucs_raises_runtime_error(
        self, tmp_path, monkeypatch
    ):
        import korshmer.patches as patches

        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")

        def missing_demucs():
            raise ImportError("No module named 'demucs'")

        monkeypatch.setattr(patches, "_get_demucs_modules", missing_demucs)
        song = write_tone(tmp_path / "song.wav")
        with pytest.raises(RuntimeError, match="torch"):
            ss.separate_stems(song)
        assert not os.path.exists(tmp_path / "data" / "_temp_work" / "song")

    def test_separate_stems_import_error_inside_run_demucs(
        self, tmp_path, monkeypatch
    ):
        import korshmer.patches as patches

        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(patches, "_apply_demucs_patches", lambda: None)
        monkeypatch.setattr(patches, "_apply_torchaudio_patches", lambda: None)

        def run_without_torch(_audio, _out, _model):
            raise ModuleNotFoundError("No module named 'torch'")

        monkeypatch.setattr(core_analysis, "run_demucs", run_without_torch)
        song = write_tone(tmp_path / "song.wav")
        with pytest.raises(RuntimeError, match="demucs"):
            ss.separate_stems(song)

    def test_separate_stems_no_output_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
        install_fake_demucs(monkeypatch, [])
        song = write_tone(tmp_path / "song.wav")
        with pytest.raises(RuntimeError, match="no stems"):
            ss.separate_stems(song)

    def test_demucs_failure_is_wrapped(self, tmp_path, monkeypatch):
        import korshmer.patches as patches

        monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(patches, "_apply_demucs_patches", lambda: None)
        monkeypatch.setattr(patches, "_apply_torchaudio_patches", lambda: None)

        def exploding(_audio, _out, _model):
            raise ValueError("bad tensor")

        monkeypatch.setattr(core_analysis, "run_demucs", exploding)
        song = write_tone(tmp_path / "song.wav")
        with pytest.raises(RuntimeError, match="Demucs failed: bad tensor"):
            ss.separate_stems(song)


class TestMixRenderWithClips:
    def test_nothing_to_mix(self, tmp_path):
        out = tmp_path / "mix.wav"
        assert ss.mix_render_with_clips(None, [], str(out)) is None
        assert (
            ss.mix_render_with_clips(
                str(tmp_path / "missing.wav"),
                [{"path": str(tmp_path / "gone.wav"), "start": 0.0}],
                str(out),
            )
            is None
        )
        assert not out.exists()

    def test_offset_gain_mono_and_resample(self, tmp_path):
        render = write_tone(
            tmp_path / "render.wav",
            seconds=1.0,
            sample_rate=44100,
            channels=2,
            level=0.25,
        )
        clip = write_tone(
            tmp_path / "clip.wav",
            seconds=0.5,
            sample_rate=22050,
            channels=1,
            level=0.5,
        )
        out = ss.mix_render_with_clips(
            render,
            [{"path": clip, "start": 1.0, "gain": 0.5, "muted": False}],
            str(tmp_path / "out" / "mix.wav"),
            sample_rate=44100,
        )
        assert out == str(tmp_path / "out" / "mix.wav")
        info = sf.info(out)
        assert info.subtype == "FLOAT"
        assert info.channels == 2
        assert info.samplerate == 44100
        data, _ = sf.read(out, dtype="float32")
        assert abs(data.shape[0] - int(1.5 * 44100)) <= 2
        assert np.allclose(data[1000:43000], 0.25, atol=1e-3)
        tail = data[44100 + 2000 : -2000]
        assert np.allclose(tail[:, 0], 0.25, atol=0.02)
        assert np.allclose(tail[:, 1], 0.25, atol=0.02)

    def test_muted_missing_and_negative_start(self, tmp_path):
        clip = write_tone(
            tmp_path / "clip.wav", seconds=0.2, sample_rate=44100, level=0.3
        )
        out = ss.mix_render_with_clips(
            None,
            [
                {"path": clip, "start": -5.0, "gain": 1.0},
                {"path": clip, "start": 0.0, "muted": True},
                {"path": str(tmp_path / "missing.wav"), "start": 0.0},
                "not a clip",
            ],
            str(tmp_path / "mix.wav"),
            sample_rate=44100,
        )
        data, _ = sf.read(out, dtype="float32")
        assert data.shape == (int(0.2 * 44100), 2)
        assert np.allclose(data, 0.3, atol=1e-4)

    def test_peak_normalization_only_when_needed(self, tmp_path):
        loud = write_tone(
            tmp_path / "loud.wav", seconds=0.1, sample_rate=44100, level=0.9
        )
        quiet = write_tone(
            tmp_path / "quiet.wav", seconds=0.1, sample_rate=44100, level=0.2
        )
        clipped_out = ss.mix_render_with_clips(
            loud,
            [{"path": loud, "start": 0.0}],
            str(tmp_path / "clipped.wav"),
        )
        data, _ = sf.read(clipped_out, dtype="float32")
        ceiling = 10 ** (-1.0 / 20.0)
        assert abs(float(np.max(np.abs(data))) - ceiling) < 1e-3
        clean_out = ss.mix_render_with_clips(
            quiet,
            [{"path": quiet, "start": 0.0}],
            str(tmp_path / "clean.wav"),
            headroom_db=-1.0,
        )
        data, _ = sf.read(clean_out, dtype="float32")
        assert np.allclose(data, 0.4, atol=1e-4)

    def test_render_sample_rate_mismatch_is_resampled(self, tmp_path):
        render = write_tone(
            tmp_path / "render48.wav", seconds=0.5, sample_rate=48000, level=0.2
        )
        out = ss.mix_render_with_clips(
            render, [], str(tmp_path / "mix.wav"), sample_rate=44100
        )
        info = sf.info(out)
        assert info.samplerate == 44100
        assert abs(info.frames - int(0.5 * 44100)) <= 2


class TestStemSeparationThread:
    def test_success_path_maps_callbacks(self, tmp_path, monkeypatch):
        session = make_session(tmp_path, with_files=False)

        def fake_separate(path, **kwargs):
            kwargs["progress_callback"](50, "half")
            assert kwargs["check_stop_callback"]() is False
            assert kwargs["model_name"] == "htdemucs_6s"
            assert kwargs["float32"] is True
            session.source_path = path
            return session

        monkeypatch.setattr(ss, "separate_stems", fake_separate)
        thread = StemSeparationThread("song.wav", float32=True)
        events = []
        thread.progress_update.connect(lambda v, m: events.append((v, m)))
        thread.session_ready.connect(lambda s: events.append(("ready", s)))
        thread.separation_failed.connect(lambda m: events.append(("fail", m)))
        thread.finished_signal.connect(lambda s: events.append(("done", s)))
        thread.run()
        assert (50, "half") in events
        assert ("ready", session) in events
        assert ("done", session) in events
        assert not any(e[0] == "fail" for e in events)
        assert session.source_path == "song.wav"

    def test_cancel_and_failure_paths(self, monkeypatch):
        def cancelled(_path, **_kwargs):
            raise InterruptedError("Stopped by user")

        monkeypatch.setattr(ss, "separate_stems", cancelled)
        thread = StemSeparationThread("song.wav")
        failures = []
        errors = []
        thread.separation_failed.connect(failures.append)
        thread.error_signal.connect(errors.append)
        thread.run()
        assert failures == ["Cancelled"]
        assert errors == ["Cancelled"]

        def broken(_path, **_kwargs):
            raise RuntimeError("Demucs failed: boom")

        monkeypatch.setattr(ss, "separate_stems", broken)
        thread = StemSeparationThread("song.wav")
        failures = []
        ready = []
        thread.separation_failed.connect(failures.append)
        thread.session_ready.connect(ready.append)
        thread.run()
        assert failures == ["Demucs failed: boom"]
        assert ready == []

    def test_stop_suppresses_progress(self, monkeypatch):
        thread = StemSeparationThread("song.wav")
        received = []
        thread.progress_update.connect(lambda v, m: received.append(v))
        thread.stop()
        thread._emit_progress(10, "ignored")
        assert received == []
        assert thread._stop_requested() is True


class FakeArranger:
    def __init__(self):
        self.clips = []
        self.gains = []
        self.midis = []
        self.cleared = 0

    def add_audio_clip(
        self, path, start_sec=0.0, name=None, kind="audio", gain=1.0
    ):
        self.clips.append((path, start_sec, name, kind))
        self.gains.append(gain)

    def load_midi_file(self, path, replace=True, tempo=None, file_tempo=True):
        self.midis.append((path, replace, tempo))

    def clear_editor(self):
        self.cleared += 1


class FakeSpin:
    def __init__(self):
        self.value = None

    def minimum(self):
        return 20

    def maximum(self):
        return 300

    def setValue(self, value):
        self.value = value


class FakeTabs:
    def __init__(self):
        self.index = 3

    def setCurrentIndex(self, index):
        self.index = index


class FakeMessageBox:
    Yes = 1
    No = 2
    asked = []
    answer = 1

    @classmethod
    def question(cls, _parent, title, text, _buttons):
        cls.asked.append((title, text))
        return cls.answer

    @classmethod
    def warning(cls, _parent, title, text):
        cls.asked.append(("warning", title, text))

    @classmethod
    def critical(cls, _parent, title, text):
        cls.asked.append(("critical", title, text))


class Host(StemsActionsMixin):
    def __init__(self, selected=None):
        self.arranger = FakeArranger()
        self.spin_bpm = FakeSpin()
        self.tabs = FakeTabs()
        self.selected = selected

    def _get_selected_file(self, is_input=False, requires_midi=False):
        return self.selected


@pytest.fixture
def clean_bus():
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def fake_message_box(monkeypatch):
    FakeMessageBox.asked = []
    FakeMessageBox.answer = FakeMessageBox.Yes
    monkeypatch.setattr(stems_actions, "QMessageBox", FakeMessageBox)
    return FakeMessageBox


class TestCleanedVocals:
    def test_cleaned_file_is_made_once_and_source_is_untouched(self, tmp_path):
        session = make_session(tmp_path)
        source = session.stems["vocals"]
        before = open(source, "rb").read()
        first = ss.cleaned_vocals_path(session)
        assert first == os.path.join(
            session.output_dir, ss.CLEAN_VOCALS_DIR, ss.CLEAN_VOCALS_NAME
        )
        assert os.path.isfile(first)
        assert open(source, "rb").read() == before
        stamp = os.path.getmtime(first)
        assert ss.cleaned_vocals_path(session) == first
        assert os.path.getmtime(first) == stamp
        data, rate = sf.read(first)
        original, original_rate = sf.read(source)
        assert rate == original_rate
        assert data.shape == original.shape

    def test_source_newer_than_cleaned_file_is_cleaned_again(self, tmp_path):
        session = make_session(tmp_path)
        first = ss.cleaned_vocals_path(session)
        newer = os.path.getmtime(first) + 10
        os.utime(session.stems["vocals"], (newer, newer))
        write_tone(session.stems["vocals"], seconds=1.0)
        os.utime(session.stems["vocals"], (newer, newer))
        assert ss.cleaned_vocals_path(session) == first
        assert sf.info(first).duration == pytest.approx(1.0, abs=0.01)

    def test_no_vocals_gives_none(self, tmp_path):
        session = make_session(tmp_path, with_files=False)
        assert ss.cleaned_vocals_path(session) is None

    def test_unreadable_vocals_fall_back_to_the_stem(
        self, tmp_path, monkeypatch
    ):
        session = make_session(tmp_path)

        def broken(_path):
            raise RuntimeError("no decoder")

        monkeypatch.setattr(ss, "_read_audio_any", broken)
        assert ss.cleaned_vocals_path(session) == session.stems["vocals"]

    def test_cleaned_file_is_not_found_as_a_separate_stem(self, tmp_path):
        session = make_session(tmp_path)
        ss.cleaned_vocals_path(session)
        found = ss.find_stem_files(session.output_dir)
        assert found["vocals"] == session.stems["vocals"]


class TestStemsActionsMixin:
    def test_load_session_into_editor(self, tmp_path, clean_bus):
        statuses = []
        subscribe(Event.STATUS_UPDATE, lambda text="": statuses.append(text))
        session = make_session(tmp_path)
        session.midi_files.append(str(tmp_path / "missing.mid"))
        host = Host()
        assert host.load_session_into_editor(session, replace=True) is True
        assert host.arranger.cleared == 1
        assert host.arranger.clips == [
            (ss.cleaned_vocals_path(session), 0.0, "Vocals", "vocals")
        ]
        assert host.arranger.clips[0][0] != session.stems["vocals"]
        assert host.arranger.gains == [
            ss.vocal_gain_for_file(host.arranger.clips[0][0])
        ]
        assert host.arranger.midis == [(session.midi_files[0], False, 98.0)]
        assert host.spin_bpm.value == 98
        assert host.tabs.index == 0
        assert any("Editor loaded 2 items" in s for s in statuses)

    def test_session_tempo_keeps_its_fraction(self, tmp_path, clean_bus):
        session = make_session(tmp_path)
        session.tempo = 123.45
        host = Host()
        host._apply_session_tempo(session)
        assert host.spin_bpm.value == 123.45

    def test_load_session_without_editor_or_vocals(self, tmp_path, clean_bus):
        session = make_session(tmp_path, with_files=False)
        host = Host()
        assert host.load_session_into_editor(session) is False
        assert host.tabs.index == 3
        bare = StemsActionsMixin()
        assert bare.load_session_into_editor(session) is False
        assert bare.load_session_into_editor(None) is False

    def test_send_audio_to_editor(self, tmp_path, clean_bus):
        host = Host()
        drums = write_tone(tmp_path / "drums.wav")
        assert host.send_audio_to_editor(drums, start_sec=-2.0) is True
        assert host.arranger.clips == [(drums, 0.0, "drums", "drums")]
        assert host.tabs.index == 0
        assert host.send_audio_to_editor(str(tmp_path / "none.wav")) is False
        host.arranger = object()
        assert host.send_audio_to_editor(drums) is False

    def test_load_output_folder_session(self, tmp_path, clean_bus):
        session = make_session(tmp_path)
        session.save()
        host = Host()
        assert host.load_output_folder_session(session.output_dir) is True
        assert host.arranger.clips[0][3] == "vocals"
        assert host.load_output_folder_session(str(tmp_path / "x")) is False

    def test_on_stems_session_ready_publishes_and_loads(
        self, tmp_path, clean_bus, fake_message_box
    ):
        refreshed = []
        subscribe(
            Event.OUTPUT_VIEW_REFRESH,
            lambda output_dir=None: refreshed.append(output_dir),
        )
        stems_ready_event = getattr(Event, "STEMS_READY", None)
        ready = []
        if stems_ready_event is not None:
            subscribe(stems_ready_event, lambda session=None: ready.append(1))
        session = make_session(tmp_path)
        host = Host()
        host.on_stems_session_ready(session)
        assert refreshed == [session.output_dir]
        assert fake_message_box.asked[0][0] == "Stems Ready"
        assert host.arranger.clips[0][2] == "Vocals"
        if stems_ready_event is not None:
            assert ready == [1]
        fake_message_box.answer = fake_message_box.No
        host2 = Host()
        host2.on_stems_session_ready(session)
        assert host2.arranger.clips == []

    def test_failure_and_progress_handlers(self, clean_bus, fake_message_box):
        progress = []
        states = []
        subscribe(
            Event.PROGRESS_UPDATE,
            lambda value=0, text="": progress.append((value, text)),
        )
        subscribe(
            Event.PROGRESS_BAR_STATE,
            lambda visible=False, value=0: states.append(visible),
        )
        host = Host()
        host.on_stem_progress(42, "working")
        assert progress == [(42, "working")]
        host.on_stem_separation_failed("Cancelled")
        assert fake_message_box.asked == []
        host.on_stem_separation_failed("Demucs failed")
        assert fake_message_box.asked[-1][0] == "critical"
        assert states == [False, False]

    def test_run_stem_separation_validates_input(
        self, clean_bus, fake_message_box
    ):
        host = Host(selected=None)
        host.run_stem_separation()
        assert not hasattr(host, "stem_thread")
        host = Host(selected="/tmp/notes.mid")
        host.run_stem_separation()
        assert fake_message_box.asked[-1][0] == "warning"
        assert not hasattr(host, "stem_thread")

    def test_run_stem_separation_starts_thread(
        self, tmp_path, clean_bus, fake_message_box, monkeypatch
    ):
        import korshmer.workers.stem_separation as worker_module

        started = []

        class FakeThread(StemSeparationThread):
            def start(self, *_args):
                started.append(self.audio_path)

        monkeypatch.setattr(worker_module, "StemSeparationThread", FakeThread)
        states = []
        subscribe(
            Event.PROGRESS_BAR_STATE,
            lambda visible=False, value=0: states.append(visible),
        )
        song = write_tone(tmp_path / "song.wav")
        host = Host(selected=song)
        host.chk_wav32 = type("Chk", (), {"isChecked": lambda _s: True})()
        host.run_stem_separation()
        assert started == [song]
        assert host.stem_thread.float32 is True
        assert states == [True]
        host.stop_stem_separation()
        assert host.stem_thread.is_running is True

    def test_named_event_helper(self, clean_bus):
        received = []
        subscribe(Event.STATUS_UPDATE, lambda text="": received.append(text))
        assert stems_actions._publish_named_event("STATUS_UPDATE", text="hi")
        assert received == ["hi"]
        assert not stems_actions._publish_named_event("NOT_AN_EVENT", x=1)


class TestTrackProcessorStemExport:
    def test_process_track_exports_stems_and_manifest(
        self, tmp_path, monkeypatch
    ):
        import shutil

        import korshmer.track_processor as track_processor

        data_dir = tmp_path / "data"
        monkeypatch.setattr(track_processor, "DATA_DIR", data_dir)
        monkeypatch.setattr(track_processor, "analyse_tempo", lambda _p: None)
        monkeypatch.setattr(
            track_processor, "detect_key_and_mode", lambda _p: (7, "Major")
        )
        monkeypatch.setattr(
            track_processor, "_apply_demucs_patches", lambda: None
        )
        monkeypatch.setattr(
            track_processor, "_apply_torchaudio_patches", lambda: None
        )

        def fake_trim(path, job_dir):
            target = os.path.join(job_dir, "temp_trimmed.wav")
            shutil.copy2(path, target)
            return target

        monkeypatch.setattr(track_processor, "trim_audio_silence", fake_trim)

        def fake_run_demucs(_audio, output_dir):
            target = os.path.join(output_dir, "htdemucs_6s", "temp_trimmed")
            os.makedirs(target, exist_ok=True)
            for name in ("vocals", "drums", "bass", "other"):
                write_tone(os.path.join(target, f"{name}.wav"), channels=2)

        monkeypatch.setattr(track_processor, "run_demucs", fake_run_demucs)
        song = write_tone(tmp_path / "Tune.wav", seconds=1.0)
        progress = []
        generated = track_processor.process_track(
            song,
            "Original",
            "Auto",
            "Auto (Detect)",
            "1/16",
            selected_stems=["vocals"],
            export_wav_32bit=True,
            progress_callback=lambda v, desc="": progress.append((v, desc)),
        )
        assert len(generated) == 4
        assert all(p.endswith(".wav") for p in generated)
        assert all(os.path.isfile(p) for p in generated)
        assert all(os.sep + "Stems" + os.sep in p for p in generated)
        assert sf.info(generated[0]).subtype == "FLOAT"
        assert not any(p.endswith(ss.MANIFEST_NAME) for p in generated)
        output_root = os.path.dirname(os.path.dirname(generated[0]))
        session = ss.StemSession.find(output_root)
        assert session is not None
        assert session.tempo == 120.0
        assert session.root == "G"
        assert session.scale == "Major"
        assert set(session.stems) == {"vocals", "drums", "bass", "other"}
        assert session.midi_files == []
        assert progress[-1][0] == 1.0
        assert not os.path.exists(data_dir / "_temp_work" / "Tune")


class TestAnalysisImportWithoutTorch:
    def test_detect_midi_key_imports_without_torch(self, tmp_path, monkeypatch):
        for blocked in ("torch", "torchaudio", "demucs", "demucs.separate"):
            monkeypatch.setitem(sys.modules, blocked, None)
        for cached in (
            "analyse_tempo",
            "estimate_meter",
            "run_demucs",
            "trim_audio_silence",
            "pitch_shift_file",
        ):
            monkeypatch.delattr(audio_processing_package, cached, raising=False)
        for module_name in (
            "korshmer.analysis",
            "korshmer.audio_processing.core_analysis",
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        with pytest.raises(ImportError):
            importlib.import_module("torch")
        fresh_core = importlib.import_module(
            "korshmer.audio_processing.core_analysis"
        )
        fresh_analysis = importlib.import_module("korshmer.analysis")
        assert fresh_core is not core_analysis
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0)
        for pitch in (57, 60, 64, 67, 69, 72):
            inst.notes.append(pretty_midi.Note(100, pitch, 0.0, 1.0))
        pm.instruments.append(inst)
        midi_path = tmp_path / "key.mid"
        pm.write(str(midi_path))
        root, mode = fresh_analysis.detect_midi_key(str(midi_path))
        assert root in ss.NOTE_NAMES
        assert mode in ("Major", "Minor")
        with pytest.raises(ImportError):
            fresh_core.run_demucs("in.wav", str(tmp_path))
