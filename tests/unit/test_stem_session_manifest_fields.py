import json
import os

import numpy as np
import pytest
import soundfile as sf

import korshmer.audio_processing as audio_processing_package
from korshmer import patches
from korshmer.audio_processing import separation_engine, stem_session as ss

SR = 22050
STEM_NAMES = ("vocals", "drums", "bass", "other")


def make_session(tmp_path, **fields):
    return ss.StemSession(
        source_path=str(tmp_path / "song.wav"),
        output_dir=str(tmp_path / "out"),
        tempo=100.0,
        root="C",
        scale="Major",
        **fields,
    )


def test_manifest_fields_roundtrip(tmp_path):
    session = make_session(
        tmp_path,
        grid_offset=0.123,
        music_regions=[[1.0, 5.5], [8.0, 20.25]],
        separation="mdx+htdemucs_6s",
    )
    manifest = session.save()
    with open(manifest, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["grid_offset"] == 0.123
    assert payload["music_regions"] == [[1.0, 5.5], [8.0, 20.25]]
    assert payload["separation"] == "mdx+htdemucs_6s"
    loaded = ss.StemSession.load(manifest)
    assert loaded.grid_offset == 0.123
    assert loaded.music_regions == [[1.0, 5.5], [8.0, 20.25]]
    assert loaded.separation == "mdx+htdemucs_6s"


def test_new_fields_default_and_follow_the_existing_fields(tmp_path):
    session = make_session(tmp_path)
    assert session.grid_offset is None
    assert session.music_regions == []
    assert session.separation == ""
    names = list(ss.StemSession.__dataclass_fields__)
    assert names[-4:] == [
        "grid_offset",
        "music_regions",
        "separation",
        "music_cut",
    ]
    assert names.index("created") < names.index("grid_offset")
    loaded = ss.StemSession.load(session.save())
    assert loaded.grid_offset is None
    assert loaded.music_regions == []
    assert loaded.separation == ""


def test_old_manifest_without_the_new_fields_still_loads(tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    manifest = out / ss.MANIFEST_NAME
    manifest.write_text(
        json.dumps({"source_path": "a.wav", "tempo": 90, "stems": {}}),
        encoding="utf-8",
    )
    loaded = ss.StemSession.load(str(manifest))
    assert loaded.tempo == 90.0
    assert loaded.grid_offset is None
    assert loaded.music_regions == []
    assert loaded.separation == ""


@pytest.mark.parametrize(
    "grid_offset, regions, separation, expected_offset, expected_regions",
    [
        ("later", "everything", 7, None, []),
        (float("nan"), None, None, None, []),
        (None, [[1.0], [2.0, "x"], [5.0, 3.0], "ab", 4], "", None, []),
        (
            "0.25",
            [[1, 2], (3.5, 4.5), [float("inf"), 9], [-1.0, 2.0]],
            "mdx",
            0.25,
            [[1.0, 2.0], [3.5, 4.5]],
        ),
    ],
)
def test_wrong_types_in_the_manifest_are_tolerated(
    tmp_path,
    grid_offset,
    regions,
    separation,
    expected_offset,
    expected_regions,
):
    out = tmp_path / "bad"
    out.mkdir()
    manifest = out / ss.MANIFEST_NAME
    payload = {
        "source_path": "a.wav",
        "grid_offset": grid_offset,
        "music_regions": regions,
        "separation": separation,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    loaded = ss.StemSession.load(str(manifest))
    assert loaded.grid_offset == expected_offset
    assert loaded.music_regions == expected_regions
    assert isinstance(loaded.separation, str)


def test_numpy_values_are_saved_as_plain_json(tmp_path):
    session = make_session(
        tmp_path,
        grid_offset=np.float32(0.5),
        music_regions=[np.array([1.0, 2.0]), (np.float64(3.0), np.int64(4))],
    )
    manifest = session.save()
    with open(manifest, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["grid_offset"] == 0.5
    assert payload["music_regions"] == [[1.0, 2.0], [3.0, 4.0]]


def test_build_session_keeps_seven_positional_arguments(tmp_path):
    plain = ss.build_session_from_processing(
        "a.wav", str(tmp_path), 120, "C", "Major", {"vocals": "v.wav"}, []
    )
    assert plain.grid_offset is None
    assert plain.music_regions == []
    assert plain.separation == ""
    full = ss.build_session_from_processing(
        "a.wav",
        str(tmp_path),
        120,
        "C",
        "Major",
        {},
        [],
        grid_offset=np.float64(0.07),
        music_regions=[(0, 12.5)],
        separation="mdx+htdemucs_6s",
    )
    assert full.grid_offset == 0.07
    assert full.music_regions == [[0.0, 12.5]]
    assert full.separation == "mdx+htdemucs_6s"


def test_clean_vocals_cache_is_not_a_stem():
    assert ss.CLEAN_VOCALS_NAME == "vocals_clean.wav"
    assert "_clean" in ss.CLEAN_VOCALS_NAME


def write_stems(target):
    os.makedirs(target, exist_ok=True)
    for name in STEM_NAMES:
        data = np.full((SR // 2, 2), 0.1, dtype=np.float32)
        sf.write(os.path.join(target, f"{name}.wav"), data, SR)


@pytest.fixture
def separation_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(patches, "_apply_demucs_patches", lambda: None)
    monkeypatch.setattr(patches, "_apply_torchaudio_patches", lambda: None)
    monkeypatch.setattr(
        audio_processing_package.core_analysis, "analyse_tempo", lambda _p: None
    )
    monkeypatch.setattr(ss, "detect_session_key", lambda _p: ("C", "Major"))
    song = tmp_path / "Tune.wav"
    sf.write(str(song), np.zeros((SR // 2, 2), dtype=np.float32), SR)
    return str(song)


def test_separate_stems_without_options_keeps_the_plain_demucs_call(
    separation_env, monkeypatch, tmp_path
):
    calls = []

    def fake_job(*args, **kwargs):
        calls.append((args, kwargs))
        write_stems(os.path.join(args[1], args[2], "Tune"))

    monkeypatch.setattr(ss, "_run_demucs_job", fake_job)
    session = ss.separate_stems(separation_env, output_root=tmp_path / "out")
    ((args, kwargs),) = calls
    assert len(args) == 3 and args[2] == "htdemucs_6s"
    assert kwargs == {}
    assert session.separation == ""


def test_separate_stems_with_options_runs_the_engine(
    separation_env, monkeypatch, tmp_path
):
    seen = {}

    def fake_engine(audio, job_dir, options, progress, should_stop):
        seen["options"] = options
        seen["stop"] = should_stop()
        progress(0.0, "Reading the audio...")
        progress(0.5, "Separating the vocals...")
        target = os.path.join(job_dir, "htdemucs_6s", "Tune")
        write_stems(target)
        progress(1.0, "Stems ready")
        return target, "mdx+htdemucs_6s"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the engine replaces the plain Demucs call")

    monkeypatch.setattr(separation_engine, "separate_with_label", fake_engine)
    monkeypatch.setattr(
        audio_processing_package.core_analysis, "run_demucs", forbidden
    )
    options = separation_engine.SeparationOptions(quality="best")
    events = []
    session = ss.separate_stems(
        separation_env,
        output_root=tmp_path / "out",
        progress_callback=lambda percent, text: events.append(percent),
        check_stop_callback=lambda: False,
        options=options,
    )
    assert seen == {"options": options, "stop": False}
    assert session.separation == "mdx+htdemucs_6s"
    assert set(session.stems) == set(STEM_NAMES)
    loaded = ss.StemSession.find(str(tmp_path / "out"))
    assert loaded.separation == "mdx+htdemucs_6s"
    assert events == sorted(events)
    assert events[0] == 5 and events[-1] == 100
    assert {15, 50, 85} <= set(events)


def test_options_are_ignored_for_other_models(
    separation_env, monkeypatch, tmp_path
):
    ran = []

    def fake_run(audio, output_dir, model):
        ran.append(model)
        write_stems(os.path.join(output_dir, model, "Tune"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the engine only handles htdemucs_6s")

    monkeypatch.setattr(
        audio_processing_package.core_analysis, "run_demucs", fake_run
    )
    monkeypatch.setattr(separation_engine, "separate_with_label", forbidden)
    session = ss.separate_stems(
        separation_env,
        output_root=tmp_path / "out",
        model_name="htdemucs_ft",
        options=separation_engine.SeparationOptions(),
    )
    assert ran == ["htdemucs_ft"]
    assert session.separation == ""


def test_engine_failures_keep_the_error_contract(
    separation_env, monkeypatch, tmp_path
):
    options = separation_engine.SeparationOptions()

    def failing(*_args, **_kwargs):
        raise ValueError("bad tensor")

    monkeypatch.setattr(separation_engine, "separate_with_label", failing)
    with pytest.raises(RuntimeError, match="Demucs failed: bad tensor"):
        ss.separate_stems(
            separation_env, output_root=tmp_path / "a", options=options
        )

    def already_prefixed(*_args, **_kwargs):
        raise RuntimeError("Demucs failed: exit code 1")

    monkeypatch.setattr(
        separation_engine, "separate_with_label", already_prefixed
    )
    with pytest.raises(RuntimeError, match="^Demucs failed: exit code 1$"):
        ss.separate_stems(
            separation_env, output_root=tmp_path / "d", options=options
        )

    def missing(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(separation_engine, "separate_with_label", missing)
    with pytest.raises(RuntimeError, match="torch"):
        ss.separate_stems(
            separation_env, output_root=tmp_path / "b", options=options
        )

    def cancelled(*_args, **_kwargs):
        raise InterruptedError("Stopped by user")

    monkeypatch.setattr(separation_engine, "separate_with_label", cancelled)
    with pytest.raises(InterruptedError):
        ss.separate_stems(
            separation_env, output_root=tmp_path / "c", options=options
        )


def test_music_cut_roundtrips_and_old_manifests_load_without_it(tmp_path):
    cut = {"timeline": [[1.0, 5.5, 0.0], [8.0, 20.25, 4.5]], "source": "m.wav"}
    session = make_session(tmp_path, music_cut=cut)
    manifest = session.save()
    assert ss.StemSession.load(manifest).music_cut == cut
    assert make_session(tmp_path).music_cut is None
    plain = ss.StemSession.load(make_session(tmp_path).save())
    assert plain.music_cut is None
    assert plain.cut_source_path() is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        "text",
        {"timeline": [], "source": "m.wav"},
        {"timeline": [[1.0, 2.0, 0.0]], "source": ""},
        {"timeline": [[1.0, 2.0, 0.0]], "source": "../m.wav"},
        {"timeline": [[1.0, 2.0, 0.0]], "source": "sub/m.wav"},
        {
            "timeline": [[2.0, 1.0, 0.0], ["a", 1, 2], [1.0, 2.0, -1.0]],
            "source": "m.wav",
        },
    ],
)
def test_music_cut_garbage_is_dropped(value):
    assert ss._clean_music_cut(value) is None


def test_cut_source_path_needs_the_file_next_to_the_manifest(tmp_path):
    cut = {"timeline": [[0.0, 1.0, 0.0]], "source": "m.wav"}
    session = make_session(tmp_path, music_cut=cut)
    assert session.cut_source_path() is None
    os.makedirs(session.output_dir)
    sf.write(os.path.join(session.output_dir, "m.wav"), np.zeros(10), SR)
    assert session.cut_source_path() == os.path.join(
        session.output_dir, "m.wav"
    )
