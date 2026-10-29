import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from korshmer.audio_processing import (
    core_analysis,
    mdx_vocals,
    separation_engine as engine,
)
from korshmer.audio_processing.separation_engine import (
    SeparationOptions,
    pick_vocal_backend,
    separate_with_label,
)
from korshmer.audio_processing.stem_session import _locate_demucs_output

RATE = 44100
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
TRACK = "Song"


def separate_to_folder(*args, **kwargs):
    return separate_with_label(*args, **kwargs)[0]


def make_source(path, channels=2, seconds=1.0):
    t = np.arange(int(seconds * RATE)) / RATE
    left = 0.4 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    data = np.stack([left, right], axis=1)[:, :channels]
    sf.write(path, data, RATE, subtype="PCM_24")
    return path


class Recorder:
    def __init__(self, tmp_path):
        self.runs = []
        self.vocal_calls = []
        self.mdx_calls = 0
        self.model = tmp_path / "model.onnx"
        self.model.write_bytes(b"model")

    def ensure_model(self, *_args, **kwargs):
        self.mdx_calls += 1
        progress = kwargs.get("progress")
        if progress is not None:
            progress(1.0, "Downloading the vocal model...")
        return self.model

    def separate_vocals(
        self, mix, rate, model_path, params=None, progress=None, **kwargs
    ):
        self.vocal_calls.append((mix.copy(), rate, model_path, params))
        if progress is not None:
            progress(0.5, "half")
            progress(1.0, "done")
        return (mix * 0.25).astype(np.float32)

    def run_demucs(
        self, audio_file, output_dir, model_name="htdemucs_6s", **kwargs
    ):
        data, rate = sf.read(audio_file, dtype="float32", always_2d=True)
        info = sf.info(audio_file)
        self.runs.append(
            {
                "file": audio_file,
                "dir": output_dir,
                "model": model_name,
                "kwargs": kwargs,
                "data": data,
                "rate": rate,
                "subtype": info.subtype,
                "channels": info.channels,
            }
        )
        track = os.path.splitext(os.path.basename(audio_file))[0]
        target = os.path.join(output_dir, model_name, track)
        os.makedirs(target, exist_ok=True)
        for index, name in enumerate(STEMS):
            tone = 0.1 * np.sin(
                2 * np.pi * 100 * (index + 1) * np.arange(2000) / RATE
            )
            sf.write(
                os.path.join(target, f"{name}.wav"),
                np.stack([tone, tone], axis=1),
                RATE,
                subtype="PCM_24",
            )


@pytest.fixture
def env(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path)
    monkeypatch.setattr(mdx_vocals, "ensure_mdx_model", recorder.ensure_model)
    monkeypatch.setattr(mdx_vocals, "separate_vocals", recorder.separate_vocals)
    monkeypatch.setattr(core_analysis, "run_demucs", recorder.run_demucs)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    source = make_source(str(tmp_path / f"{TRACK}.wav"))
    return types.SimpleNamespace(
        rec=recorder, job_dir=str(job_dir), source=source, tmp=tmp_path
    )


def decoded(path):
    data, _rate = sf.read(path, dtype="float32", always_2d=True)
    return data.T


@pytest.mark.parametrize(
    "quality, demucs_shifts, demucs_overlap, mdx_overlap",
    [("balanced", 1, 0.25, 0.25), ("best", 2, 0.25, 0.5)],
)
def test_balanced_uses_mdx_vocals_and_accompaniment_pass(
    env, quality, demucs_shifts, demucs_overlap, mdx_overlap
):
    stems_dir, label = separate_with_label(
        env.source, env.job_dir, SeparationOptions(quality=quality)
    )
    mix = decoded(env.source)
    ((heard, rate, model, params),) = env.rec.vocal_calls
    assert heard.shape == mix.shape and heard.dtype == np.float32
    assert np.allclose(heard, mix, atol=1e-6)
    assert rate == RATE
    assert str(model) == str(env.rec.model)
    assert params.overlap == mdx_overlap
    (run,) = env.rec.runs
    assert run["file"] == os.path.join(
        env.job_dir, "_accompaniment", f"{TRACK}.wav"
    )
    assert run["model"] == "htdemucs_6s"
    assert run["kwargs"] == {"shifts": demucs_shifts, "overlap": demucs_overlap}
    assert run["channels"] == 2 and run["rate"] == RATE
    assert run["subtype"] == "FLOAT"
    assert np.allclose(run["data"].T, mix * 0.75, atol=1e-6)
    written = sf.info(os.path.join(stems_dir, "vocals.wav"))
    assert written.subtype == "PCM_24" and written.samplerate == RATE
    assert np.allclose(
        decoded(os.path.join(stems_dir, "vocals.wav")),
        mix * 0.25,
        atol=1e-5,
    )
    assert label == "mdx+htdemucs_6s"


@pytest.mark.parametrize(
    "quality, specialist_call",
    [("balanced", (1, 0.25)), ("best", (2, 0.25))],
)
def test_missing_model_falls_back_to_demucs_ft_then_legacy(
    env, monkeypatch, quality, specialist_call
):
    monkeypatch.setattr(mdx_vocals, "ensure_mdx_model", lambda **_kw: None)
    ft_calls = []

    def specialist(mix, shifts, overlap, _should_stop):
        ft_calls.append((shifts, overlap))
        return (mix * 0.5).astype(np.float32)

    monkeypatch.setattr(engine, "_specialist_vocals", specialist)
    texts = []
    stems_dir, label = separate_with_label(
        env.source,
        env.job_dir,
        SeparationOptions(quality=quality),
        progress=lambda fraction, text: texts.append(text),
    )
    mix = decoded(env.source)
    assert ft_calls == [specialist_call]
    assert env.rec.vocal_calls == []
    assert any("unavailable" in text for text in texts)
    assert label == "htdemucs_ft+htdemucs_6s"
    assert os.path.basename(env.rec.runs[-1]["file"]) == f"{TRACK}.wav"
    assert np.allclose(env.rec.runs[-1]["data"].T, mix * 0.5, atol=1e-6)
    assert np.allclose(
        decoded(os.path.join(stems_dir, "vocals.wav")), mix * 0.5, atol=1e-5
    )

    def broken(_mix, _shifts, _overlap, _should_stop):
        raise RuntimeError("no weights")

    monkeypatch.setattr(engine, "_specialist_vocals", broken)
    texts.clear()
    stems_dir, label = separate_with_label(
        env.source,
        env.job_dir,
        SeparationOptions(quality=quality),
        progress=lambda fraction, text: texts.append(text),
    )
    run = env.rec.runs[-1]
    assert label == "htdemucs_6s"
    assert run["file"] == os.path.join(env.job_dir, "_mix", f"{TRACK}.wav")
    assert np.allclose(run["data"].T, mix, atol=1e-6)
    assert any(text.startswith("Vocal model unavailable") for text in texts)
    assert any(
        text.startswith("Demucs vocal model unavailable") for text in texts
    )
    assert sf.info(os.path.join(stems_dir, "vocals.wav")).frames == 2000


def test_a_failing_mdx_stage_falls_back_without_hiding_stop_requests(
    env, monkeypatch
):
    def exploding(*_args, **_kwargs):
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr(mdx_vocals, "separate_vocals", exploding)
    monkeypatch.setattr(
        engine, "_specialist_vocals", lambda mix, *_a: mix * 0.5
    )
    _stems_dir, label = separate_with_label(env.source, env.job_dir)
    assert label == "htdemucs_ft+htdemucs_6s"

    def cancelled(*_args, **_kwargs):
        raise InterruptedError("Stopped by user")

    monkeypatch.setattr(mdx_vocals, "separate_vocals", cancelled)
    with pytest.raises(InterruptedError):
        separate_with_label(env.source, env.job_dir)


@pytest.mark.parametrize("channels", [1, 2])
def test_fast_quality_is_the_legacy_single_pass_on_stereo_input(
    env, monkeypatch, channels
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("fast quality must not load the vocal model")

    monkeypatch.setattr(mdx_vocals, "ensure_mdx_model", forbidden)
    source = make_source(str(env.tmp / "Mono.wav"), channels=channels)
    stems_dir, label = separate_with_label(
        source, env.job_dir, SeparationOptions(quality="fast")
    )
    (run,) = env.rec.runs
    assert run["file"] == os.path.join(env.job_dir, "_mix", "Mono.wav")
    assert run["channels"] == 2
    assert run["subtype"] == "FLOAT" and run["rate"] == RATE
    assert run["kwargs"] == {"shifts": 1, "overlap": 0.1}
    original = decoded(source)
    expected = original if channels == 2 else np.repeat(original, 2, axis=0)
    assert np.allclose(run["data"].T, expected, atol=1e-6)
    assert label == "htdemucs_6s"
    assert sf.info(os.path.join(stems_dir, "vocals.wav")).frames == 2000


def test_layout_matches_track_processor_expectation(env):
    stems_dir = separate_to_folder(env.source, env.job_dir)
    model_root = os.path.join(env.job_dir, "htdemucs_6s")
    assert stems_dir == os.path.join(model_root, TRACK)
    subdirs = [
        d
        for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ]
    assert subdirs == [TRACK]
    assert sorted(os.listdir(stems_dir)) == sorted(f"{n}.wav" for n in STEMS)
    assert _locate_demucs_output(env.job_dir, "htdemucs_6s", TRACK) == stems_dir
    os.makedirs(os.path.join(env.job_dir, "_accompaniment"))
    sf.write(
        os.path.join(env.job_dir, "_accompaniment", "a.wav"),
        np.zeros((10, 2)),
        RATE,
    )
    assert _locate_demucs_output(env.job_dir, "htdemucs_6s", TRACK) == stems_dir


def test_scratch_dirs_are_removed(env, monkeypatch):
    separate_to_folder(env.source, env.job_dir)
    assert sorted(os.listdir(env.job_dir)) == ["htdemucs_6s"]

    def failing(audio_file, *_args, **_kwargs):
        assert os.path.isfile(audio_file)
        raise RuntimeError("Demucs failed: boom")

    monkeypatch.setattr(core_analysis, "run_demucs", failing)
    with pytest.raises(RuntimeError, match="Demucs failed: boom"):
        separate_to_folder(env.source, env.job_dir)
    assert sorted(os.listdir(env.job_dir)) == ["htdemucs_6s"]

    monkeypatch.setattr(core_analysis, "run_demucs", lambda *_a, **_k: None)
    fresh = env.tmp / "fresh"
    fresh.mkdir()
    with pytest.raises(RuntimeError, match="wrote no stems"):
        separate_to_folder(env.source, str(fresh))
    assert os.listdir(fresh) == []


def test_stop_request_raises_and_cleans_up(env):
    calls = []

    def should_stop():
        calls.append(1)
        return len(calls) >= 2

    with pytest.raises(InterruptedError):
        separate_to_folder(env.source, env.job_dir, should_stop=should_stop)
    assert env.rec.runs == []
    assert os.listdir(env.job_dir) == []


def test_options_are_validated():
    assert SeparationOptions() == SeparationOptions("best", "auto")
    for bad in (
        {"quality": "ultra"},
        {"vocal_backend": "roformer"},
        {"quality": None},
    ):
        with pytest.raises(ValueError, match="got"):
            SeparationOptions(**bad)
    default = SeparationOptions()
    assert pick_vocal_backend(default, True) == "mdx"
    assert pick_vocal_backend(default, False) == "demucs_ft"
    fast = SeparationOptions(quality="fast")
    assert pick_vocal_backend(fast, True) == "demucs_6s"
    legacy = SeparationOptions(quality="best", vocal_backend="demucs_6s")
    assert pick_vocal_backend(legacy, True) == "demucs_6s"
    specialist = SeparationOptions(vocal_backend="demucs_ft")
    assert pick_vocal_backend(specialist, True) == "demucs_ft"
    assert (
        pick_vocal_backend(SeparationOptions(vocal_backend="mdx"), False)
        == "demucs_ft"
    )


@pytest.mark.parametrize(
    "options",
    [
        SeparationOptions(),
        SeparationOptions(quality="fast"),
        SeparationOptions(vocal_backend="demucs_ft"),
    ],
)
def test_progress_is_monotonic_and_ends_at_one(env, monkeypatch, options):
    monkeypatch.setattr(
        engine, "_specialist_vocals", lambda mix, *_a: mix * 0.5
    )
    seen = []
    separate_to_folder(
        env.source,
        env.job_dir,
        options,
        progress=lambda fraction, text: seen.append((fraction, text)),
    )
    fractions = [fraction for fraction, _text in seen]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(0.0 <= value <= 1.0 and text for value, text in seen)
    if options == SeparationOptions():
        assert any(0.05 < value < 0.45 for value in fractions)
        assert any(0.45 <= value < 1.0 for value in fractions)


def test_loud_vocals_are_scaled_below_full_scale(env, monkeypatch):
    monkeypatch.setattr(
        mdx_vocals,
        "separate_vocals",
        lambda mix, *_a, **_k: (mix * 3.0).astype(np.float32),
    )
    stems_dir = separate_to_folder(env.source, env.job_dir)
    vocals = decoded(os.path.join(stems_dir, "vocals.wav"))
    assert 0.9 < np.max(np.abs(vocals)) < 1.0


def test_specialist_uses_the_vocal_member_of_the_bag(monkeypatch):
    import torch

    seen = {}

    class Member:
        sources = ["drums", "bass", "other", "vocals"]

    members = [Member() for _ in range(4)]

    def get_model(name):
        seen["name"] = name
        return types.SimpleNamespace(models=members)

    def apply_model(model, wav, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs
        silent = torch.zeros_like(wav)
        return torch.stack([silent, silent, silent, wav], dim=1)

    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(
        sys.modules,
        "demucs.apply",
        types.SimpleNamespace(apply_model=apply_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "demucs.pretrained",
        types.SimpleNamespace(get_model=get_model),
    )
    mix = (0.3 * np.random.default_rng(1).standard_normal((2, 5000))).astype(
        np.float32
    ) + 0.05
    vocals = engine._specialist_vocals(mix, 2, 0.25, lambda: False)
    assert seen["name"] == "htdemucs_ft"
    assert seen["model"] is members[3]
    assert seen["kwargs"]["shifts"] == 2 and seen["kwargs"]["overlap"] == 0.25
    assert seen["kwargs"]["split"] is True
    assert np.allclose(vocals, mix, atol=1e-5)


def fake_demucs_package(monkeypatch, main):
    separate = types.SimpleNamespace(main=main)
    package = types.ModuleType("demucs")
    package.separate = separate
    monkeypatch.setitem(sys.modules, "demucs", package)
    monkeypatch.setitem(sys.modules, "demucs.separate", separate)
    return separate


def test_run_demucs_passes_the_quality_flags(monkeypatch):
    commands = []
    fake_demucs_package(monkeypatch, lambda args: commands.append(list(args)))
    core_analysis.run_demucs("in.wav", "out")
    core_analysis.run_demucs(
        "in.wav", "out", "htdemucs_6s", shifts=2, overlap=0.5
    )
    default, tuned = commands
    assert default[default.index("--shifts") + 1] == "1"
    assert default[default.index("--overlap") + 1] == "0.1"
    assert tuned[tuned.index("--shifts") + 1] == "2"
    assert tuned[tuned.index("--overlap") + 1] == "0.5"
    assert default[default.index("-n") + 1] == "htdemucs_6s"
    assert default[default.index("-o") + 1] == "out"
    assert "--int24" in default and default[-1] == "in.wav"


def test_run_demucs_turns_a_fatal_exit_into_a_runtime_error(monkeypatch):
    def fatal(_args):
        raise SystemExit(1)

    fake_demucs_package(monkeypatch, fatal)
    with pytest.raises(RuntimeError, match="Demucs failed: exit code 1"):
        core_analysis.run_demucs("in.wav", "out")


@pytest.mark.parametrize(
    "kwargs",
    [{"shifts": -1}, {"shifts": 1.5}, {"overlap": 1.0}, {"overlap": -0.1}],
)
def test_run_demucs_validates_its_options(kwargs):
    with pytest.raises(ValueError, match="got"):
        core_analysis.run_demucs("in.wav", "out", **kwargs)


def test_a_stop_request_reaches_the_demucs_vocal_model(monkeypatch):
    bag = types.SimpleNamespace(
        models=[None, None, None, types.SimpleNamespace(sources=["vocals"])]
    )

    def apply_model(_model, _wav, **kwargs):
        kwargs["callback"]({})
        raise AssertionError("the callback must stop the pass")

    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(
        sys.modules,
        "demucs.apply",
        types.SimpleNamespace(apply_model=apply_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "demucs.pretrained",
        types.SimpleNamespace(get_model=lambda _name: bag),
    )
    with pytest.raises(InterruptedError):
        engine._specialist_vocals(
            np.zeros((2, 100), dtype=np.float32), 1, 0.25, lambda: True
        )


def test_a_stop_request_during_the_model_download_is_not_a_fallback(
    env, monkeypatch
):
    def cancelled(**_kwargs):
        raise InterruptedError("Stopped by user")

    monkeypatch.setattr(mdx_vocals, "ensure_mdx_model", cancelled)
    monkeypatch.setattr(
        engine, "_specialist_vocals", lambda *_a: pytest.fail("fell back")
    )
    with pytest.raises(InterruptedError):
        separate_with_label(env.source, env.job_dir)


def test_a_demucs_without_callback_support_still_separates(monkeypatch):
    bag = types.SimpleNamespace(
        models=[None, None, None, types.SimpleNamespace(sources=["vocals"])]
    )

    def apply_model(
        _model, wav, shifts=1, split=True, overlap=0.25, device=None
    ):
        return wav[:, None]

    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(
        sys.modules,
        "demucs.apply",
        types.SimpleNamespace(apply_model=apply_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "demucs.pretrained",
        types.SimpleNamespace(get_model=lambda _name: bag),
    )
    mix = np.ones((2, 100), dtype=np.float32)
    vocals = engine._specialist_vocals(mix, 1, 0.25, lambda: False)
    assert vocals.shape == (2, 100)
    with pytest.raises(InterruptedError):
        engine._specialist_vocals(mix, 1, 0.25, lambda: True)
