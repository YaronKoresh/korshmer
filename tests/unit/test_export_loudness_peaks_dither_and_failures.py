import os
import subprocess
import threading
import tracemalloc

import librosa
import numpy as np
import pyloudnorm
import pytest
import soundfile as sf
from scipy import signal

from korshmer.project_management import export_manager
from korshmer.project_management.export_manager import (
    ExportManager,
    validate_export_preset,
)

RATE = 44100


def _sine(freq, seconds, amplitude, rate=RATE, phase=0.0):
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def _write(tmp_path, name, audio, rate=RATE, subtype="DOUBLE"):
    path = tmp_path / name
    sf.write(str(path), audio, rate, subtype=subtype)
    return path


def _finish_exports():
    for thread in threading.enumerate():
        if thread.name == "korshmer-export":
            thread.join(60)


def _export(manager, source, output, preset, callback=None):
    finished = threading.Event()
    result = {}

    def done(job, success):
        result["job"] = job
        finished.set()
        if callback is not None:
            callback(job, success)

    assert manager.add_to_queue(source, output, preset, callback=done)
    manager.process_queue()
    assert finished.wait(60)
    _finish_exports()
    return result["job"]


def _export_with(manager, tmp_path, audio, settings, name="out.wav"):
    source = _write(tmp_path, "song.wav", audio)
    output = tmp_path / name
    manager.save_preset("Test", settings)
    job = _export(manager, source, output, "Test")
    return job, output


def _one_shot_true_peak(audio):
    return np.max(np.abs(signal.resample_poly(audio, 4, 1, axis=0)))


def _true_peak_db(audio):
    return 20 * np.log10(_one_shot_true_peak(audio))


@pytest.fixture
def manager(tmp_path):
    return ExportManager(tmp_path)


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    calls = []

    def run(cmd, **options):
        calls.append((cmd, options))
        with open(cmd[-1].removeprefix("file:"), "wb") as handle:
            handle.write(b"encoded")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(
        export_manager.shutil, "which", lambda name: "/opt/tools/ffmpeg"
    )
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_a_failed_sample_rate_conversion_fails_the_export(
    manager, tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise ValueError("resampler exploded")

    monkeypatch.setattr(librosa, "resample", broken)

    job, output = _export_with(
        manager,
        tmp_path,
        _sine(440, 1, 0.5),
        {"format": "wav", "sample_rate": 48000, "bit_depth": 24},
    )

    assert job["status"] == "failed"
    assert not output.exists()
    assert "44100" in job["error"] and "48000" in job["error"]


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({"format": "mp3", "bitrate": 192}, "out.mp3"),
        ({"format": "m4a", "bitrate": 192}, "out.m4a"),
    ],
)
def test_a_compressed_export_without_ffmpeg_fails_and_names_ffmpeg(
    manager, tmp_path, monkeypatch, settings, name
):
    monkeypatch.setenv("PATH", str(tmp_path))

    job, output = _export_with(
        manager, tmp_path, _sine(440, 1, 0.5), settings, name
    )

    assert job["status"] == "failed"
    assert not output.exists()
    assert "ffmpeg" in job["error"]


def test_an_export_that_wrote_no_file_is_never_reported_as_completed(
    manager, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        export_manager.shutil, "which", lambda name: "/opt/tools/ffmpeg"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""),
    )

    job, output = _export_with(
        manager, tmp_path, _sine(440, 1, 0.5), {"format": "mp3"}, "out.mp3"
    )

    assert job["status"] == "failed"
    assert str(output) in job["error"]


def test_ffmpeg_is_run_by_full_path_without_a_terminal_on_plain_file_names(
    manager, tmp_path, fake_ffmpeg, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source = _write(tmp_path, "song.wav", _sine(440, 1, 0.5))
    manager.save_preset("Test", {"format": "mp3", "bitrate": 192})

    job = _export(manager, source, "-odd name.mp3", "Test")

    cmd, options = fake_ffmpeg[-1]
    assert job["status"] == "completed"
    assert cmd[0] == "/opt/tools/ffmpeg"
    assert "-nostdin" in cmd
    assert cmd[cmd.index("-v") + 1] == "error"
    source_arg = cmd[cmd.index("-i") + 1]
    assert source_arg.startswith("file:") and os.path.isabs(source_arg[5:])
    assert cmd[-1] == "file:" + str(tmp_path / "-odd name.mp3")
    assert options["timeout"] > 0


def test_an_ffmpeg_that_hangs_fails_the_export_with_a_clear_message(
    manager, tmp_path, monkeypatch
):
    def hang(cmd, **options):
        raise subprocess.TimeoutExpired(cmd, options["timeout"])

    monkeypatch.setattr(
        export_manager.shutil, "which", lambda name: "/opt/tools/ffmpeg"
    )
    monkeypatch.setattr(subprocess, "run", hang)

    job, _ = _export_with(
        manager, tmp_path, _sine(440, 1, 0.5), {"format": "mp3"}, "out.mp3"
    )

    assert job["status"] == "failed"
    assert "ffmpeg" in job["error"] and "did not finish" in job["error"]
    assert not manager.get_queue_status()["is_exporting"]


def test_a_listener_that_crashes_does_not_leave_the_exporter_stuck(
    manager, tmp_path
):
    def crash(job, success):
        raise RuntimeError("the window is gone")

    source = _write(tmp_path, "song.wav", _sine(440, 1, 0.5))
    manager.save_preset("Test", {"format": "wav", "bit_depth": 24})

    _export(manager, source, tmp_path / "out.wav", "Test", callback=crash)

    assert not manager.get_queue_status()["is_exporting"]


def test_a_minus_14_lufs_export_really_measures_minus_14_lufs(
    manager, tmp_path
):
    loud = np.column_stack([_sine(3000, 4, 0.05), _sine(3100, 4, 0.05)])
    audio = np.concatenate([loud, np.zeros_like(loud)])

    job, output = _export_with(
        manager,
        tmp_path,
        audio,
        {
            "format": "wav",
            "bit_depth": 32,
            "normalize": "lufs",
            "target_lufs": -14,
        },
    )

    exported, rate = sf.read(str(output))
    measured = pyloudnorm.Meter(rate).integrated_loudness(exported)
    assert job["status"] == "completed"
    assert measured == pytest.approx(-14.0, abs=0.1)


@pytest.mark.parametrize("target", [1, -100, "loud", None, True, float("nan")])
def test_an_impossible_loudness_target_is_refused_when_the_preset_is_saved(
    target,
):
    with pytest.raises(ValueError, match="target_lufs"):
        validate_export_preset({"normalize": "lufs", "target_lufs": target})


def test_gain_that_would_pass_full_scale_is_turned_down_not_flat_topped(
    manager, tmp_path
):
    audio = _sine(997, 2, 0.1)

    job, output = _export_with(
        manager,
        tmp_path,
        audio,
        {
            "format": "wav",
            "bit_depth": 32,
            "normalize": "rms",
            "target_rms": -1.0,
        },
    )

    exported, _ = sf.read(str(output))
    scale = np.max(np.abs(exported)) / np.max(np.abs(audio))
    assert job["status"] == "completed"
    assert _true_peak_db(exported) <= -1.0 + 0.05
    assert np.allclose(exported, audio * scale, atol=1e-5)


def test_peaks_between_the_samples_are_kept_under_the_ceiling(
    manager, tmp_path
):
    audio = _sine(RATE / 4, 2, 0.1, phase=np.pi / 4)

    _, output = _export_with(
        manager,
        tmp_path,
        audio,
        {
            "format": "wav",
            "bit_depth": 32,
            "normalize": "rms",
            "target_rms": -2.0,
        },
    )

    exported, _ = sf.read(str(output))
    middle = exported[RATE // 2 : -RATE // 2]
    assert np.max(np.abs(middle)) <= 10 ** (-1.0 / 20) / np.sqrt(2) * 1.01


def test_the_true_peak_ceiling_follows_the_export_settings(manager, tmp_path):
    _, output = _export_with(
        manager,
        tmp_path,
        _sine(997, 2, 0.1),
        {
            "format": "wav",
            "bit_depth": 32,
            "normalize": "rms",
            "target_rms": -1.0,
            "true_peak_ceiling": -6.0,
        },
    )

    exported, _ = sf.read(str(output))
    assert _true_peak_db(exported) == pytest.approx(-6.0, abs=0.05)


@pytest.mark.parametrize("ceiling", [0.5, -200, "loud", True, None])
def test_an_impossible_true_peak_ceiling_is_refused(ceiling):
    with pytest.raises(ValueError, match="true_peak_ceiling"):
        validate_export_preset(
            {"normalize": "lufs", "true_peak_ceiling": ceiling}
        )


def test_true_peak_of_a_long_song_matches_measuring_it_in_one_go(monkeypatch):
    monkeypatch.setattr(export_manager, "EXPORT_TRUE_PEAK_CHUNK_FRAMES", 1000)
    rng = np.random.default_rng(4)
    audio = rng.uniform(-0.5, 0.5, (10500, 2))

    assert export_manager.true_peak(audio) == pytest.approx(
        _one_shot_true_peak(audio), rel=1e-9
    )


def test_true_peak_finds_a_peak_that_sits_between_two_samples():
    audio = _sine(RATE / 4, 1, 0.5, phase=np.pi / 4)

    assert export_manager.true_peak(audio) == pytest.approx(0.5, rel=0.02)


def test_measuring_true_peak_of_a_long_song_needs_little_extra_memory():
    frames = 3_000_000
    audio = np.zeros((frames, 2), dtype="float32")
    audio[frames // 2] = 0.25

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        peak = export_manager.true_peak(audio)
        _, most = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak == pytest.approx(0.25, rel=0.01)
    assert most - before < audio.nbytes


def _whisper(bit_depth):
    return _sine(441, 1, 0.4 / 2 ** (bit_depth - 1))


def _export_whisper(manager, tmp_path, bit_depth, **settings):
    name = f"out_{len(list(tmp_path.glob('out_*')))}.wav"
    job, output = _export_with(
        manager,
        tmp_path,
        _whisper(bit_depth),
        {"format": "wav", "bit_depth": bit_depth, "normalize": "none"}
        | settings,
        name,
    )
    assert job["status"] == "completed"
    exported, _ = sf.read(str(output), dtype="int32")
    return exported >> (32 - bit_depth)


def test_sound_quieter_than_one_16_bit_step_keeps_its_shape_at_16_bit(
    manager, tmp_path
):
    exported = _export_whisper(manager, tmp_path, 16, dither_seed=3)

    assert exported.any()
    assert abs(float(np.mean(exported))) < 0.05
    assert np.corrcoef(exported, _whisper(16))[0, 1] > 0.2


def test_dither_can_be_switched_off_in_the_export_settings(manager, tmp_path):
    exported = _export_whisper(manager, tmp_path, 16, dither=False)

    assert not exported.any()


def test_24_bit_exports_are_only_dithered_on_request(manager, tmp_path):
    plain = _export_whisper(manager, tmp_path, 24)
    dithered = _export_whisper(manager, tmp_path, 24, dither=True)

    assert not plain.any()
    assert dithered.any() and np.max(np.abs(dithered)) <= 2


def test_dither_is_repeatable_under_a_seed(manager, tmp_path):
    first = _export_whisper(manager, tmp_path, 16, dither_seed=7)
    again = _export_whisper(manager, tmp_path, 16, dither_seed=7)
    other = _export_whisper(manager, tmp_path, 16, dither_seed=8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_dither_noise_is_triangular_and_two_steps_wide(manager, tmp_path):
    _, output = _export_with(
        manager,
        tmp_path,
        np.zeros(RATE * 2),
        {
            "format": "wav",
            "bit_depth": 16,
            "normalize": "none",
            "dither_seed": 1,
        },
    )

    exported, _ = sf.read(str(output), dtype="int16")
    share = [np.mean(exported == step) for step in (-1, 0, 1)]
    assert set(np.unique(exported)) <= {-1, 0, 1}
    assert share == pytest.approx([0.125, 0.75, 0.125], abs=0.02)


@pytest.mark.parametrize(
    "settings", [{"dither": "yes"}, {"dither_seed": -1}, {"dither_seed": 1.5}]
)
def test_nonsense_dither_settings_are_refused(settings):
    with pytest.raises(ValueError, match="dither"):
        validate_export_preset(settings)


@pytest.mark.parametrize(
    ("preset", "setting", "value"),
    [
        ("Spotify", "true_peak_ceiling", -1.0),
        ("YouTube", "true_peak_ceiling", -1.0),
        ("YouTube", "dither", False),
        ("MP3 High", "true_peak_ceiling", -1.0),
        ("CD Quality", "dither_seed", None),
    ],
)
def test_the_built_in_presets_spell_out_their_finishing_settings(
    manager, preset, setting, value
):
    config = manager.export_presets[preset]

    assert setting in config and config[setting] == value


class TestBehaviourThatMustStayAsItWas:
    def test_audio_too_short_for_a_gated_measurement_is_still_levelled(
        self, manager, tmp_path
    ):
        job, output = _export_with(
            manager,
            tmp_path,
            _sine(1000, 0.2, 0.01),
            {"format": "wav", "bit_depth": 32, "normalize": "lufs"},
        )

        exported, _ = sf.read(str(output))
        rms_db = 20 * np.log10(np.sqrt(np.mean(exported**2)))
        assert job["status"] == "completed"
        assert rms_db - 0.691 == pytest.approx(-14.0, abs=0.1)

    def test_a_silent_song_stays_silent_when_levelled_to_a_loudness(
        self, manager, tmp_path
    ):
        job, output = _export_with(
            manager,
            tmp_path,
            np.zeros(RATE * 2),
            {"format": "wav", "bit_depth": 32, "normalize": "lufs"},
        )

        exported, _ = sf.read(str(output))
        assert job["status"] == "completed"
        assert not exported.any()

    def test_a_16_bit_song_exported_untouched_at_16_bit_stays_bit_identical(
        self, manager, tmp_path
    ):
        source = _write(
            tmp_path, "cd.wav", _sine(441, 1, 0.5), subtype="PCM_16"
        )
        output = tmp_path / "out.wav"
        manager.save_preset(
            "Test", {"format": "wav", "bit_depth": 16, "normalize": "none"}
        )

        _export(manager, source, output, "Test")

        original, _ = sf.read(str(source), dtype="int16")
        exported, _ = sf.read(str(output), dtype="int16")
        assert np.array_equal(exported, original)
