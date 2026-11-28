import importlib.util
import re
import runpy
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

if not QApplication.instance():
    app = QApplication(sys.argv)

import korshmer
import korshmer.audio_processors as audio_processors
from korshmer import constants, download_models
from korshmer.mixins.actions import project_management as song_actions
from korshmer.mixins.actions.project_management import (
    ProjectManagementActionsMixin,
)
from korshmer.mixins.actions.stems import StemsActionsMixin
from korshmer.project_management.project_manager import ProjectManager
from korshmer.ui.widgets.arranger.timeline import ArrangerTimeline

REPOSITORY = Path(korshmer.__file__).resolve().parents[2]
BUILD_SCRIPT = REPOSITORY / "scripts" / "compile.bat"


def _build_script():
    return BUILD_SCRIPT.read_bytes().decode("ascii")


def _options(name):
    return re.findall(rf"--{name}=(\S+)", _build_script().replace("^", " "))


def test_the_windows_build_script_uses_windows_line_endings():
    raw = BUILD_SCRIPT.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 20


def test_the_build_compiles_the_real_program_entry_point():
    script = _build_script()
    lines = [line.strip() for line in script.splitlines()]
    target = lines[lines.index("if errorlevel 1 goto failed", 40) - 1]
    assert target == "src\\korshmer"
    assert (REPOSITORY / "src" / "korshmer" / "__main__.py").is_file()
    assert "--python-flag=-m" in script and "--mode=standalone" in script
    assert "korshmer.PY" not in script


def test_the_build_uses_the_installed_environment_and_stops_on_errors():
    script = _build_script()
    assert ".venv\\Scripts\\activate.bat" in script
    assert not re.search(r"^py\s", script, re.M)
    assert script.count("if errorlevel 1") >= 3
    assert "exit /b 1" in script and "exit /b 0" in script


def test_optional_folders_are_only_bundled_when_they_exist():
    script = _build_script()
    for folder in ("models", "fluidsynth", "assets\\icon.ico"):
        assert f'if exist "{folder}' in script
    assert "--include-raw-dir=fluidsynth=fluidsynth" in script


def test_the_build_version_comes_from_the_program_itself():
    script = _build_script()
    assert "korshmer.__version__" in script
    assert "--file-version=%APP_VERSION%" in script
    assert not re.search(r"--file-version=\d", script)


def test_every_bundled_library_is_installed():
    wanted = _options("include-package") + _options("include-package-data")
    assert "basic_pitch" in wanted and "demucs" in wanted
    missing = [
        name
        for name in wanted
        if importlib.util.find_spec(name.split(".")[0]) is None
    ]
    assert missing == []


def test_nothing_the_program_needs_is_left_out_of_the_build():
    left_out = _options("nofollow-import-to")
    assert len(left_out) > 40
    standard = [
        m for m in left_out if m.split(".")[0] in sys.stdlib_module_names
    ]
    assert standard == ["tkinter"]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY / "src").rglob("*.py")
    )
    used = [
        name
        for name in left_out
        if re.search(
            rf"^\s*(?:import|from)\s+{re.escape(name)}(?:[\s.,]|$)",
            source,
            re.M,
        )
    ]
    assert used == []
    inner = [
        m
        for m in left_out
        if m.split(".")[0] in ("torch", "scipy", "sklearn", "numpy")
    ]
    assert inner == [
        "sklearn.cluster.tests",
        "torch._dynamo",
        "torch._inductor",
        "torch._lazy",
        "torch._numpy",
        "torch.onnx",
        "torch.utils.benchmark",
        "torch.utils.tensorboard",
    ]


def test_a_compiled_program_looks_for_its_data_beside_itself():
    path = constants.__file__
    plain = runpy.run_path(path)
    compiled = runpy.run_path(path, init_globals={"__compiled__": object()})
    assert compiled["BASE_DIR"] == Path(sys.executable).parent.resolve()
    assert plain["BASE_DIR"] == Path(path).resolve().parents[2]


def test_the_transcription_model_is_copied_whether_it_is_a_file_or_a_folder(
    tmp_path,
):
    single = tmp_path / "model.onnx"
    single.write_bytes(b"weights")
    folder = tmp_path / "saved"
    (folder / "variables").mkdir(parents=True)
    (folder / "saved_model.pb").write_bytes(b"graph")
    target = tmp_path / "out"
    copied = download_models.copy_transcription_model(single, target)
    assert copied.read_bytes() == b"weights"
    copied = download_models.copy_transcription_model(folder, target)
    assert (copied / "saved_model.pb").is_file() and (
        copied / "variables"
    ).is_dir()
    download_models.copy_transcription_model(folder, target)
    with pytest.raises(FileNotFoundError):
        download_models.copy_transcription_model(tmp_path / "missing", target)


def _drums_over_a_held_chord(rate=44100, seconds=4):
    t = np.arange(rate * seconds) / rate
    chord = 0.02 * sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
    drums = np.zeros_like(chord)
    hit = np.exp(-np.arange(2000) / 300.0) * np.sin(
        2 * np.pi * 90.0 * np.arange(2000) / rate
    )
    for start in range(0, len(drums) - 2000, rate // 4):
        drums[start : start + 2000] += 0.5 * hit
    mono = (chord + drums).astype(np.float32)
    return np.stack([mono, mono], axis=1), drums


def test_drum_hits_keep_their_punch_when_the_mix_is_levelled():
    mix, drums = _drums_over_a_held_chord()
    levelled = audio_processors.level_render(mix.copy())
    peaks = np.argsort(np.abs(drums))[-2000:]
    quiet = np.abs(drums) < 1e-6
    before = np.abs(mix[peaks, 0]).mean() / np.abs(mix[quiet, 0]).mean()
    after = (
        np.abs(levelled[peaks, 0]).mean() / np.abs(levelled[quiet, 0]).mean()
    )
    assert 20 * np.log10(before / after) < 1.5
    assert np.max(np.abs(levelled)) <= 1.0


class _Studio(ProjectManagementActionsMixin, StemsActionsMixin):
    def __init__(self, base):
        self.project_manager = ProjectManager(base)
        self.arranger = ArrangerTimeline()
        self.current_session = None


def _studio_with_a_note(tmp_path):
    studio = _Studio(tmp_path)
    track = studio.arranger.model.add_track("Lead")
    studio.arranger.model.add_note(track.id, 60, 0.0, 0.5, 90)
    studio.arranger.piano_roll.load_model()
    return studio


def test_starting_a_new_song_can_be_cancelled_to_keep_unsaved_notes(tmp_path):
    studio = _studio_with_a_note(tmp_path)
    with (
        mock.patch.object(
            song_actions, "get_limited_text", return_value=("Next", True)
        ),
        mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Cancel
        ),
    ):
        studio.new_project_action()
    assert len(studio.arranger.model.notes) == 1
    assert studio.project_manager.current_project_path is None
    with (
        mock.patch.object(
            song_actions, "get_limited_text", return_value=("Next", True)
        ),
        mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Discard
        ),
    ):
        studio.new_project_action()
    assert len(studio.arranger.model.notes) == 0


def test_opening_another_song_asks_about_unsaved_notes_first(tmp_path):
    studio = _studio_with_a_note(tmp_path)
    with (
        mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Cancel
        ),
        mock.patch.object(QFileDialog, "getOpenFileName") as dialog,
    ):
        studio.load_project_action()
    dialog.assert_not_called()
    with (
        mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Discard
        ),
        mock.patch.object(
            QFileDialog, "getOpenFileName", return_value=("", "")
        ) as dialog,
    ):
        studio.load_project_action()
    dialog.assert_called_once()


def test_the_build_leaves_out_only_parts_the_program_never_loads():
    left_out = set(_options("nofollow-import-to"))
    for name in (
        "torch._dynamo",
        "torch._inductor",
        "torch.onnx",
        "numba.cuda",
        "numba.tests",
        "onnxruntime.tools",
        "basic_pitch.data",
        "pytest",
        "_pytest",
    ):
        assert name in left_out
    assert "*.tests" not in left_out
    assert "sklearn" not in _options("include-package")
    assert "sklearn.cluster" in _options("include-package")
    assert "librosa" not in _options("include-package-data")
    spectrum = REPOSITORY / "src" / "korshmer" / "visualization" / "spectrum.py"
    assert 'matplotlib.use("Agg")' in spectrum.read_text(encoding="utf-8")


def test_the_build_stops_torch_from_rewriting_compiled_functions():
    yaml = pytest.importorskip("yaml")
    config = REPOSITORY / "scripts" / "nuitka-package.config.yml"
    assert (
        "--user-package-configuration-file=scripts\\nuitka-package.config.yml"
        in _build_script()
    )
    entries = yaml.safe_load(config.read_text(encoding="utf-8"))
    rule = next(
        entry
        for entry in entries
        if entry["module-name"] == "torch.utils._contextlib"
    )
    replacements = rule["anti-bloat"][0]["replacements_plain"]
    spec = importlib.util.find_spec("torch")
    if spec is None:
        pytest.skip("torch is not installed")
    source = (Path(spec.origin).parent / "utils" / "_contextlib.py").read_text(
        encoding="utf-8"
    )
    assert replacements
    for old, new in replacements.items():
        assert "__code__ =" in old
        assert "__code__" not in new
    rewrites = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\.__code__\s*=(?!=)", line)
    ]
    assert all(line in replacements for line in rewrites)


def test_the_build_installs_the_nuitka_version_it_needs():
    script = _build_script()
    assert "python -m nuitka --version" not in script
    install = 'python -m pip install -e ".[build]"'
    assert install in script
    assert script.index(install) < script.index("python -m nuitka ^")
    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    pinned = re.search(r'"nuitka==(\d+)\.(\d+)\.(\d+)"', pyproject)
    assert pinned is not None
    assert tuple(int(part) for part in pinned.groups()) >= (4, 3, 5)
