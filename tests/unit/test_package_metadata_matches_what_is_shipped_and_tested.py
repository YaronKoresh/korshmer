import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
PROJECT = PYPROJECT["project"]
EXTRAS = PROJECT["optional-dependencies"]
CI_PATH = ROOT / ".github" / "workflows" / "check-quality.yml"


def names(requirements):
    return {
        re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0].lower() for r in requirements
    }


def ci():
    assert CI_PATH.is_file(), "the CI workflow is missing"
    return yaml.safe_load(CI_PATH.read_text("utf-8"))


def test_the_licence_is_declared_as_mit_and_its_text_is_shipped():
    assert PROJECT["license"] == "MIT"
    assert PROJECT["license-files"] == ["LICENSE"]
    assert "MIT License" in (ROOT / "LICENSE").read_text("utf-8")


def test_the_build_tool_is_new_enough_for_the_licence_fields():
    assert "hatchling>=1.27" in PYPROJECT["build-system"]["requires"]


def test_every_python_version_the_package_claims_is_tested():
    claimed = {
        c.rsplit(" ", 1)[1]
        for c in PROJECT["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    }
    blocking = ci()["jobs"]["test"]
    tested = set(blocking["strategy"]["matrix"]["python-version"])
    assert claimed and claimed == tested


def test_the_package_is_described_as_beta_audio_software():
    classifiers = PROJECT["classifiers"]
    assert "Development Status :: 4 - Beta" in classifiers
    assert "Topic :: Multimedia :: Sound/Audio" in classifiers
    assert not any("License ::" in c for c in classifiers)
    assert PROJECT["keywords"]
    assert {"Changelog", "Documentation"} <= set(PROJECT["urls"])


def test_unused_libraries_are_not_installed():
    assert "pywavelets" not in names(PROJECT["dependencies"])
    for extra in EXTRAS.values():
        assert "pydantic" not in names(extra)


def test_build_tools_are_their_own_optional_group():
    assert {"nuitka", "pyinstaller", "briefcase"} <= names(EXTRAS["build"])
    assert {"pytest", "hypothesis"} <= names(EXTRAS["test"])
    assert {"ruff", "mypy", "vulture"} <= names(EXTRAS["lint"])
    assert not {"nuitka", "pyinstaller"} & names(EXTRAS["test"])


def test_the_dev_group_still_installs_everything():
    assert EXTRAS["dev"] == ["korshmer[test,lint,build]"]


def test_the_app_bundle_settings_point_at_no_missing_icon():
    app = PYPROJECT["tool"]["briefcase"]["app"]["korshmer"]
    icon = app.get("icon")
    assert icon is None or list(ROOT.glob(icon + ".*"))


def test_ci_has_read_only_access_and_cancels_superseded_runs():
    workflow = ci()
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True


def test_timing_benchmarks_cannot_block_a_change():
    jobs = ci()["jobs"]
    blocking = " ".join(
        str(step.get("run", "")) for step in jobs["test"]["steps"]
    )
    assert "tests/unit" in blocking
    assert "tests/integration" in blocking
    assert "tests/regression" in blocking
    assert "tests/performance" not in blocking
    benchmarks = jobs["benchmarks"]
    assert benchmarks["continue-on-error"] is True
    assert "tests/performance" in " ".join(
        str(step.get("run", "")) for step in benchmarks["steps"]
    )
