import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_requirements.py"
WORKFLOW = ROOT / ".github" / "workflows" / "dependency-audit.yml"


@pytest.fixture
def export_requirements():
    pytest.importorskip("tomllib")
    spec = importlib.util.spec_from_file_location("export_requirements", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_requirements_skip_direct_references(
    export_requirements, tmp_path
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "demo"\n'
        "dependencies = [\n"
        '    "numpy>=1.24.0",\n'
        "    \"basic-pitch @ git+https://example.com/basic-pitch.git ; python_version < '3.12'\",\n"
        '    "keyring>=24.0.0",\n'
        "]\n",
        encoding="utf-8",
    )
    assert export_requirements.runtime_requirements(pyproject) == [
        "numpy>=1.24.0",
        "keyring>=24.0.0",
    ]


def test_project_requirements_are_exportable(export_requirements):
    lines = export_requirements.runtime_requirements(ROOT / "pyproject.toml")
    assert any(line.startswith("numpy") for line in lines)
    assert not any("@" in line for line in lines)


def test_main_prints_one_requirement_per_line(
    export_requirements, tmp_path, capsys
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\ndependencies = ["a>=1", "b>=2"]\n',
        encoding="utf-8",
    )
    assert export_requirements.main([str(pyproject)]) == 0
    assert capsys.readouterr().out.splitlines() == ["a>=1", "b>=2"]


def test_workflow_runs_pip_audit_on_schedule_and_changes():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = config.get("on", config.get(True))
    assert {"pull_request", "push", "schedule"} <= set(triggers)
    steps = config["jobs"]["audit"]["steps"]
    commands = " ".join(step.get("run", "") for step in steps)
    assert "pip-audit" in commands
    assert "scripts/export_requirements.py" in commands
