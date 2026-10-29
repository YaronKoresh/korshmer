from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"


@pytest.fixture
def config():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_runs_on_pull_requests_pushes_and_schedule(config):
    triggers = config.get("on", config.get(True))
    assert {"pull_request", "push", "schedule"} <= set(triggers)


def test_analyzes_python_with_security_queries(config):
    steps = config["jobs"]["analyze"]["steps"]
    init = next(
        s
        for s in steps
        if s.get("uses", "").startswith("github/codeql-action/init@")
    )
    assert init["with"]["languages"] == "python"
    assert init["with"]["queries"] == "security-extended"
    assert any(
        s.get("uses", "").startswith("github/codeql-action/analyze@")
        for s in steps
    )


def test_job_can_upload_results_and_nothing_else_writes(config):
    assert config["permissions"] == {"contents": "read"}
    permissions = config["jobs"]["analyze"]["permissions"]
    assert permissions["security-events"] == "write"
    assert permissions["contents"] == "read"
