import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dependabot_update_type.py"
WORKFLOW = ROOT / ".github" / "workflows" / "dependabot-auto-merge.yml"

TEMPLATE = (
    "meta: bump {name}\n\n---\nupdated-dependencies:\n"
    "- dependency-name: {name}\n  dependency-type: direct:production\n"
    "  update-type: {kind}\n...\n"
)


@pytest.fixture
def update_type():
    spec = importlib.util.spec_from_file_location("update_type", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message(kind, name="demo"):
    return TEMPLATE.format(name=name, kind=kind)


def test_patch_update_is_accepted(update_type):
    assert update_type.is_patch_only(message("version-update:semver-patch"))


@pytest.mark.parametrize(
    "kind", ["version-update:semver-minor", "version-update:semver-major"]
)
def test_minor_and_major_updates_are_rejected(update_type, kind):
    assert not update_type.is_patch_only(message(kind))


def test_mixed_group_is_rejected(update_type):
    text = message("version-update:semver-patch", "a") + message(
        "version-update:semver-minor", "b"
    )
    assert not update_type.is_patch_only(text)


def test_message_without_metadata_is_rejected(update_type):
    assert not update_type.is_patch_only("Fix a typo\n")
    assert not update_type.is_patch_only("")


def run_script(text):
    return subprocess.run(
        [sys.executable, str(SCRIPT)], input=text, text=True
    ).returncode


def test_script_exit_code_follows_update_type():
    assert run_script(message("version-update:semver-patch")) == 0
    assert run_script(message("version-update:semver-major")) == 1


def load_workflow():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_waits_for_successful_check_run():
    config = load_workflow()
    triggers = config.get("on", config.get(True))
    assert triggers["workflow_run"]["workflows"] == ["Check"]
    condition = config["jobs"]["merge-patch"]["if"]
    assert "conclusion == 'success'" in condition
    assert "dependabot[bot]" in condition


def test_workflow_merges_only_patch_updates():
    config = load_workflow()
    steps = config["jobs"]["merge-patch"]["steps"]
    commands = " ".join(step.get("run", "") for step in steps)
    assert "scripts/dependabot_update_type.py" in commands
    assert "gh pr merge" in commands
    assert "--match-head-commit" in commands


def test_workflow_never_checks_out_pull_request_code():
    config = load_workflow()
    for step in config["jobs"]["merge-patch"]["steps"]:
        assert "ref" not in step.get("with", {})
