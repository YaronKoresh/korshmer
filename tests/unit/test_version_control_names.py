import shutil
import subprocess

import pytest

from korshmer.project_management.version_control import VersionControlManager

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required"
)


@pytest.fixture
def vc(tmp_path, monkeypatch):
    for key, value in {
        "GIT_AUTHOR_NAME": "Tester",
        "GIT_AUTHOR_EMAIL": "tester@example.com",
        "GIT_COMMITTER_NAME": "Tester",
        "GIT_COMMITTER_EMAIL": "tester@example.com",
    }.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "saved_data").mkdir()
    manager = VersionControlManager(tmp_path)
    assert manager.init_repository()
    return manager


def _git(manager, *args):
    return subprocess.run(
        ["git", *args],
        cwd=manager.repo_dir,
        capture_output=True,
        text=True,
    ).stdout


def test_tag_name_that_looks_like_an_option_is_rejected(vc):
    assert vc.tag_version("-l") is False
    assert vc.tag_version("--delete") is False


def test_valid_tag_is_created(vc):
    assert vc.tag_version("v1.0", "first release") is True
    assert "v1.0" in _git(vc, "tag").split()


def test_tag_with_invalid_message_is_rejected(vc):
    assert vc.tag_version("v2.0", "bad\x00message") is False
    assert _git(vc, "tag").split() == []


@pytest.mark.parametrize("name", ["--abort", "-s", "a b", "x..y", ""])
def test_merge_rejects_invalid_branch_names(vc, name):
    assert vc.merge_branch(name) is False


def test_merge_still_works_for_a_valid_branch(vc):
    assert vc.create_branch("feature")
    assert vc.merge_branch("feature") is True


@pytest.mark.parametrize(
    "remote, branch",
    [("--all", "main"), ("origin", "--delete"), ("or igin", "main")],
)
def test_push_and_pull_reject_invalid_remote_or_branch(
    vc, monkeypatch, remote, branch
):
    calls = []
    monkeypatch.setattr(
        vc, "_run_git_command", lambda args, cwd=None: calls.append(args)
    )
    assert vc.push_to_remote(remote, branch) is False
    assert vc.pull_from_remote(remote, branch) is False
    assert calls == []


def test_push_and_pull_pass_valid_names_to_git(vc, monkeypatch):
    calls = []

    def fake(args, cwd=None):
        calls.append(args)
        return True, ""

    monkeypatch.setattr(vc, "_run_git_command", fake)
    assert vc.push_to_remote("origin", "main") is True
    assert vc.pull_from_remote("origin", "main") is True
    assert calls == [["push", "origin", "main"], ["pull", "origin", "main"]]


def test_file_names_starting_with_dash_are_not_options(vc):
    odd = vc.repo_dir / "-n"
    odd.write_text("data")
    assert vc.commit("add odd file", [odd]) is True
    assert "-n" in _git(vc, "ls-files").split()
    odd.write_text("changed")
    assert "changed" in vc.get_diff(odd)


def test_remote_name_cannot_start_with_dash(vc):
    assert vc.add_remote("--mirror", "https://example.com/a.git") is False
    assert vc.add_remote("origin", "https://example.com/a.git") is True


def _commit_all(vc, *names):
    for name in names:
        (vc.repo_dir / name).write_text("1")
    assert vc.commit("add files")


def test_status_keeps_full_name_of_first_modified_file(vc):
    _commit_all(vc, "song.json")
    (vc.repo_dir / "song.json").write_text("2")
    status = vc.get_status()
    assert status["modified_files"] == ["song.json"]
    assert status["staged_files"] == []


def test_status_reports_names_with_spaces_and_non_ascii(vc):
    _commit_all(vc, "my song.json", "שיר.json")
    (vc.repo_dir / "my song.json").write_text("2")
    (vc.repo_dir / "שיר.json").write_text("2")
    (vc.repo_dir / "new take.mid").write_text("x")
    status = vc.get_status()
    assert sorted(status["modified_files"]) == [
        "my song.json",
        "שיר.json",
    ]
    assert status["untracked_files"] == ["new take.mid"]


def test_status_names_survive_a_cp1252_default_encoding(vc, monkeypatch):
    _commit_all(vc, "שיר.json")
    (vc.repo_dir / "שיר.json").write_text("2")
    real_run = subprocess.run

    def run_with_cp1252_default(*args, **kwargs):
        if kwargs.pop("text", False):
            kwargs.setdefault("encoding", "cp1252")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_cp1252_default)
    assert vc.get_status()["modified_files"] == ["שיר.json"]


def test_status_handles_a_staged_rename(vc):
    _commit_all(vc, "old.json", "other.json")
    _git(vc, "mv", "old.json", "new.json")
    (vc.repo_dir / "other.json").write_text("2")
    status = vc.get_status()
    assert status["staged_files"] == ["new.json"]
    assert status["modified_files"] == ["other.json"]
