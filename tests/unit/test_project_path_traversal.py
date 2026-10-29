import os
import zipfile
from pathlib import Path

import pytest

from korshmer.project_management.project_manager import (
    ProjectManager,
    is_safe_name,
    is_valid_new_name,
    resolve_inside,
)


@pytest.fixture
def workspace(tmp_path):
    base = tmp_path / "app"
    base.mkdir()
    return tmp_path, ProjectManager(base_dir=str(base))


def _outside_entries(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name != "app")


class TestIsSafeName:
    @pytest.mark.parametrize(
        "name",
        ["", "   ", ".", "..", "../evil", "a/b", "/tmp/x", "a\\b", "a\x00b"],
    )
    def test_rejects_unsafe_names(self, name):
        assert is_safe_name(name) is False

    def test_rejects_non_strings(self):
        assert is_safe_name(None) is False
        assert is_safe_name(5) is False

    @pytest.mark.parametrize("name", ["My Song", "Song [Remix]", "a.b", "x"])
    def test_accepts_regular_names(self, name):
        assert is_safe_name(name) is True


class TestIsValidNewName:
    @pytest.mark.parametrize(
        "name",
        [
            "a:b",
            "x?y",
            "a*b",
            'q"t',
            "<a>",
            "p|q",
            "a\tb",
            "a\nb",
            "CON",
            "con",
            "NUL.txt",
            "song.",
            "song ",
            "x" * 300,
            "a/b",
            "..",
            "",
            "a\x00b",
        ],
    )
    def test_rejects_invalid_names(self, name):
        assert is_valid_new_name(name) is False

    def test_rejects_non_strings(self):
        assert is_valid_new_name(None) is False
        assert is_valid_new_name(5) is False

    @pytest.mark.parametrize(
        "name",
        ["My Song", "Song [Remix]", "a.b", "x", "שיר חדש", "Song (v2)"],
    )
    def test_accepts_regular_names(self, name):
        assert is_valid_new_name(name) is True


class TestResolveInside:
    def test_rejects_sibling_with_common_prefix(self, tmp_path):
        base = tmp_path / "saved_data"
        base.mkdir()
        assert resolve_inside(base, tmp_path / "saved_data_evil" / "x") is None

    def test_rejects_parent_traversal(self, tmp_path):
        base = tmp_path / "saved_data"
        assert resolve_inside(base, base / ".." / "x") is None

    def test_rejects_base_itself(self, tmp_path):
        assert resolve_inside(tmp_path, tmp_path) is None

    def test_accepts_child_with_relative_base(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert (
            resolve_inside("base", Path("base") / "a" / "b.kproj")
            == (tmp_path / "base" / "a" / "b.kproj").resolve()
        )

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
    def test_rejects_symlink_escape(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            (base / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not permitted")
        assert resolve_inside(base, base / "link" / "x.kproj") is None


class TestCreateProject:
    @pytest.mark.parametrize(
        "name", ["../../../outside_proj", "a/b", "/tmp/x_korshmer_pt", ".."]
    )
    def test_unsafe_names_rejected(self, workspace, name):
        tmp_path, pm = workspace
        before = sorted(p.name for p in pm.projects_dir.iterdir())
        assert pm.create_project(name) is None
        assert _outside_entries(tmp_path) == []
        assert sorted(p.name for p in pm.projects_dir.iterdir()) == before
        assert not Path("/tmp/x_korshmer_pt").exists()

    def test_regular_name_still_works(self, workspace):
        _, pm = workspace
        project = pm.create_project("My Song")
        assert project is not None
        assert project["name"] == "My Song"
        assert (pm.projects_dir / "My Song").is_dir()

    @pytest.mark.parametrize("name", ["Song: Remix", "Mix?", "CON", "song."])
    def test_filesystem_unsafe_names_rejected(self, workspace, name):
        _, pm = workspace
        before = sorted(p.name for p in pm.projects_dir.iterdir())
        assert pm.create_project(name) is None
        assert sorted(p.name for p in pm.projects_dir.iterdir()) == before

    def test_too_long_name_returns_none_without_oserror(self, workspace):
        _, pm = workspace
        assert pm.create_project("x" * 300) is None


class TestTemplates:
    def test_save_as_template_rejects_traversal(self, workspace):
        tmp_path, pm = workspace
        pm.create_project("Song")
        assert pm.save_as_template("../x") is False
        assert pm.save_as_template("../../../outside_tpl") is False
        assert not (pm.base_dir / "saved_data" / "x.kproj").exists()
        assert _outside_entries(tmp_path) == []

    def test_load_template_rejects_traversal(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        outside = pm.templates_dir.parent / "x.kproj"
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("manifest.json", '{"name": "outside"}')
        assert outside.exists()
        assert pm.load_template("../x") is None

    def test_save_as_template_rejects_unsafe_chars(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        assert pm.save_as_template("a:b") is False
        assert not (pm.templates_dir / "a:b.kproj").exists()

    def test_template_round_trip(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        assert pm.save_as_template("Pop Template") is True
        assert "Pop Template" in pm.list_templates()
        data = pm.load_template("Pop Template")
        assert data is not None
        assert data["name"] == "Pop Template"


class TestSaveLoad:
    def test_save_outside_saved_data_refused(self, workspace):
        tmp_path, pm = workspace
        pm.create_project("Song")
        target = tmp_path / "a.kproj"
        assert pm.save_project(target) is False
        assert not target.exists()

    def test_save_to_prefix_sibling_refused(self, workspace):
        tmp_path, pm = workspace
        pm.create_project("Song")
        sibling = pm.base_dir / "saved_data_evil"
        sibling.mkdir()
        assert pm.save_project(sibling / "a.kproj") is False
        assert list(sibling.iterdir()) == []

    def test_save_with_traversal_inside_path_refused(self, workspace):
        tmp_path, pm = workspace
        pm.create_project("Song")
        target = pm.projects_dir / ".." / ".." / ".." / "b.kproj"
        assert pm.save_project(target) is False
        assert not (tmp_path / "b.kproj").exists()

    def test_uncompressed_final_suffix_checked(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        path = pm.projects_dir / "Song.kproj"
        assert pm.save_project(path, compress=False) is True
        assert (pm.projects_dir / "Song.json").exists()

    def test_save_and_load_inside_projects_dir(self, workspace):
        _, pm = workspace
        pm.create_project("My Song")
        path = pm.projects_dir / "My Song.kproj"
        assert pm.save_project(path) is True
        assert path.exists()
        pm2 = ProjectManager(base_dir=str(pm.base_dir))
        data = pm2.load_project(path)
        assert data is not None
        assert data["name"] == "My Song"


class TestSnapshots:
    def test_restore_snapshot_outside_snapshots_dir_refused(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        path = pm.projects_dir / "Song.kproj"
        assert pm.save_project(path) is True
        assert pm.restore_snapshot(path) is False

    def test_restore_snapshot_inside_works(self, workspace):
        _, pm = workspace
        pm.create_project("Song")
        assert pm.save_project(pm.projects_dir / "Song.kproj") is True
        assert pm.create_snapshot("first") is True
        snapshots = pm.list_snapshots("Song")
        assert len(snapshots) == 1
        assert pm.restore_snapshot(snapshots[0]) is True

    def test_list_with_unsafe_project_name_returns_empty(self, workspace):
        _, pm = workspace
        assert pm.list_backups("../../../outside") == []
        assert pm.list_snapshots("../../../outside") == []


class TestRepairAndMigrate:
    def test_repair_outside_saved_data_refused(self, workspace):
        tmp_path, pm = workspace
        outside = tmp_path / "broken.kproj"
        outside.write_bytes(b"not a zip")
        assert pm.repair_project(outside) is False
        assert not (tmp_path / "broken.kproj.backup").exists()

    def test_migrate_outside_saved_data_refused(self, workspace):
        tmp_path, pm = workspace
        outside = tmp_path / "old.kproj"
        with zipfile.ZipFile(outside, "w") as zf:
            zf.writestr("manifest.json", '{"name": "old", "version": "0.1.0"}')
        assert pm.migrate_project(outside) is False
        assert _outside_entries(tmp_path) == ["old.kproj"]
