import json
import os

import pytest

from korshmer.project_management.import_manager import ImportManager
from korshmer.project_management.project_manager import ProjectManager

BAD_SESSION_NAMES = [
    "../escape",
    "../../escape",
    "..\\escape",
    "/abs/escape",
    "bad\x00name",
    "..",
    "",
]


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "app"
    b.mkdir()
    return b


def _outside(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name != "app")


class TestCreateImportSession:
    @pytest.mark.parametrize("name", BAD_SESSION_NAMES)
    def test_rejects_unsafe_names(self, base, tmp_path, name):
        manager = ImportManager(base)
        with pytest.raises(ValueError):
            manager.create_import_session(name)
        assert _outside(tmp_path) == []
        assert list(manager.imports_dir.iterdir()) == []

    def test_regular_name_stays_inside_imports_dir(self, base):
        manager = ImportManager(base)
        session = manager.create_import_session("My Session")
        assert session.is_dir()
        assert session.parent == manager.imports_dir

    @pytest.mark.skipif(
        not hasattr(os, "symlink") or os.name == "nt",
        reason="symlinks need privileges on Windows",
    )
    def test_symlink_name_is_not_followed_out(self, base, tmp_path):
        manager = ImportManager(base)
        link = manager.imports_dir / "link"
        link.symlink_to(tmp_path)
        with pytest.raises(ValueError):
            manager.create_import_session("link/../../escape")
        assert _outside(tmp_path) == []


class TestLoadProjectPaths:
    def test_null_byte_path_does_not_load(self, base):
        manager = ProjectManager(base_dir=str(base))
        try:
            result = manager.load_project(str(base / "a\x00.json"))
        except (ValueError, OSError):
            return
        assert not result

    def test_traversal_to_missing_file_does_not_create_anything(
        self, base, tmp_path
    ):
        manager = ProjectManager(base_dir=str(base))
        try:
            result = manager.load_project(str(base / ".." / "missing.json"))
        except (ValueError, OSError):
            result = None
        assert not result
        assert _outside(tmp_path) == []

    def test_external_project_file_is_read_only(self, base, tmp_path):

        external = tmp_path / "outside.json"
        external.write_text(json.dumps({"name": "x"}), encoding="utf-8")
        before = external.read_text(encoding="utf-8")
        manager = ProjectManager(base_dir=str(base))
        try:
            manager.load_project(str(external))
        except Exception:
            pass
        assert external.read_text(encoding="utf-8") == before
        assert _outside(tmp_path) == ["outside.json"]
