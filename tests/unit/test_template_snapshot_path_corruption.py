from __future__ import annotations

import tempfile
from pathlib import Path

from korshmer.project_management.project_manager import ProjectManager


def test_save_as_template_preserves_project_path():
    with tempfile.TemporaryDirectory() as td:
        mgr = ProjectManager(Path(td))
        mgr.create_project("myproject")
        project_path = mgr.projects_dir / "myproject" / "myproject.kproj"
        mgr.save_project(project_path)

        original_path = mgr.current_project_path
        mgr.save_as_template("mytemplate")
        assert mgr.current_project_path == original_path, (
            "save_as_template must not change current_project_path"
        )


def test_create_snapshot_preserves_project_path():
    with tempfile.TemporaryDirectory() as td:
        mgr = ProjectManager(Path(td))
        mgr.create_project("snapproject")
        project_path = mgr.projects_dir / "snapproject" / "snapproject.kproj"
        mgr.save_project(project_path)

        original_path = mgr.current_project_path
        mgr.create_snapshot("test snapshot")
        assert mgr.current_project_path == original_path, (
            "create_snapshot must not change current_project_path"
        )


def test_load_template_preserves_project_state():
    with tempfile.TemporaryDirectory() as td:
        mgr = ProjectManager(Path(td))
        mgr.create_project("base")
        project_path = mgr.projects_dir / "base" / "base.kproj"
        mgr.save_project(project_path)
        mgr.save_as_template("tpl")

        original_path = mgr.current_project_path
        original_data_name = mgr.current_project_data.get("name")
        tpl = mgr.load_template("tpl")

        assert tpl is not None
        assert mgr.current_project_path == original_path, (
            "load_template must not change current_project_path"
        )
        assert mgr.current_project_data.get("name") == original_data_name, (
            "load_template must not change current_project_data"
        )
