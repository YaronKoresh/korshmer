import tempfile
from pathlib import Path

from korshmer import __version__
from korshmer.project_management.project_manager import ProjectManager


class TestProjectManagerInit:
    def test_init_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            assert pm.projects_dir.exists()
            assert pm.backups_dir.exists()
            assert pm.templates_dir.exists()
            assert pm.snapshots_dir.exists()

    def test_init_sets_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            assert pm.base_dir == Path(tmpdir)


class TestCreateProject:
    def test_create_project_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            project = pm.create_project("Test Project")

            assert project["name"] == "Test Project"
            assert "version" in project
            assert "created" in project
            assert "modified" in project

    def test_create_project_version_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            project = pm.create_project("Version Test")

            assert project["version"] == __version__, (
                f"Project version {project['version']!r} "
                f"does not match __version__ {__version__!r}"
            )

    def test_create_project_has_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            project = pm.create_project("Metadata Test")

            assert "metadata" in project
            assert "bpm" in project["metadata"]
            assert "time_signature" in project["metadata"]
            assert "key" in project["metadata"]

    def test_create_project_has_empty_tracks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            project = pm.create_project("Tracks Test")

            assert project["tracks"] == []

    def test_create_project_default_bpm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            project = pm.create_project("BPM Test")

            assert project["metadata"]["bpm"] == 120


class TestSaveAndLoadProject:
    def test_save_project_no_path_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            pm.create_project("No Path")
            result = pm.save_project()
            assert result is False

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            pm.create_project("Roundtrip Test")

            project_path = pm.projects_dir / "test_project.kproj"
            result = pm.save_project(project_path, compress=True)
            assert result is True
            assert project_path.exists()

            loaded = pm.load_project(project_path)
            assert loaded is not None
            assert loaded["name"] == "Roundtrip Test"
            assert loaded["version"] == __version__


class TestMigrateProject:
    def test_migrate_default_target_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            import inspect

            sig = inspect.signature(pm.migrate_project)
            default = sig.parameters["target_version"].default
            assert default == __version__, (
                f"migrate_project default target {default!r} "
                f"should be {__version__!r}"
            )


class TestConcurrentProjectAccess:
    def test_parallel_save_and_load_keep_project_file_valid(self):
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = ProjectManager(tmpdir)
            pm.create_project("Concurrent")
            project_path = pm.projects_dir / "Concurrent.kproj"
            assert pm.save_project(project_path)

            errors: list[BaseException] = []

            def worker(index: int):
                try:
                    for step in range(5):
                        pm.set_metadata({"description": f"{index}-{step}"})
                        assert pm.save_project(project_path)
                        assert pm.list_backups() is not None
                        assert pm.get_project_stats()
                except BaseException as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)

            assert not any(t.is_alive() for t in threads)
            assert errors == []

            reloaded = ProjectManager(tmpdir)
            data = reloaded.load_project(project_path)
            assert data is not None
            assert data["name"] == "Concurrent"
            assert data["metadata"]["description"]
