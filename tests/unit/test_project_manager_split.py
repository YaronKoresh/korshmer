import gzip
import json

import pytest

from korshmer.project_management.project_manager import ProjectManager
from korshmer.project_management.project_migration import (
    migrate_project_data,
    needs_migration,
)


@pytest.fixture
def pm(tmp_path):
    base = tmp_path / "app"
    base.mkdir()
    return ProjectManager(base_dir=str(base))


class TestMigrationModule:
    def test_needs_migration(self):
        assert needs_migration("0.1.0", "0.5.1") is True
        assert needs_migration("0.5.1", "0.5.1") is False
        assert needs_migration("0.6.0", "0.5.1") is False

    def test_migrate_adds_sections(self):
        data = {"version": "0.1.0", "metadata": {}}
        out = migrate_project_data(data, "0.5.1", "/tmp/x.backup")
        assert out["version"] == "0.5.1"
        assert "ui_settings" in out
        assert "ai_settings" in out
        assert "project_management" in out
        assert out["migration_history"][-1]["from_version"] == "0.1.0"
        assert out["migration_history"][-1]["backup_path"] == "/tmp/x.backup"


class TestBackup:
    def test_create_backup_and_cleanup(self, pm):
        pm.create_project("Song")
        pm.save_project(pm.projects_dir / "Song.kproj")
        pm.max_backups = 2
        pm.auto_backup_interval = 0
        created = 0
        for _ in range(4):
            pm.last_backup_time = 0
            if pm.create_backup():
                created += 1
        backups = pm.list_backups("Song")
        assert created >= 1
        assert len(backups) <= pm.max_backups


class TestSnapshot:
    def test_snapshot_round_trip(self, pm):
        pm.create_project("Song")
        pm.save_project(pm.projects_dir / "Song.kproj")
        pm.current_project_data["marker"] = "value"
        assert pm.create_snapshot("test") is True
        snaps = pm.list_snapshots("Song")
        assert len(snaps) == 1
        restored = pm.restore_snapshot(snaps[0])
        assert restored is True
        assert "snapshot_info" not in pm.current_project_data


class TestRecovery:
    def test_recovery_round_trip(self, pm):
        pm.create_project("Song")
        pm.current_project_path = pm.projects_dir / "Song.kproj"
        assert pm.save_recovery_state() is True
        recovery_file = pm.base_dir / "saved_data" / ".recovery.gz"
        assert recovery_file.exists()
        with gzip.open(recovery_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert data["project_data"]["name"] == "Song"
        loaded = pm.enable_auto_recovery()
        assert loaded["project_data"]["name"] == "Song"
        assert pm.clear_recovery_state() is True
        assert not recovery_file.exists()


class TestMigrateProjectIntegration:
    def test_migrate_old_project(self, pm):
        pm.create_project("Song")
        path = pm.projects_dir / "Song.kproj"
        pm.current_project_data["version"] = "0.1.0"
        pm.save_project(path)
        assert pm.migrate_project(path, target_version="0.5.1") is True
        data = pm.load_project(path)
        assert data["version"] == "0.5.1"
        assert "migration_history" in data


class TestLoadWithFallback:
    def _corrupt_with_backup(self, pm):
        pm.create_project("Song")
        path = pm.projects_dir / "Song.kproj"
        pm.save_project(path)
        pm.current_project_data["marker"] = "from-backup"
        pm.save_project(path)
        pm.auto_backup_interval = 0
        pm.last_backup_time = 0
        assert pm.create_backup() is True
        path.write_bytes(b"not a project")
        return path

    def test_corrupt_project_loads_backup(self, pm):
        path = self._corrupt_with_backup(pm)
        data = pm.load_project_with_fallback(path)
        assert data["marker"] == "from-backup"
        assert pm.last_load_backup is not None
        assert pm.current_project_path == path

    def test_no_backup_returns_none(self, pm):
        missing = pm.projects_dir / "Nothing.kproj"
        assert pm.load_project_with_fallback(missing) is None
        assert pm.last_load_backup is None

    def test_valid_project_not_marked_backup(self, pm):
        pm.create_project("Song")
        path = pm.projects_dir / "Song.kproj"
        pm.save_project(path)
        assert pm.load_project_with_fallback(path) is not None
        assert pm.last_load_backup is None


class TestMigrationMalformedSections:
    @pytest.mark.parametrize("metadata", ["x", None, 5, ["bpm"]])
    def test_non_object_metadata_does_not_crash(self, metadata):
        data = {"version": "0.1.0", "metadata": metadata}
        out = migrate_project_data(data, "0.5.1", "/tmp/x.backup")
        assert out["version"] == "0.5.1"
        assert out["metadata"] == metadata
        assert out["theory_settings"]["global_bpm"] == 120

    @pytest.mark.parametrize("history", ["x", None, 5, {"a": 1}])
    def test_non_list_migration_history_is_replaced(self, history):
        data = {"version": "0.5.1", "migration_history": history}
        out = migrate_project_data(data, "0.5.1", "/tmp/x.backup")
        assert len(out["migration_history"]) == 1
