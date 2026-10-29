import sys

import pytest

from korshmer.project_management.asset_manager import AssetManager
from korshmer.project_management.project_backup import cleanup_old_backups
from korshmer.project_management.project_manager import ProjectManager
from korshmer.utils.search_input import (
    MAX_SEARCH_QUERY_LENGTH,
    sanitize_search_query,
)

skip_on_windows_star = pytest.mark.skipif(
    sys.platform == "win32",
    reason='"*" is not a legal filename character on Windows',
)


class TestSanitizeSearchQuery:
    def test_removes_null_and_control_characters(self):
        assert sanitize_search_query("ab\x00cd") == "abcd"
        assert sanitize_search_query("a\nb\tc\x1fd") == "abcd"
        assert sanitize_search_query("x\x7fy") == "xy"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_search_query("  hello  ") == "hello"

    def test_truncates_to_max_length(self):
        result = sanitize_search_query("a" * 5000)
        assert len(result) == MAX_SEARCH_QUERY_LENGTH
        assert len(result) == 4096

    def test_does_not_filter_dangerous_words(self):

        assert sanitize_search_query("Drop Bass") == "Drop Bass"
        assert sanitize_search_query("OR 1=1; --") == "OR 1=1; --"

    def test_internal_spaces_preserved(self):
        assert sanitize_search_query("Rock n Roll") == "Rock n Roll"

    def test_non_string_returns_empty(self):
        assert sanitize_search_query(None) == ""
        assert sanitize_search_query(12345) == ""


class TestListBackupsGlobEscaping:
    def test_bracket_name_matches_only_itself(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.backups_dir / "Song [Remix]_backup_1.kproj").write_text("")
        (pm.backups_dir / "Song R_backup_1.kproj").write_text("")

        result = pm.list_backups("Song [Remix]")
        names = [p.name for p in result]
        assert names == ["Song [Remix]_backup_1.kproj"]

    @skip_on_windows_star
    def test_star_name_matches_only_itself(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.backups_dir / "a*_backup_1.kproj").write_text("")
        (pm.backups_dir / "abc_backup_1.kproj").write_text("")

        result = pm.list_backups("a*")
        names = [p.name for p in result]
        assert names == ["a*_backup_1.kproj"]


class TestListSnapshotsGlobEscaping:
    def test_bracket_name_matches_only_itself(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.snapshots_dir / "Song [Remix]_snapshot_1.kproj").write_text("")
        (pm.snapshots_dir / "Song R_snapshot_1.kproj").write_text("")

        result = pm.list_snapshots("Song [Remix]")
        names = [p.name for p in result]
        assert names == ["Song [Remix]_snapshot_1.kproj"]

    @skip_on_windows_star
    def test_star_name_matches_only_itself(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.snapshots_dir / "a*_snapshot_1.kproj").write_text("")
        (pm.snapshots_dir / "abc_snapshot_1.kproj").write_text("")

        result = pm.list_snapshots("a*")
        names = [p.name for p in result]
        assert names == ["a*_snapshot_1.kproj"]


class TestCleanupOldBackupsDoesNotTouchOtherProjects:
    def test_cleanup_for_bracket_name_leaves_other_project_backups(
        self, tmp_path
    ):
        pm = ProjectManager(base_dir=str(tmp_path))

        for i in range(11):
            (pm.backups_dir / f"Song R_backup_{i:02d}.kproj").write_text("")
        (pm.backups_dir / "Song [Remix]_backup_1.kproj").write_text("")

        cleanup_old_backups(pm.backups_dir, "Song [Remix]", pm.max_backups)

        remaining = {
            p.name for p in pm.backups_dir.glob("Song R_backup_*.kproj")
        }
        assert len(remaining) == 11, (
            "cleanup for 'Song [Remix]' must not delete 'Song R' backups"
        )
        assert (pm.backups_dir / "Song [Remix]_backup_1.kproj").exists()


class TestListWithoutArgumentUnchanged:
    def test_list_backups_no_arg_returns_all(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.backups_dir / "ProjA_backup_1.kproj").write_text("")
        (pm.backups_dir / "ProjB_backup_1.kproj").write_text("")

        result = pm.list_backups()
        assert len(result) == 2

    def test_list_snapshots_no_arg_returns_all(self, tmp_path):
        pm = ProjectManager(base_dir=str(tmp_path))
        (pm.snapshots_dir / "ProjA_snapshot_1.kproj").write_text("")
        (pm.snapshots_dir / "ProjB_snapshot_1.kproj").write_text("")

        result = pm.list_snapshots()
        assert len(result) == 2


class TestSearchAssetsSanitized:
    def _make_manager(self, tmp_path):
        am = AssetManager(base_dir=str(tmp_path))
        am.asset_library = {
            "id1": {"name": "Drop Bass", "tags": ["bass"], "type": "audio"},
            "id2": {"name": "Lead Synth", "tags": ["synth"], "type": "audio"},
        }
        return am

    def test_plain_query_matches(self, tmp_path):
        am = self._make_manager(tmp_path)
        results = am.search_assets("drop")
        assert [r["name"] for r in results] == ["Drop Bass"]

    def test_query_with_null_byte_does_not_crash(self, tmp_path):
        am = self._make_manager(tmp_path)
        results = am.search_assets("dr\x00op")
        assert [r["name"] for r in results] == ["Drop Bass"]

    def test_case_insensitive_preserved(self, tmp_path):
        am = self._make_manager(tmp_path)
        assert am.search_assets("DROP")
        assert am.search_assets("drop")
