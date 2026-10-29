import gzip
import json

from korshmer.project_management.project_recovery import (
    clear_recovery_file,
    load_recovery_file,
    save_recovery_file,
    validate_recovery_data,
)


def _write(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(payload if isinstance(payload, str) else json.dumps(payload))


def test_round_trip_returns_saved_project(tmp_path):
    recovery = tmp_path / ".recovery.gz"

    assert save_recovery_file(recovery, tmp_path / "song.kmp", {"bpm": 120})

    data = load_recovery_file(recovery)
    assert data["project_data"] == {"bpm": 120}
    assert data["project_path"] == str(tmp_path / "song.kmp")
    assert recovery.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_file_loads_nothing(tmp_path):
    assert load_recovery_file(tmp_path / ".recovery.gz") is None


def test_unreadable_gzip_is_set_aside(tmp_path):
    recovery = tmp_path / ".recovery.gz"
    recovery.write_bytes(b"not a gzip file")

    assert load_recovery_file(recovery) is None
    assert not recovery.exists()
    assert (
        tmp_path / ".recovery.gz.corrupt"
    ).read_bytes() == b"not a gzip file"
    assert load_recovery_file(recovery) is None


def test_truncated_gzip_is_set_aside(tmp_path):
    recovery = tmp_path / ".recovery.gz"
    assert save_recovery_file(recovery, None, {"tracks": list(range(500))})
    recovery.write_bytes(recovery.read_bytes()[:-20])

    assert load_recovery_file(recovery) is None
    assert (tmp_path / ".recovery.gz.corrupt").exists()


def test_wrong_shape_is_set_aside(tmp_path):
    good = {
        "timestamp": "2026-01-01T10:00:00",
        "project_path": None,
        "project_data": {},
    }
    cases = [
        [1, 2],
        {**good, "project_data": "text"},
        {**good, "project_data": None},
        {**good, "timestamp": 5},
        {**good, "timestamp": "yesterday"},
        {**good, "project_path": 7},
        {"project_data": {}},
        "{not json",
    ]
    for case in cases:
        recovery = tmp_path / ".recovery.gz"
        _write(recovery, case)
        assert load_recovery_file(recovery) is None, case
        assert not recovery.exists(), case


def test_valid_file_is_kept_in_place(tmp_path):
    recovery = tmp_path / ".recovery.gz"
    _write(
        recovery,
        {
            "timestamp": "2026-01-01T10:00:00",
            "project_path": None,
            "project_data": {"a": 1},
        },
    )

    assert load_recovery_file(recovery)["project_data"] == {"a": 1}
    assert recovery.exists()


def test_failed_save_keeps_previous_recovery_file(tmp_path):
    recovery = tmp_path / ".recovery.gz"
    assert save_recovery_file(recovery, None, {"v": 1})
    before = recovery.read_bytes()

    assert not save_recovery_file(recovery, None, {"v": {1, 2}})

    assert recovery.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_validate_accepts_complete_data():
    assert (
        validate_recovery_data(
            {
                "timestamp": "2026-01-01T10:00:00",
                "project_path": "x.kmp",
                "project_data": {},
            }
        )
        is None
    )


def test_clear_removes_file(tmp_path):
    recovery = tmp_path / ".recovery.gz"
    assert save_recovery_file(recovery, None, {})
    assert clear_recovery_file(recovery)
    assert not recovery.exists()
