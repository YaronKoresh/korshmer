import pytest

from korshmer.project_management.cloud_sync import (
    MAX_AUTO_SYNC_MINUTES,
    CloudSyncManager,
)


@pytest.fixture
def manager(tmp_path):
    m = CloudSyncManager(tmp_path / "app")
    assert m.connect("folder", {"path": str(tmp_path / "Dropbox")})
    yield m
    m.disconnect()


@pytest.mark.parametrize(
    "minutes", [1e10, 1e300, MAX_AUTO_SYNC_MINUTES + 1, float("inf")]
)
def test_an_interval_the_timer_cannot_hold_is_refused(manager, minutes):
    assert manager.enable_auto_sync(minutes) is False
    assert manager.auto_sync_active is False


@pytest.mark.parametrize("minutes", [1, 30, 90.5, MAX_AUTO_SYNC_MINUTES])
def test_a_long_but_valid_interval_keeps_the_timer_running(manager, minutes):
    assert manager.enable_auto_sync(minutes) is True
    assert manager.auto_sync_active is True
    manager.disable_auto_sync()
    assert manager.auto_sync_active is False
