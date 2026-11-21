import json

from korshmer.ui.managers import workspace
from korshmer.ui.managers.workspace import WorkspaceManager

LAYOUT_FILE = "window_layout.json"
SCREEN = (0, 0, 1920, 1080)


def on_screen(geometry, screens=(SCREEN,)):
    return workspace.geometry_on_screen(geometry, list(screens), 640, 480)


def test_a_saved_window_layout_is_read_back_by_a_new_session(tmp_path):
    layout = {"geometry": [10, 20, 900, 700], "tab": 2, "splitter": [0, 900]}
    WorkspaceManager(data_dir=tmp_path).save_window_layout(layout)
    assert WorkspaceManager(data_dir=tmp_path).load_window_layout() == layout


def test_the_layout_is_kept_in_one_file_in_the_data_folder(tmp_path):
    WorkspaceManager(data_dir=tmp_path).save_window_layout({"tab": 1})
    assert json.loads((tmp_path / LAYOUT_FILE).read_text()) == {"tab": 1}


def test_a_missing_layout_file_gives_an_empty_layout(tmp_path):
    assert WorkspaceManager(data_dir=tmp_path).load_window_layout() == {}


def test_a_damaged_layout_file_gives_an_empty_layout(tmp_path):
    (tmp_path / LAYOUT_FILE).write_text("{not json")
    assert WorkspaceManager(data_dir=tmp_path).load_window_layout() == {}


def test_a_layout_file_holding_a_list_gives_an_empty_layout(tmp_path):
    (tmp_path / LAYOUT_FILE).write_text(json.dumps([1, 2, 3]))
    assert WorkspaceManager(data_dir=tmp_path).load_window_layout() == {}


def test_without_a_data_folder_no_layout_is_saved():
    assert WorkspaceManager().save_window_layout({"tab": 1}) is False


def test_remembering_one_setting_keeps_the_other_saved_settings(tmp_path):
    manager = WorkspaceManager(data_dir=tmp_path)
    manager.save_window_layout({"tab": 3})
    manager.remember_layout_value("time_display", "bars")
    assert WorkspaceManager(data_dir=tmp_path).load_window_layout() == {
        "tab": 3,
        "time_display": "bars",
    }


def test_the_layout_file_is_not_mistaken_for_a_named_workspace(tmp_path):
    manager = WorkspaceManager(data_dir=tmp_path)
    manager.save_window_layout({"tab": 1})
    manager._load_from_disk()
    assert sorted(manager.get_all_workspaces()) == [
        "editing",
        "mastering",
        "mixing",
        "recording",
    ]


def test_a_window_inside_the_screen_keeps_its_place():
    assert on_screen([100, 80, 900, 700]) == (100, 80, 900, 700)


def test_a_window_saved_on_a_monitor_that_is_gone_moves_onto_the_screen():
    x, y, w, h = on_screen([2500, 100, 900, 700])
    assert 0 <= x and x + w <= 1920 and 0 <= y and y + h <= 1080


def test_a_window_larger_than_the_screen_is_shrunk_to_fit():
    assert on_screen([0, 0, 4000, 3000]) == SCREEN


def test_a_window_is_never_restored_below_its_minimum_size():
    assert on_screen([10, 10, 50, 40])[2:] == (640, 480)


def test_a_window_on_the_second_monitor_stays_there():
    second = (1920, 0, 1920, 1080)
    assert on_screen([2000, 50, 900, 700], (SCREEN, second)) == (
        2000,
        50,
        900,
        700,
    )


def test_a_malformed_saved_geometry_is_ignored():
    assert on_screen(["a", None, 3]) is None


def test_without_any_screen_the_saved_geometry_is_ignored():
    assert on_screen([0, 0, 900, 700], ()) is None
