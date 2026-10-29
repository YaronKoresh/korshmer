import json

import pytest

from korshmer.project_management.asset_manager import AssetManager


def _write(tmp_path, name, payload):
    folder = tmp_path / "saved_data"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("payload", [[1, 2], "text", 5, None])
def test_library_that_is_not_an_object_loads_empty(tmp_path, payload):
    _write(tmp_path, "asset_library.json", payload)

    manager = AssetManager(tmp_path)

    assert manager.asset_library == {}
    assert manager.get_all_tags() == []
    assert manager.search_assets() == []


def test_library_drops_entries_that_are_not_objects(tmp_path):
    _write(
        tmp_path,
        "asset_library.json",
        {"a": 5, "b": {"size": 3, "tags": ["x"]}, "c": [1]},
    )

    manager = AssetManager(tmp_path)

    assert list(manager.asset_library) == ["b"]
    assert manager.get_asset_stats()["total_assets"] == 1


@pytest.mark.parametrize("payload", ["abc", {"k": 1}, 7])
def test_favorites_that_are_not_a_list_load_empty(tmp_path, payload):
    _write(tmp_path, "favorites.json", payload)

    assert AssetManager(tmp_path).favorites == set()


def test_favorites_keep_only_text_items(tmp_path):
    _write(tmp_path, "favorites.json", ["a", 3, ["b"], "c"])

    assert AssetManager(tmp_path).favorites == {"a", "c"}
