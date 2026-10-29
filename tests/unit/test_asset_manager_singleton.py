import threading
import time

from korshmer.project_management.asset_manager import AssetManager


def test_get_instance_returns_same_object_for_same_base_dir(tmp_path):
    first = AssetManager.get_instance(tmp_path)
    second = AssetManager.get_instance(str(tmp_path))

    assert first is second


def test_get_instance_separates_different_base_dirs(tmp_path):
    a = AssetManager.get_instance(tmp_path / "a")
    b = AssetManager.get_instance(tmp_path / "b")

    assert a is not b


def test_constructor_signature_still_creates_separate_objects(tmp_path):
    direct = AssetManager(tmp_path)
    shared = AssetManager.get_instance(tmp_path)

    assert direct is not shared


def test_concurrent_first_calls_create_one_instance(tmp_path, monkeypatch):
    creations = []
    original_load = AssetManager._load_library

    def slow_load(self):
        creations.append(self)
        time.sleep(0.05)
        return original_load(self)

    monkeypatch.setattr(AssetManager, "_load_library", slow_load)

    start = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def call():
        start.wait(timeout=10)
        instance = AssetManager.get_instance(tmp_path / "shared")
        with lock:
            results.append(instance)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 8
    assert all(r is results[0] for r in results)
    assert len(creations) == 1
