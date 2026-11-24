import hashlib
import json
import pickle

import pytest

from korshmer.plugins.registry import PluginRegistry

SWITCH = "KORSHMER_ENABLE_USER_PLUGINS"
APPROVALS = "approved_plugins.json"
PLUGIN_SOURCE = (
    "from korshmer.plugins.plugin_interface import AudioEffectPlugin\n"
    "class Ext(AudioEffectPlugin):\n"
    "    name = 'Folder Effect'\n"
    "    def process(self, audio_path, output_path, **params):\n"
    "        return output_path\n"
)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    PluginRegistry.reset()
    yield PluginRegistry()
    PluginRegistry.reset()


@pytest.fixture
def folder(tmp_path):
    plugins = tmp_path / "saved_data" / "plugins"
    plugins.mkdir(parents=True)
    return plugins


def write_plugin(folder, name="ext.py", source=PLUGIN_SOURCE):
    path = folder / name
    path.write_text(source)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve(folder, **entries):
    (folder.parent / APPROVALS).write_text(json.dumps(entries))


def test_a_plugin_file_in_the_folder_is_not_run_without_consent(
    folder, registry
):
    marker = folder.parent / "ran.txt"
    write_plugin(
        folder, source=f"open({str(marker)!r}, 'w').close()\n" + PLUGIN_SOURCE
    )
    registry.load_all(plugin_dir=str(folder))
    assert not marker.exists()
    assert registry.get_plugin("Folder Effect") is None


def test_built_in_plugins_load_without_consent_as_a_safeguard(folder, registry):
    from korshmer.plugins.builtin import BUILTIN_PLUGINS

    registry.load_all(plugin_dir=str(folder))
    assert len(registry.get_plugins()) >= len(BUILTIN_PLUGINS)


def test_the_environment_switch_loads_every_plugin_as_a_safeguard(
    folder, registry, monkeypatch
):
    write_plugin(folder)
    monkeypatch.setenv(SWITCH, "1")
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is not None


def test_a_switch_set_to_zero_does_not_load_plugins(
    folder, registry, monkeypatch
):
    write_plugin(folder)
    monkeypatch.setenv(SWITCH, "0")
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is None


def test_an_approved_plugin_with_a_matching_checksum_loads_as_a_safeguard(
    folder, registry
):
    approve(folder, **{"ext.py": write_plugin(folder)})
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is not None


def test_an_approved_plugin_that_changed_afterwards_is_skipped(
    folder, registry
):
    approve(folder, **{"ext.py": write_plugin(folder)})
    (folder / "ext.py").write_text(PLUGIN_SOURCE + "X = 1\n")
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is None


def test_an_approval_list_dropped_into_the_plugin_folder_approves_nothing(
    folder, registry
):
    digest = write_plugin(folder)
    (folder / APPROVALS).write_text(json.dumps({"ext.py": digest}))
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is None


def test_a_skipped_plugin_is_reported_with_its_checksum(
    folder, registry, caplog
):
    digest = write_plugin(folder)
    with caplog.at_level("WARNING", logger="korshmer.plugins.registry"):
        registry.load_all(plugin_dir=str(folder))
    assert any(
        "ext.py" in r.getMessage() and digest in r.getMessage()
        for r in caplog.records
    )


def test_a_damaged_approval_list_approves_nothing(folder, registry):
    write_plugin(folder)
    (folder.parent / APPROVALS).write_text("{not json")
    registry.load_all(plugin_dir=str(folder))
    assert registry.get_plugin("Folder Effect") is None


def test_objects_made_by_a_plugin_can_be_saved_and_restored(
    folder, registry, monkeypatch
):
    monkeypatch.setenv(SWITCH, "1")
    write_plugin(
        folder,
        source=PLUGIN_SOURCE
        + "from dataclasses import dataclass\n"
        + "@dataclass\n"
        + "class Settings:\n"
        + "    amount: 'float' = 0.5\n"
        + "Ext.settings_type = Settings\n",
    )
    registry.load_all(plugin_dir=str(folder))
    settings = registry.get_plugin("Folder Effect").settings_type(0.75)
    assert pickle.loads(pickle.dumps(settings)) == settings


def test_a_plugin_that_fails_to_start_leaves_no_half_loaded_module_as_a_safeguard(
    folder, registry, monkeypatch
):
    import sys

    monkeypatch.setenv(SWITCH, "1")
    write_plugin(folder, name="broken.py", source="raise RuntimeError('no')\n")
    registry.load_all(plugin_dir=str(folder))
    assert "korshmer_plugin_broken" not in sys.modules
