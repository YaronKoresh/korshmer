import os
import subprocess
import sys
from pathlib import Path

from korshmer.plugins import PluginRegistry
from korshmer.plugins.builtin import BUILTIN_PLUGINS


def test_plugin_interface_exports_base_objects():
    from korshmer.plugins import base, plugin_interface

    for name in plugin_interface.__all__:
        assert getattr(plugin_interface, name) is getattr(base, name)


def test_discover_directory_with_plugin_interface_import(tmp_path):
    (tmp_path / "ext.py").write_text(
        "from korshmer.plugins.plugin_interface import AudioEffectPlugin\n"
        "class Ext(AudioEffectPlugin):\n"
        "    name = 'Interface Ext'\n"
        "    def process(self, audio_path, output_path, **params):\n"
        "        return output_path\n"
    )
    PluginRegistry.reset()
    try:
        assert PluginRegistry().discover_directory(str(tmp_path)) == 1
    finally:
        PluginRegistry.reset()


def test_registry_discovers_directory_in_fresh_interpreter(tmp_path):
    (tmp_path / "ext.py").write_text(
        "from korshmer.plugins.plugin_interface import ExporterPlugin\n"
        "class Ext(ExporterPlugin):\n"
        "    name = 'Fresh Ext'\n"
        "    def export(self, midi_path, output_path, **params):\n"
        "        return output_path\n"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    code = (
        f"import sys; sys.path.insert(0, {src!r});"
        "from korshmer.plugins.registry import PluginRegistry;"
        f"print(PluginRegistry().discover_directory({str(tmp_path)!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert out.stdout.strip().endswith("1"), out.stderr


def test_load_plugins_registers_builtins(tmp_path, monkeypatch):
    import korshmer.__main__ as entry

    monkeypatch.setattr(entry, "DATA_DIR", tmp_path)
    PluginRegistry.reset()
    try:
        entry.load_plugins()
        assert len(PluginRegistry().get_plugins()) >= len(BUILTIN_PLUGINS)
    finally:
        PluginRegistry.reset()
