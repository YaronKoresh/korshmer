import os

import pytest

from korshmer.plugins.base import (
    AudioEffectPlugin,
    ExporterPlugin,
    MidiProcessorPlugin,
    PluginParam,
    PluginType,
)
from korshmer.plugins.registry import PluginRegistry


class DummyMidiPlugin(MidiProcessorPlugin):
    name = "Test MIDI Plugin"
    description = "A test MIDI processor"
    category = "Test"
    version = "0.1.0"

    @classmethod
    def params(cls):
        return [
            PluginParam("amount", "Amount", "float", 50, 0, 100),
            PluginParam(
                "mode", "Mode", "choice", "fast", choices=["fast", "slow"]
            ),
        ]

    def process(self, midi_path, output_dir, **params):
        return os.path.join(output_dir, "result.mid")


class DummyAudioPlugin(AudioEffectPlugin):
    name = "Test Audio Plugin"
    category = "Effects"

    @classmethod
    def params(cls):
        return [PluginParam("gain", "Gain", "float", 0.5, 0, 1.0)]

    def process(self, audio_path, output_path, **params):
        return output_path


class DummyExporter(ExporterPlugin):
    name = "Test Exporter"
    category = "Export"

    def export(self, midi_path, output_path, **params):
        return output_path


@pytest.fixture(autouse=True)
def clean_registry():
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


class TestPluginParam:
    def test_float_validate(self):
        p = PluginParam("gain", "Gain", "float", 0.5, 0, 1.0)
        assert p.validate(0.5)
        assert p.validate(0)
        assert p.validate(1.0)
        assert not p.validate(-1)
        assert not p.validate(2.0)

    def test_int_validate(self):
        p = PluginParam("count", "Count", "int", 5, 1, 10)
        assert p.validate(5)
        assert not p.validate(0)

    def test_choice_validate(self):
        p = PluginParam("mode", "Mode", "choice", "a", choices=["a", "b", "c"])
        assert p.validate("a")
        assert not p.validate("d")

    def test_bool_validate(self):
        p = PluginParam("enable", "Enable", "bool", True)
        assert p.validate(True)
        assert p.validate(False)
        assert not p.validate("yes")


class TestPluginBase:
    def test_midi_plugin_type(self):
        assert DummyMidiPlugin.plugin_type == PluginType.MIDI_PROCESSOR

    def test_audio_plugin_type(self):
        assert DummyAudioPlugin.plugin_type == PluginType.AUDIO_EFFECT

    def test_exporter_type(self):
        assert DummyExporter.plugin_type == PluginType.EXPORTER

    def test_param_defaults(self):
        defaults = DummyMidiPlugin.param_defaults()
        assert defaults == {"amount": 50, "mode": "fast"}

    def test_process_callable(self):
        plugin = DummyMidiPlugin()
        result = plugin.process("/input.mid", "/output")
        assert result.endswith("result.mid")


class TestPluginRegistry:
    def test_register(self):
        reg = PluginRegistry()
        assert reg.register(DummyMidiPlugin)

    def test_register_duplicate(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        assert not reg.register(DummyMidiPlugin)

    def test_unregister(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        assert reg.unregister(DummyMidiPlugin)
        assert not reg.unregister(DummyMidiPlugin)

    def test_get_plugins_all(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        reg.register(DummyAudioPlugin)
        reg.register(DummyExporter)
        assert len(reg.get_plugins()) == 3

    def test_get_plugins_by_type(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        reg.register(DummyAudioPlugin)
        midi_plugins = reg.get_plugins(PluginType.MIDI_PROCESSOR)
        assert len(midi_plugins) == 1
        assert midi_plugins[0] is DummyMidiPlugin

    def test_get_plugins_by_category(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        reg.register(DummyAudioPlugin)
        test_plugins = reg.get_plugins(category="Test")
        assert len(test_plugins) == 1

    def test_get_plugin_by_name(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        result = reg.get_plugin("Test MIDI Plugin")
        assert result is DummyMidiPlugin

    def test_get_plugin_by_name_and_type(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        result = reg.get_plugin("Test MIDI Plugin", PluginType.MIDI_PROCESSOR)
        assert result is DummyMidiPlugin

    def test_get_plugin_not_found(self):
        reg = PluginRegistry()
        assert reg.get_plugin("Nonexistent") is None

    def test_get_categories(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        reg.register(DummyAudioPlugin)
        reg.register(DummyExporter)
        cats = reg.get_categories()
        assert "Test" in cats
        assert "Effects" in cats
        assert "Export" in cats

    def test_singleton(self):
        a = PluginRegistry()
        b = PluginRegistry()
        assert a is b

    def test_reset(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        PluginRegistry.reset()
        reg2 = PluginRegistry()
        assert len(reg2.get_plugins()) == 0

    def test_clear(self):
        reg = PluginRegistry()
        reg.register(DummyMidiPlugin)
        reg.clear()
        assert len(reg.get_plugins()) == 0

    def test_discover_directory(self, tmp_path):
        plugin_code = """
from korshmer.plugins.base import MidiProcessorPlugin, PluginParam

class MyTestPlugin(MidiProcessorPlugin):
    name = "File Discovered Plugin"
    category = "Discovered"

    def process(self, midi_path, output_dir, **params):
        return "done"
"""
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(plugin_code)

        reg = PluginRegistry()
        count = reg.discover_directory(str(tmp_path))
        assert count == 1

        plugins = reg.get_plugins(category="Discovered")
        assert len(plugins) == 1
        assert plugins[0].name == "File Discovered Plugin"

    def test_discover_directory_nonexistent(self):
        reg = PluginRegistry()
        count = reg.discover_directory("/nonexistent/path")
        assert count == 0

    def test_discover_directory_skips_private(self, tmp_path):
        (tmp_path / "_private.py").write_text("x = 1")
        (tmp_path / "__init__.py").write_text("")
        reg = PluginRegistry()
        count = reg.discover_directory(str(tmp_path))
        assert count == 0


class TestBuiltinPlugins:
    def test_register_builtins(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS, register_builtins

        reg = PluginRegistry()
        count = register_builtins(reg)
        assert count == len(BUILTIN_PLUGINS)

    def test_register_builtins_idempotent(self):
        from korshmer.plugins.builtin import register_builtins

        reg = PluginRegistry()
        n1 = register_builtins(reg)
        n2 = register_builtins(reg)
        assert n1 > 0
        assert n2 == 0

    def test_all_builtins_are_audio_effects(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS

        for cls in BUILTIN_PLUGINS:
            assert cls.plugin_type == PluginType.AUDIO_EFFECT

    def test_all_builtins_have_name(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS

        names = [cls.name for cls in BUILTIN_PLUGINS]
        assert len(names) == len(set(names))

    def test_all_builtins_have_category(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS

        for cls in BUILTIN_PLUGINS:
            assert cls.category
            assert cls.category != "General"

    def test_builtin_params_valid(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS

        for cls in BUILTIN_PLUGINS:
            params = cls.params()
            assert isinstance(params, list)
            for p in params:
                assert isinstance(p, PluginParam)
                assert p.validate(p.default)

    def test_builtin_param_defaults(self):
        from korshmer.plugins.builtin import BUILTIN_PLUGINS

        for cls in BUILTIN_PLUGINS:
            defaults = cls.param_defaults()
            assert isinstance(defaults, dict)
            for p in cls.params():
                assert p.name in defaults

    def test_builtin_categories(self):
        from korshmer.plugins.builtin import register_builtins

        reg = PluginRegistry()
        register_builtins(reg)
        cats = reg.get_categories(PluginType.AUDIO_EFFECT)
        assert "Modulation" in cats
        assert "Space" in cats
        assert "Dynamics" in cats

    def test_load_all_includes_builtins(self):
        reg = PluginRegistry()
        reg.load_all()
        plugins = reg.get_plugins(PluginType.AUDIO_EFFECT)
        names = {p.name for p in plugins}
        assert "Reverb" in names
        assert "Delay" in names
        assert "Phaser" in names
        assert "Smart Mastering" in names

    def test_reverb_plugin_process(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from korshmer.plugins.builtin import ReverbPlugin

        y = np.random.uniform(-0.5, 0.5, (2, 22050)).astype(np.float32)
        in_path = str(tmp_path / "input.wav")
        out_path = str(tmp_path / "output.wav")
        sf.write(in_path, y.T, 44100)

        plugin = ReverbPlugin()
        result = plugin.process(
            in_path, out_path, room_size=0.3, decay=0.3, wet=0.2
        )
        assert os.path.exists(result)
        data, sr = sf.read(result)
        assert sr == 44100
        assert len(data) > 0

    def test_normalize_plugin_process(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from korshmer.plugins.builtin import NormalizePlugin

        y = np.random.uniform(-0.1, 0.1, 4410).astype(np.float32)
        in_path = str(tmp_path / "input.wav")
        out_path = str(tmp_path / "output.wav")
        sf.write(in_path, y, 44100)

        plugin = NormalizePlugin()
        result = plugin.process(in_path, out_path)
        assert os.path.exists(result)

    def test_filter_plugin_process(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from korshmer.plugins.builtin import FilterPlugin

        y = np.random.uniform(-0.5, 0.5, (2, 22050)).astype(np.float32)
        in_path = str(tmp_path / "input.wav")
        out_path = str(tmp_path / "output.wav")
        sf.write(in_path, y.T, 44100)

        plugin = FilterPlugin()
        result = plugin.process(
            in_path, out_path, filter_type="high", cutoff=500
        )
        assert os.path.exists(result)
