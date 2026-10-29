import builtins
import sys

import pretty_midi
import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox, QSpinBox

if not QApplication.instance():
    QApplication(sys.argv)

from korshmer.mixins.actions.editor import EditorActionsMixin


class FakeSlider:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeArranger:
    def __init__(self):
        self.calls = []
        self.piano_roll = self
        self.snap_enabled = True
        self.midi = pretty_midi.PrettyMIDI(initial_tempo=120)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "get_midi_from_editor":
                return self.midi
            if name == "get_audio_clips":
                return []
            return None

        return record


class FakeTabs:
    def __init__(self, arranger):
        self.current = arranger

    def currentWidget(self):
        return self.current

    def setCurrentWidget(self, widget):
        self.current = widget


class Host(EditorActionsMixin):
    def __init__(self):
        self.arranger = FakeArranger()
        self.tabs = FakeTabs(self.arranger)
        self.combo_root = QComboBox()
        self.combo_root.addItems(["Auto", "C", "D", "A"])
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Auto (Detect)", "Major", "Minor"])
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(20, 300)
        self.spin_bpm.setValue(100)
        self.slider_quantize = FakeSlider(50)
        self.slider_human_time = FakeSlider(5)
        self.slider_human_vel = FakeSlider(12)
        self.spin_gen_seed = QSpinBox()
        self.spin_gen_seed.setRange(0, 99999)
        self.gen_role_boxes = {}


@pytest.fixture
def host():
    return Host()


def _calls(host, name):
    return [c for c in host.arranger.calls if c[0] == name]


class TestKeyResolution:
    def test_explicit_key_is_kept(self, host):
        host.combo_root.setCurrentText("D")
        host.combo_scale.setCurrentText("Minor")
        assert host._resolve_key(None) == ("D", "Minor")

    def test_auto_with_empty_editor_defaults_to_c_major(self, host):
        host.combo_root.setCurrentText("Auto")
        host.combo_scale.setCurrentText("Auto (Detect)")
        assert host._resolve_key(None) == ("C", "Major")

    def test_auto_root_keeps_explicit_scale(self, host):
        host.combo_root.setCurrentText("Auto")
        host.combo_scale.setCurrentText("Minor")
        root, scale = host._resolve_key(pretty_midi.PrettyMIDI())
        assert root == "C"
        assert scale == "Minor"


class TestEditorOperations:
    def test_quantize_uses_theory_strength(self, host):
        host.on_piano_roll_quantize()
        assert _calls(host, "quantize_selected") == [
            ("quantize_selected", (), {"strength": 0.5})
        ]

    def test_humanize_uses_dynamics_sliders(self, host):
        host.on_piano_roll_humanize()
        assert _calls(host, "humanize_selected") == [
            ("humanize_selected", (0.05, 12), {})
        ]

    def test_transpose_and_octave(self, host):
        host.on_piano_roll_transpose_up()
        host.on_piano_roll_transpose_down()
        host.on_piano_roll_octave_up()
        host.on_piano_roll_octave_down()
        amounts = [c[1][0] for c in _calls(host, "transpose_selected")]
        assert amounts == [1, -1, 12, -12]

    def test_clipboard_and_selection_ops(self, host):
        host.on_piano_roll_select_all()
        host.on_piano_roll_copy()
        host.on_piano_roll_paste()
        host.on_piano_roll_duplicate()
        host.on_piano_roll_delete()
        names = [c[0] for c in host.arranger.calls]
        assert names == [
            "select_all",
            "copy_selected",
            "paste",
            "duplicate_selected",
            "delete_selected",
        ]

    def test_snap_toggle_flips_state(self, host):
        host.editor_toggle_snap()
        assert _calls(host, "set_snap") == [("set_snap", (False,), {})]

    def test_zoom_forwards_factor(self, host):
        host.editor_zoom(1.25)
        assert _calls(host, "zoom") == [("zoom", (1.25, 1.0), {})]


class TestGeneratorHelpers:
    def test_seed_zero_picks_random_positive_seed(self, host):
        host.spin_gen_seed.setValue(0)
        seed = host._generator_seed()
        assert isinstance(seed, int)
        assert seed > 0
        assert host.spin_gen_seed.value() == seed

    def test_seed_value_is_respected(self, host):
        host.spin_gen_seed.setValue(42)
        assert host._generator_seed() == 42

    def test_roles_follow_the_checked_boxes(self, host):
        from PySide6.QtWidgets import QCheckBox

        for key, checked in (("drums", True), ("bass", False), ("pad", True)):
            box = QCheckBox(key)
            box.setChecked(checked)
            host.gen_role_boxes[key] = box
        assert host._generator_roles() == ("drums", "pad")

    def test_roles_are_empty_without_boxes(self, host):
        host.gen_role_boxes = {}
        assert host._generator_roles() == ()

    def test_bpm_falls_back_when_invalid(self, host):
        host.spin_bpm = None
        assert host._current_bpm() == 120.0

    def test_generator_without_engine_warns(self, host, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append(args)),
        )
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("korshmer.midi_generation"):
                raise ImportError("engine missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        host.run_midi_generator()
        assert len(warnings) == 1
        assert not _calls(host, "load_midi_file")
