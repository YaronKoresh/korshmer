import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from korshmer.utils.track_layout import DRUMS_NAME, PROGRAM_NAMES

if not QApplication.instance():
    QApplication(sys.argv)

try:
    from korshmer.__main__ import MainWindow
except Exception as import_error:
    MainWindow = None
    MAIN_IMPORT_ERROR = str(import_error)
else:
    MAIN_IMPORT_ERROR = ""


pytestmark = pytest.mark.integration


def _write_song(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    keys = pretty_midi.Instrument(program=0, name="Keys")
    for i in range(8):
        keys.notes.append(pretty_midi.Note(90, 60 + i, i * 0.5, i * 0.5 + 0.4))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for i in range(8):
        drums.notes.append(pretty_midi.Note(100, 36, i * 0.5, i * 0.5 + 0.1))
    pm.instruments.extend([keys, drums])
    path = tmp_path / "song.mid"
    pm.write(str(path))
    return str(path)


def _write_vocals(tmp_path):
    path = tmp_path / "vocals.wav"
    data = np.random.uniform(-0.2, 0.2, (44100, 1)).astype("float32")
    sf.write(str(path), data, 44100)
    return str(path)


@pytest.fixture
def window(monkeypatch):
    if MainWindow is None:
        pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: None)
    )
    win = MainWindow()
    yield win
    win.close()
    QApplication.processEvents()


class TestMainWindowEditorWiring:
    def test_builder_tab_is_gone(self, window):
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels[:2] == ["EDITOR", "THEORY"]
        assert "BUILDER" not in labels
        assert "STEMS" in labels

    def test_generators_category_holds_one_midi_generator(self, window):
        names = [
            window.list_categories.item(i).text()
            for i in range(window.list_categories.count())
        ]
        assert "MIDI Generator" in names
        assert "Generators & AI" not in names
        assert hasattr(window, "combo_gen_mode")
        assert not hasattr(window, "combo_ai_bars")

    def test_tempo_box_holds_fractions_and_reaches_the_editor(self, window):
        from PySide6.QtWidgets import QDoubleSpinBox

        assert isinstance(window.spin_bpm, QDoubleSpinBox)
        assert window.spin_bpm.decimals() == 2
        window.spin_bpm.setValue(123.45)
        assert window.spin_bpm.value() == pytest.approx(123.45)
        window._apply_tempo_update()
        assert window.arranger.piano_roll.tempo == pytest.approx(123.45)
        window._apply_osc_tempo(97.6)
        assert window.spin_bpm.value() == pytest.approx(97.6)
        window._apply_link_tempo(101.25)
        assert window.spin_bpm.value() == pytest.approx(101.25)
        window.spin_bpm.setValue(120.0)

    def test_midi_loads_as_lanes_and_clip_attaches(self, window, tmp_path):
        song = _write_song(tmp_path)
        vocals = _write_vocals(tmp_path)

        window.arranger.load_midi_file(song, replace=True, tempo=100.0)
        tracks = list(window.arranger.model.tracks.values())
        assert [t.name for t in tracks] == ["Keys", "Drums"]
        assert tracks[1].is_drum
        assert window.spin_bpm.value() == 100

        assert window.send_audio_to_editor(vocals) is True
        window.arranger.wait_for_waveforms()
        clips = window.arranger.get_audio_clips()
        assert len(clips) == 1
        assert clips[0]["path"] == vocals
        assert clips[0]["kind"] == "vocals"

        window.add_midi_lanes_to_editor(song)
        assert len(window.arranger.model.tracks) == 4
        exported = window.arranger.get_midi_from_editor()
        assert [i.name for i in exported.instruments] == ["Keys", "Drums"]
        assert sum(len(i.notes) for i in exported.instruments) == 16

    def test_editing_shortcut_handlers_round_trip(self, window, tmp_path):
        window.arranger.load_midi_file(_write_song(tmp_path), replace=True)
        before = window.arranger.model.note_count()

        window._handle_select_all()
        window.on_piano_roll_duplicate()
        assert window.arranger.model.note_count() == before * 2

        window.on_piano_roll_transpose_up()
        window.on_piano_roll_quantize()
        window.on_piano_roll_humanize()
        assert window.arranger.undo() is True
        window.editor_toggle_snap()
        assert window.arranger.piano_roll.snap_enabled is False
        window.editor_zoom(1.25)
        project = window.arranger.to_project_dict()
        assert {"tracks", "notes", "clips", "tempo"} <= set(project)

    def test_stem_session_loads_with_tempo(self, window, tmp_path):
        from korshmer.audio_processing.stem_session import StemSession

        song = _write_song(tmp_path)
        vocals = _write_vocals(tmp_path)
        session = StemSession(
            source_path=song,
            output_dir=str(tmp_path),
            tempo=98.0,
            stems={"vocals": vocals},
            midi_files=[song],
        )
        session.save()

        assert window.load_output_folder_session(str(tmp_path)) is True
        window.arranger.wait_for_waveforms()
        assert window.spin_bpm.value() == 98
        assert abs(window.arranger.piano_roll.tempo - 98.0) < 1e-6
        assert len(window.arranger.get_all_clips()) == 1
        assert len(window.arranger.model.tracks) == 2
        assert window.tabs.currentWidget() is window.arranger

    def test_generator_degrades_without_engine(self, window, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("korshmer.midi_generation"):
                raise ImportError("engine missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        window.run_midi_generator()
        assert window.arranger.model.note_count() == 0


class TestFullMixPreview:
    def test_full_mix_has_one_track_per_instrument(
        self, window, tmp_path, monkeypatch
    ):
        from korshmer import __main__ as main_module

        monkeypatch.setattr(main_module, "MidiPreviewThread", MagicMock())
        for name, pitch in (("piano.mid", 60), ("guitar.mid", 64)):
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(
                program=0, name="Acoustic Grand Piano"
            )
            inst.notes.append(pretty_midi.Note(90, pitch, 0.0, 0.5))
            pm.instruments.append(inst)
            pm.write(str(tmp_path / name))
        window.current_output_dir = str(tmp_path)

        window.play_full_mix_action()

        mixed = pretty_midi.PrettyMIDI(str(tmp_path / "full_mix_temp.mid"))
        assert [i.name for i in mixed.instruments] == ["Grand Piano"]
        assert len(mixed.instruments[0].notes) == 2


class TestMidiGeneratorInWindow:
    @pytest.fixture(autouse=True)
    def _data_dir(self, monkeypatch, tmp_path):
        from korshmer.mixins.actions import editor

        monkeypatch.setattr(editor, "DATA_DIR", str(tmp_path))

    @pytest.fixture
    def warnings(self, monkeypatch):
        shown = []
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            staticmethod(lambda *a, **k: shown.append(a[2])),
        )
        return shown

    def _setup(self, window, root="A", scale="Pentatonic Minor"):
        window.combo_root.setCurrentText(root)
        window.combo_scale.setCurrentText(scale)
        window.spin_bpm.setValue(120)
        window.spin_gen_minutes.setValue(0.5)
        window.spin_gen_seed.setValue(5)

    def test_generate_fills_the_editor_with_role_tracks_only(
        self, window, warnings
    ):
        self._setup(window)
        window.run_midi_generator()
        assert warnings == []
        model = window.arranger.model
        tracks = list(model.tracks.values())
        names = [t.name for t in tracks]
        assert DRUMS_NAME in names
        assert len(set(names)) == len(names)
        assert set(names) <= {DRUMS_NAME, *PROGRAM_NAMES}
        keys = [(t.is_drum, 0 if t.is_drum else t.program) for t in tracks]
        assert len(set(keys)) == len(keys)
        assert model.note_count() > 100
        assert window.tabs.currentWidget() is window.arranger

    def test_replace_clears_old_content_and_append_keeps_it(
        self, window, warnings
    ):
        self._setup(window)
        window.run_midi_generator()
        model = window.arranger.model
        first = len(model.tracks)
        first_notes = model.note_count()
        window.chk_gen_replace.setChecked(False)
        window.spin_gen_seed.setValue(6)
        window.run_midi_generator()
        keys = [
            (t.is_drum, 0 if t.is_drum else t.program)
            for t in model.tracks.values()
        ]
        assert len(set(keys)) == len(keys)
        assert len(model.tracks) >= first
        assert model.note_count() > first_notes
        window.chk_gen_replace.setChecked(True)
        window.spin_gen_seed.setValue(5)
        window.run_midi_generator()
        assert len(model.tracks) == first

    def test_auto_root_and_scale_are_resolved_from_the_editor(
        self, window, warnings
    ):
        self._setup(window, root="Auto", scale="Auto (Detect)")
        window.run_midi_generator()
        assert warnings == []
        assert window.arranger.model.note_count() > 0

    def test_continue_mode_adds_notes_after_the_existing_music(
        self, window, warnings, tmp_path
    ):
        self._setup(window, scale="Minor")
        window.load_midi_for_test = None
        window.arranger.load_midi_file(
            _write_song(tmp_path), replace=True, tempo=120
        )
        before = window.arranger.model.note_count()
        end_before = max(
            n.start + n.duration for n in window.arranger.model.notes
        )
        window.combo_gen_mode.setCurrentText("Continue editor")
        window.run_midi_generator()
        assert warnings == []
        added = window.arranger.model.notes[before:]
        assert added
        assert min(n.start for n in added) >= end_before - 1e-6

    def test_layer_mode_stays_inside_the_song(self, window, warnings, tmp_path):
        self._setup(window, scale="Minor")
        window.arranger.load_midi_file(
            _write_song(tmp_path), replace=True, tempo=120
        )
        before = window.arranger.model.note_count()
        end_before = max(
            n.start + n.duration for n in window.arranger.model.notes
        )
        window.combo_gen_mode.setCurrentText("Layer over editor")
        window.run_midi_generator()
        assert warnings == []
        added = window.arranger.model.notes[before:]
        assert added
        assert min(n.start for n in added) < end_before
        assert window.arranger.model.note_count() > before

    def test_layer_status_names_the_parts_of_the_song(
        self, window, warnings, tmp_path, monkeypatch
    ):
        from korshmer.mixins.actions import editor

        messages = []
        real = editor.publish

        def spy(event, **kwargs):
            messages.append(kwargs.get("text", ""))
            return real(event, **kwargs)

        monkeypatch.setattr(editor, "publish", spy)
        self._setup(window, scale="Minor")
        window.arranger.load_midi_file(
            _write_song(tmp_path), replace=True, tempo=120
        )
        window.combo_gen_mode.setCurrentText("Layer over editor")
        window.run_midi_generator()

        generated = [m for m in messages if m.startswith("Generated")]
        assert generated and " | " in generated[-1]

    def test_length_field_follows_the_generator_mode(self, window):
        window.combo_gen_mode.setCurrentText("Layer over editor")
        assert not window.spin_gen_minutes.isEnabled()
        assert not window.chk_gen_replace.isEnabled()
        window.combo_gen_mode.setCurrentText("New song")
        assert window.spin_gen_minutes.isEnabled()
        assert window.chk_gen_replace.isEnabled()

    def test_engine_managed_options_are_not_asked_from_the_user(self, window):
        for name in (
            "combo_gen_match",
            "combo_gen_structure",
            "combo_gen_texture",
        ):
            assert not hasattr(window, name)

    def test_no_roles_warns_and_changes_nothing(self, window, warnings):
        self._setup(window)
        for box in window.gen_role_boxes.values():
            box.setChecked(False)
        window.run_midi_generator()
        assert warnings and "role" in warnings[0]
        assert window.arranger.model.note_count() == 0


def _action_for(window, sequence):
    from PySide6.QtGui import QKeySequence

    wanted = QKeySequence(sequence)
    matches = [a for a in window.actions() if a.shortcut() == wanted]
    assert len(matches) == 1, f"expected one action for {sequence}"
    return matches[0]


class TestWindowUiSetup:
    def test_setup_lives_in_mixin(self):
        import korshmer.__main__ as main_module
        from korshmer.mixins import MainWindowUiSetupMixin

        if MainWindow is None:
            pytest.skip(f"MainWindow unavailable here: {MAIN_IMPORT_ERROR}")
        assert issubclass(MainWindow, MainWindowUiSetupMixin)
        for name in (
            "_setup_status_bar",
            "_setup_keyboard_shortcuts",
            "_run_editor_shortcut",
            "_setup_context_menus",
            "_setup_tooltips",
        ):
            assert name not in MainWindow.__dict__
            assert name in MainWindowUiSetupMixin.__dict__
        assert not hasattr(main_module, "QStatusBar")

    def test_status_bar_hosts_enhanced_status_bar(self, window):
        from korshmer.ui.widgets.statusbar import EnhancedStatusBar

        assert isinstance(window.enhanced_status_bar, EnhancedStatusBar)
        assert window.enhanced_status_bar.parent() is window.statusBar()
        assert window.statusBar().isHidden()
        window.enhanced_status_bar.update_action("Ready")
        assert not window.statusBar().isHidden()
        window.enhanced_status_bar.clear_action()
        assert window.statusBar().isHidden()

    def test_tooltips_and_context_menu(self, window):
        from PySide6.QtCore import Qt

        tips = window.tooltip_manager
        assert window.control_bar.btn_play.toolTip() == tips.get("play_pause")
        assert window.control_bar.btn_stop.toolTip() == tips.get("stop")
        assert window.spin_bpm.toolTip() == tips.get("bpm")
        assert (
            window.arranger.piano_roll.contextMenuPolicy()
            == Qt.CustomContextMenu
        )

    def test_each_editor_shortcut_runs_its_own_handler(
        self, window, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(
            window, "_run_editor_shortcut", lambda h: seen.append(h)
        )
        expected = {
            "Ctrl+D": window.on_piano_roll_duplicate,
            "Q": window.on_piano_roll_quantize,
            "U": window.on_piano_roll_humanize,
            "N": window.editor_toggle_snap,
            "Ctrl+B": window.run_midi_generator,
        }
        for sequence, handler in expected.items():
            seen.clear()
            _action_for(window, sequence).trigger()
            assert seen == [handler], sequence

    def test_editor_shortcut_ignored_in_line_edit(self, window, monkeypatch):
        from PySide6.QtWidgets import QLineEdit

        calls = []
        line_edit = QLineEdit()
        monkeypatch.setattr(window, "_editor_tab_active", lambda: True)
        monkeypatch.setattr(
            QApplication, "focusWidget", staticmethod(lambda: line_edit)
        )
        window._run_editor_shortcut(lambda: calls.append(1))
        assert calls == []

        monkeypatch.setattr(
            QApplication, "focusWidget", staticmethod(lambda: None)
        )
        window._run_editor_shortcut(lambda: calls.append(1))
        assert calls == [1]


class TestMainWindowAudioSetup:
    def test_all_audio_and_midi_processors_attached(self, window):
        from tests.unit.test_audio_setup import PROCESSOR_NAMES

        for name in PROCESSOR_NAMES:
            assert hasattr(window, name), name


class TestMainWindowHealthServer:
    def test_health_server_off_by_default(self, window):
        assert window.health_server is None

    def test_health_server_reports_window_state(self, monkeypatch, window):
        import json
        import urllib.request

        monkeypatch.setenv("KORSHMER_HEALTH_PORT", "0")
        window._start_health_server()
        server = window.health_server
        assert server is not None and server.running

        url = f"http://127.0.0.1:{server.port}/health/state"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["status"] == "ok"
        assert body["state"]["processing"] is False
        assert body["state"]["osc"] is False
        assert "memory_peak" in body["state"]["performance"]
        assert body["state"]["playback"]

        window.close()
        QApplication.processEvents()
        assert window.health_server is None
        assert not server.running
