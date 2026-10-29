import copy
import gc
import logging
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path

import pretty_midi
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from korshmer import __version__
from korshmer.audio_processors import render_editor_mix
from korshmer.constants import BASE_DIR, DATA_DIR, FLUID_DIR, SF3_PATH
from korshmer.core.audio_setup import init_audio_and_midi_processors
from korshmer.core.window_initializer import init_window_chrome
from korshmer.event_bus import Event, publish, subscribe
from korshmer.mixins.actions import ActionsMixin
from korshmer.mixins.files import FilesMixin
from korshmer.mixins.performance_mode import PerformanceModeMixin
from korshmer.mixins.presets_window import PresetsMixin
from korshmer.mixins.ui import UiMixin
from korshmer.mixins.window_ui_setup import MainWindowUiSetupMixin
from korshmer.network.health_server import HealthServer, port_from_env
from korshmer.network.link_sync import LinkSync, link_available
from korshmer.network.obs_client import (
    DEFAULT_URL as OBS_DEFAULT_URL,
    ObsClient,
)
from korshmer.network.osc_server import OscServer
from korshmer.project_management import (
    AnalyticsManager,
    AssetManager,
    CloudSyncManager,
    CollaborationManager,
    ExportManager,
    ImportManager,
    ProjectManager,
    VersionControlManager,
)
from korshmer.track_processor import process_track
from korshmer.ui.animations import BpmAnimator
from korshmer.ui.managers import (
    AccessibilityManager,
    ContextMenuManager,
    KeyboardShortcutManager,
    PerformanceMonitor,
    TooltipManager,
    TutorialManager,
    UndoRedoManager,
    WorkspaceManager,
)
from korshmer.ui.managers.accessibility import apply_accessible_names
from korshmer.ui.managers.theme_initializer import init_theme_manager
from korshmer.ui.styling import (
    MAT_BG,
    gradient,
)
from korshmer.ui.theming.color_utils import (
    BEAT_INDICATOR_OFF,
    BEAT_INDICATOR_ON,
)
from korshmer.utils import initialize_logging, setup_event_loop
from korshmer.utils.crash_reporter import install_crash_handler
from korshmer.utils.credential_store import get_secret
from korshmer.utils.filesystem import is_midi_path
from korshmer.utils.log_stream import install_log_stream, remove_log_stream
from korshmer.utils.tempo import TapTempo, validate_bpm
from korshmer.utils.track_layout import merge_by_instrument
from korshmer.workers.generic import WorkerThread
from korshmer.workers.midi_preview import MidiPreviewThread

setup_event_loop()
initialize_logging()


class MainWindow(
    QMainWindow,
    UiMixin,
    MainWindowUiSetupMixin,
    FilesMixin,
    PresetsMixin,
    ActionsMixin,
    PerformanceModeMixin,
):
    link_tempo_received = Signal(float)
    osc_play = Signal()
    osc_pause = Signal()
    osc_stop = Signal()
    osc_tempo = Signal(float)
    obs_status = Signal(str)
    obs_connected = Signal(bool)

    def __init__(self):
        super().__init__()
        self.is_processing = False
        self.link_sync = None
        self.osc_server = None
        self.health_server = None
        self.obs_client = None
        self._obs_signals_connected = False
        self._obs_playback_connected = False
        self._applying_link_tempo = False
        init_window_chrome(self)

        self.file_list_paths = []
        self.generated_files_map = {}
        self.setAcceptDrops(True)
        self.presets = self.load_presets()
        self.current_output_dir = DATA_DIR

        self.bpm_update_timer = QTimer()
        self.bpm_update_timer.setSingleShot(True)
        self.bpm_update_timer.timeout.connect(self._apply_tempo_update)
        self.pending_bpm_value = None
        self.bpm_animator = None
        self.tap_tempo = TapTempo()
        self.beat_timer = QTimer(self)
        self.beat_timer.timeout.connect(self._on_beat)
        self.beat_off_timer = QTimer(self)
        self.beat_off_timer.setSingleShot(True)
        self.beat_off_timer.timeout.connect(self._on_beat_off)

        self.setup_ui()
        self._restart_beat_timer(float(self.spin_bpm.value()))
        self._subscribe_events()
        self.load_input_list()
        self.load_output_view()

        current_vol = self.control_bar.player.volume()
        self.visualizer.set_volume_level(current_vol)

        self.apply_preset("Default")
        self.sync_convert_button()

        self.setMouseTracking(True)

        self.title_bar.installEventFilter(self)
        self.centralWidget().installEventFilter(self)

        self.keyboard_shortcuts = KeyboardShortcutManager(data_dir=DATA_DIR)
        self.theme_manager = init_theme_manager(DATA_DIR)
        self.undo_redo = UndoRedoManager()
        self.workspace_manager = WorkspaceManager(data_dir=DATA_DIR)
        self.performance_monitor = PerformanceMonitor()
        self.context_menu_manager = ContextMenuManager()
        self.tooltip_manager = TooltipManager()
        self.tutorial_manager = TutorialManager(self)
        self.accessibility = AccessibilityManager()

        self.project_manager = ProjectManager(BASE_DIR)
        self.asset_manager = AssetManager.get_instance(BASE_DIR)
        self.export_manager = ExportManager(BASE_DIR)
        self.import_manager = ImportManager(BASE_DIR)
        self.version_control = VersionControlManager(BASE_DIR)
        self.cloud_sync = CloudSyncManager(BASE_DIR)
        self.collaboration = CollaborationManager(BASE_DIR)
        self.analytics = AnalyticsManager(BASE_DIR)

        init_audio_and_midi_processors(self)

        self._check_auto_recovery()

        self.setup_window_ui_extras()
        self._start_performance_monitoring()
        self._start_health_server()
        self._start_auto_backup()
        self._start_recovery_timer()

        self.workspace_manager._load_from_disk()
        apply_accessible_names(self)

    def _subscribe_events(self):
        subscribe(Event.STATUS_UPDATE, self._on_status_update)
        subscribe(Event.AUDIO_LOAD_REQUESTED, self._on_audio_load)
        subscribe(Event.PROGRESS_UPDATE, self._on_progress_update)
        subscribe(Event.CONVERT_BUTTON_STATE, self._on_convert_button_state)
        subscribe(Event.CONVERT_BUTTON_TEXT, self._on_convert_button_text)
        subscribe(Event.CONVERT_BUTTON_ROLE, self._on_convert_button_role)
        subscribe(Event.PROGRESS_BAR_STATE, self._on_progress_bar_state)
        subscribe(Event.UI_RESET, self._on_ui_reset)
        subscribe(Event.OUTPUT_VIEW_REFRESH, self._on_output_refresh)
        subscribe(Event.FILE_LOCK_RELEASED, self._on_file_lock_released)
        subscribe(Event.EDITOR_ADD_AUDIO_CLIP, self._on_editor_add_clip)
        subscribe(Event.EDITOR_LOAD_MIDI, self._on_editor_load_midi)
        subscribe(Event.STEMS_READY, self._on_stems_ready)

    def _on_status_update(self, text=""):
        self.control_bar.lbl_status.setText(text)

    def _on_audio_load(self, path="", analyze=True, load_player=True):
        if analyze:
            if hasattr(self, "visualizer"):
                self.visualizer.analyze_audio(path)
            if hasattr(self, "spectrogram"):
                self.spectrogram.set_audio(path)
        if load_player:
            self.control_bar.load_file(path)

    def _on_progress_update(self, value=0, text=""):
        self.control_bar.lbl_status.setText(text)
        if hasattr(self.control_bar, "progress_bar"):
            self.control_bar.progress_bar.setValue(value)

    def _on_convert_button_state(self, enabled=True):
        self.control_bar.btn_convert.setEnabled(enabled)

    def _on_convert_button_text(self, text=""):
        self.control_bar.btn_convert.setText(text)

    def _on_convert_button_role(self, role="primary"):
        self.control_bar.btn_convert.set_role(role)

    def _on_progress_bar_state(self, visible=False, value=0):
        if hasattr(self.control_bar, "progress_bar"):
            self.control_bar.progress_bar.setValue(value)
            self.control_bar.progress_bar.setVisible(visible)

    def _on_ui_reset(self):
        self.reset_ui_state()

    def _on_output_refresh(self, output_dir=None):
        if output_dir is not None:
            self.current_output_dir = output_dir
        self.load_output_view()

    def _on_file_lock_released(self):
        self.release_file_lock()

    def _on_editor_add_clip(self, path="", start=0.0, name=None, kind="audio"):
        if not path or not hasattr(self, "arranger"):
            return
        try:
            self.arranger.add_audio_clip(path, start, name=name, kind=kind)
            self.tabs.setCurrentWidget(self.arranger)
        except Exception as e:
            logging.error(f"Could not add audio clip: {e}")

    def _on_editor_load_midi(self, path="", replace=False):
        if not path or not hasattr(self, "arranger"):
            return
        try:
            self.arranger.load_midi_file(
                path, replace=replace, tempo=float(self.spin_bpm.value())
            )
            self.tabs.setCurrentWidget(self.arranger)
        except Exception as e:
            logging.error(f"Could not load MIDI into editor: {e}")

    def _on_stems_ready(self, session=None):
        if session is None:
            return
        stems = ", ".join(sorted(session.stems.keys()))
        publish(Event.STATUS_UPDATE, text=f"Stems ready: {stems}")

    def _start_performance_monitoring(self):
        self.perf_timer = QTimer(self)
        self.perf_timer.timeout.connect(self._update_performance_display)
        self.perf_timer.start(500)

    def _update_performance_display(self):
        self.performance_monitor.update()
        self._beat_health_server()

    def _start_health_server(self):
        port = port_from_env(os.environ)
        if port is None:
            return
        log_stream = install_log_stream()
        server = HealthServer(port=port, log_stream=log_stream)
        if not server.start():
            remove_log_stream(log_stream)
            publish(
                Event.STATUS_UPDATE,
                text=f"Health check could not listen on port {port}.",
            )
            return
        self.health_server = server
        self._beat_health_server()

    def _beat_health_server(self):
        server = self.health_server
        if server is None:
            return
        try:
            player = self.control_bar.player
            playback = getattr(
                player.playbackState(), "name", str(player.playbackState())
            )
        except Exception:
            playback = "unknown"
        server.beat(
            processing=bool(self.is_processing),
            playback=playback,
            performance=self.performance_monitor.get_summary(),
            osc=self.osc_server is not None and self.osc_server.running,
            obs=self.obs_client is not None and self.obs_client.connected,
            link=self.link_sync is not None and self.link_sync.enabled,
        )

    def _start_auto_backup(self):
        self.backup_timer = QTimer(self)
        self.backup_timer.timeout.connect(self._perform_auto_backup)
        self.backup_timer.start(60000)

    def _perform_auto_backup(self):
        if not self.project_manager.backup_enabled:
            return

        try:
            success = self.project_manager.create_backup()
            if success:
                logging.info("Auto-backup completed successfully")
        except Exception as e:
            logging.error(f"Auto-backup failed: {e}")

    def _start_recovery_timer(self):
        self.recovery_timer = QTimer(self)
        self.recovery_timer.timeout.connect(self._save_recovery_state)
        self.recovery_timer.start(30000)

    def _save_recovery_state(self):
        if self.project_manager.current_project_path:
            try:
                self.project_manager.save_recovery_state()
            except Exception as e:
                logging.error(f"Failed to save recovery state: {e}")

    def _check_auto_recovery(self):
        recovery_data = self.project_manager.enable_auto_recovery()
        if recovery_data:
            reply = QMessageBox.question(
                self,
                "Auto-Recovery",
                f"A recovery file was found from {recovery_data.get('timestamp')}.\nWould you like to recover the unsaved changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                try:
                    self.project_manager.current_project_data = (
                        recovery_data.get("project_data", {})
                    )
                    project_path = recovery_data.get("project_path")
                    if project_path:
                        self.project_manager.current_project_path = Path(
                            project_path
                        )
                    logging.info("Project recovered from auto-recovery file")
                    QMessageBox.information(
                        self,
                        "Recovery Complete",
                        "Your project has been recovered successfully.",
                    )
                    self.project_manager.clear_recovery_state()
                except Exception as e:
                    logging.error(f"Recovery failed: {e}")
                    QMessageBox.warning(
                        self,
                        "Recovery Failed",
                        f"Failed to recover project: {e}",
                    )
            else:
                logging.info("User declined auto-recovery")

    def _handle_play_pause(self):
        if hasattr(self, "control_bar") and hasattr(self.control_bar, "player"):
            if self.control_bar.player.is_playing:
                self.control_bar.player.pause()
            else:
                self.control_bar.player.play()

    def _handle_undo(self):
        if self._editor_tab_active() and self.arranger.undo():
            if hasattr(self, "enhanced_status_bar"):
                self.enhanced_status_bar.update_action("Undo: editor edit")
            return
        action = self.undo_redo.undo()
        if action:
            logging.debug(f"Undo: {action.get('description', 'Unknown')}")
            if hasattr(self, "enhanced_status_bar"):
                self.enhanced_status_bar.update_action(
                    f"Undo: {action.get('description', '')}"
                )

    def _handle_redo(self):
        if self._editor_tab_active() and self.arranger.redo():
            if hasattr(self, "enhanced_status_bar"):
                self.enhanced_status_bar.update_action("Redo: editor edit")
            return
        action = self.undo_redo.redo()
        if action:
            logging.debug(f"Redo: {action.get('description', 'Unknown')}")
            if hasattr(self, "enhanced_status_bar"):
                self.enhanced_status_bar.update_action(
                    f"Redo: {action.get('description', '')}"
                )

    def _handle_save(self):
        if hasattr(self, "save_project"):
            self.save_project()

    def _handle_select_all(self):
        if self._editor_tab_active():
            self.arranger.select_all()

    def _show_help(self):
        help_text = f"""
KORSHMER v{__version__} - Keyboard Shortcuts

Playback:
  Space - Play/Pause
  Escape - Stop
  R - Record

Editing (EDITOR tab):
  Ctrl+Z - Undo
  Ctrl+Y / Ctrl+Shift+Z - Redo
  Ctrl+A - Select All
  Ctrl+D - Duplicate Selection
  Delete - Delete Selection
  Ctrl+C - Copy
  Ctrl+V - Paste
  Up / Down - Transpose 1 semitone
  Shift+Up / Shift+Down - Transpose octave
  Left / Right - Nudge by grid step
  Click empty space - Add note on the active lane
  Ctrl+Wheel over note - Change velocity
  Drag audio/MIDI from the output list - Add clip / lanes

Tools:
  Q - Quantize Selection
  U - Humanize Selection
  N - Toggle Snap
  Ctrl+B - Generate MIDI (MIDI Generator settings)

View:
  Ctrl++ - Zoom In
  Ctrl+- - Zoom Out
  H - Zoom Horizontal
  V - Zoom Vertical
  F11 - Fullscreen
  F10 - Performance Mode

File:
  Ctrl+N - New Project
  Ctrl+O - Open Project
  Ctrl+S - Save Project
  Ctrl+E - Export

Help:
  F1 - Show This Help
"""
        QMessageBox.information(self, "Keyboard Shortcuts", help_text)

    def start_tutorial_action(self):
        try:
            if hasattr(self, "tutorial_manager"):
                self.tutorial_manager.start_tutorial("intro")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Tutorial Error",
                f"Failed to start tutorial: {str(e)}\n\nPlease try again or consult the Help menu.",
            )

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_piano_roll_context_menu(self, position):
        menu = self.context_menu_manager.create_piano_roll_menu(self)
        menu.exec(self.arranger.piano_roll.mapToGlobal(position))

    def record_undo_action(self, description, data):
        self.undo_redo.push(
            {"description": description, "data": data, "timestamp": time.time()}
        )

    def _stop_background_threads(self, timeout_ms=3000):
        for name in (
            "preview_thread",
            "worker",
            "stem_thread",
            "export_thread",
            "chord_thread",
        ):
            thread = getattr(self, name, None)
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    if hasattr(thread, "stop"):
                        thread.stop()
                    if not thread.wait(timeout_ms):
                        thread.terminate()
                        thread.wait(1000)
            except RuntimeError as e:
                logging.debug(f"Thread {name} already gone: {e}")
        if hasattr(self, "arranger") and hasattr(
            self.arranger, "wait_for_waveforms"
        ):
            self.arranger.wait_for_waveforms(timeout_ms)

    def closeEvent(self, event):
        try:
            if hasattr(self, "control_bar"):
                self.control_bar.player.stop()
        except Exception as e:
            logging.debug(f"Player stop on close failed: {e}")
        for timer_name in (
            "perf_timer",
            "backup_timer",
            "recovery_timer",
            "bpm_update_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                timer.stop()
        hotplug = getattr(self, "midi_hotplug", None)
        if hotplug is not None:
            hotplug.stop()
        if self.link_sync is not None:
            self.link_sync.stop()
        if self.osc_server is not None:
            self.osc_server.stop()
        if self.health_server is not None:
            self.health_server.stop()
            if self.health_server.log_stream is not None:
                remove_log_stream(self.health_server.log_stream)
            self.health_server = None
        if self.obs_client is not None:
            self.obs_client.stop()
        self._stop_background_threads()
        super().closeEvent(event)

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)

            if (
                not self.metal_texture
                or self.metal_texture.size() != self.size()
            ):
                self.metal_texture = QImage(self.size(), QImage.Format_ARGB32)
                with QPainter(self.metal_texture) as tex_painter:
                    vertical_grad = gradient(
                        MAT_BG, MAT_BG, MAT_BG, MAT_BG, deg=0
                    )
                    radial_grad = gradient(MAT_BG, MAT_BG)

                    tex_painter.fillRect(self.rect(), vertical_grad)
                    tex_painter.fillRect(self.rect(), radial_grad)

            painter.drawImage(0, 0, self.metal_texture)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.HoverMove, QEvent.MouseMove):
            if self.moving_window:
                if self.drag_pos:
                    self.move(event.globalPosition().toPoint() - self.drag_pos)
                event.accept()
                return True

            global_pos = QCursor.pos()
            local_pos = self.mapFromGlobal(global_pos)

            if self.resize_direction:
                self._handle_resize_move(global_pos)
                return True

            is_on_edge = self._check_cursor_shape(local_pos)

            if is_on_edge:
                return True

        elif (
            event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
        ):
            global_pos = event.globalPosition().toPoint()
            local_pos = self.mapFromGlobal(global_pos)
            direction = self._get_resize_direction(local_pos)

            if direction:
                self.resize_direction = direction
                self.drag_pos = global_pos
                self.start_geometry = self.geometry()
                return True

        elif event.type() == QEvent.MouseButtonRelease:
            if self.resize_direction:
                self.resize_direction = None
                self.start_geometry = None
                self.drag_pos = None
                self.unsetCursor()
                return True

        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            child = self.childAt(event.position().toPoint())
            if child:
                curr = child
                while curr:
                    if curr == self.title_bar:
                        self.moving_window = True
                        self.drag_pos = (
                            event.globalPosition().toPoint()
                            - self.frameGeometry().topLeft()
                        )
                        event.accept()
                        return
                    curr = curr.parentWidget()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.moving_window and self.drag_pos:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.moving_window = False
        self.drag_pos = None
        super().mouseReleaseEvent(event)

    def _check_cursor_shape(self, pos):
        direction = self._get_resize_direction(pos)

        if direction:
            if direction == "top_left" or direction == "bottom_right":
                self.setCursor(Qt.SizeFDiagCursor)
            elif direction == "top_right" or direction == "bottom_left":
                self.setCursor(Qt.SizeBDiagCursor)
            elif direction == "top" or direction == "bottom":
                self.setCursor(Qt.SizeVerCursor)
            elif direction == "left" or direction == "right":
                self.setCursor(Qt.SizeHorCursor)
            return True

        else:
            if self.cursor().shape() in (
                Qt.SizeVerCursor,
                Qt.SizeHorCursor,
                Qt.SizeFDiagCursor,
                Qt.SizeBDiagCursor,
            ):
                self.unsetCursor()
            return False

    def _get_resize_direction(self, pos):
        w, h = self.width(), self.height()
        m = self.resize_margin
        x, y = pos.x(), pos.y()

        on_left = x < m
        on_right = x > w - m
        on_top = y < m
        on_bottom = y > h - m

        if on_top and on_left:
            return "top_left"
        if on_top and on_right:
            return "top_right"
        if on_bottom and on_left:
            return "bottom_left"
        if on_bottom and on_right:
            return "bottom_right"
        if on_top:
            return "top"
        if on_bottom:
            return "bottom"
        if on_left:
            return "left"
        if on_right:
            return "right"
        return None

    def _handle_resize_move(self, global_pos):
        if not self.start_geometry or not self.drag_pos:
            self.resize_direction = None
            return

        diff = global_pos - self.drag_pos
        dx, dy = diff.x(), diff.y()

        g = self.start_geometry
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        dir = self.resize_direction

        min_w = self.minimumWidth()
        min_h = self.minimumHeight()

        if "right" in dir:
            w += dx
        if "bottom" in dir:
            h += dy
        if "left" in dir:
            new_w = w - dx
            if new_w >= min_w:
                x += dx
                w = new_w
            else:
                right_edge = x + w
                w = min_w
                x = right_edge - min_w

        if "top" in dir:
            new_h = h - dy
            if new_h >= min_h:
                y += dy
                h = new_h
            else:
                bottom_edge = y + h
                h = min_h
                y = bottom_edge - min_h

        w = max(w, min_w)
        h = max(h, min_h)

        self.setGeometry(x, y, w, h)

    def start_conversion_logic(self, paths=None):
        target_paths = list(paths) if paths else list(self.file_list_paths)
        if not target_paths:
            QMessageBox.warning(
                self, "No Files", "Please add input audio files first."
            )
            return

        from korshmer.audio_processing.music_regions import MusicRegionOptions
        from korshmer.audio_processing.separation_engine import (
            SeparationOptions,
        )

        self.is_processing = True
        publish(Event.CONVERT_BUTTON_TEXT, text="Stop Processing")
        publish(Event.CONVERT_BUTTON_ROLE, role="danger")
        publish(Event.PROGRESS_BAR_STATE, visible=True, value=0)

        s = []
        if self.btn_drums.isChecked():
            s.append("drums")
        if self.btn_bass.isChecked():
            s.append("bass")
        if self.btn_other.isChecked():
            s.append("other")
        if self.btn_vocals.isChecked():
            s.append("vocals")

        if self.btn_harmonic.isChecked():
            s.append("guitar")
            s.append("piano")

        v_mode_ui = self.combo_velocity.currentText()
        v_mode = "Original"
        if v_mode_ui == "Fixed":
            v_mode = "Fixed"
        elif v_mode_ui == "Compressed":
            v_mode = "Compressed"

        raw_v = self.slider_vel_val.value()
        v_val = int((raw_v / 100) * 127)

        oct_val = self.combo_octave.currentText()
        if oct_val == "Original":
            oct_val = "Original"

        root_val = self.combo_root.currentText()
        if root_val == "Auto":
            root_val = "Auto"

        scale_val = self.combo_scale.currentText()
        if scale_val == "Auto (Detect)":
            scale_val = "Auto (Detect)"

        params = {
            "octave": oct_val,
            "root": root_val,
            "scale": scale_val,
            "grid": self.combo_grid.currentText(),
            "quantize": self.slider_quantize.value() / 100.0,
            "swing": self.slider_swing.value() / 100.0,
            "vel_mode": v_mode,
            "vel_val": v_val,
            "stems": s,
            "export": "Merged" if self.rad_merged.isChecked() else "Separate",
            "cc_gen": self.chk_cc.isChecked(),
            "gm_drums": self.chk_gm.isChecked(),
            "wav32": self.chk_wav32.isChecked(),
            "music_options": MusicRegionOptions(
                enabled=self.chk_music_only.isChecked()
            ),
            "separation_options": SeparationOptions(
                quality=self.combo_separation.currentText().lower()
            ),
        }

        self.worker = WorkerThread(
            target_paths, params, process_track_func=process_track
        )
        self.worker.progress_update.connect(self.update_progress_ui)
        self.worker.process_finished.connect(self.batch_done)
        self.worker.start()

    def stop_conversion(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            publish(Event.CONVERT_BUTTON_STATE, enabled=False)
            publish(Event.CONVERT_BUTTON_TEXT, text="Stopping...")
            self.worker.stop()
            self.worker.finished.connect(self.reset_ui_state)
        else:
            self.reset_ui_state()

    def toggle_processing(self):
        if self.is_processing:
            self.stop_conversion()
        else:
            self.start_conversion_logic()

    def start_processing(self):
        self.toggle_processing()

    def update_progress_ui(self, value, msg):
        publish(Event.PROGRESS_UPDATE, value=value, text=msg)

    def on_playback_state_changed(self, state):
        is_playing = state == 1
        if hasattr(self, "visualizer"):
            self.visualizer.set_playing(is_playing)
        if hasattr(self, "control_bar"):
            self.control_bar.btn_play.setText(
                "❚❚\ufe0e" if is_playing else "▶\ufe0e"
            )

        if is_playing:
            path = getattr(self.control_bar.player, "current_path", None)
            if (
                path
                and os.path.exists(path)
                and path.lower().endswith(
                    (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
                )
            ):
                needs_spectrogram = (
                    self.spectrogram.spectrogram_image is None
                    and not self.spectrogram.loading
                )
                needs_waveform = self.visualizer.rms_data is None
                if needs_spectrogram or needs_waveform:
                    publish(
                        Event.AUDIO_LOAD_REQUESTED,
                        path=path,
                        analyze=True,
                        load_player=False,
                    )

    def handle_editor_play_request(self, midi_obj):
        try:
            target_bpm = float(self.spin_bpm.value())
        except (ValueError, AttributeError):
            target_bpm = 120.0

        if target_bpm <= 0:
            target_bpm = 120.0

        play_midi = pretty_midi.PrettyMIDI(initial_tempo=target_bpm)

        for inst in midi_obj.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name
            )
            new_inst.notes = copy.deepcopy(inst.notes)
            new_inst.control_changes = copy.deepcopy(inst.control_changes)
            play_midi.instruments.append(new_inst)

        notes_count = sum(len(i.notes) for i in play_midi.instruments)
        clips = []
        if hasattr(self.arranger, "get_audio_clips"):
            clips = self.arranger.get_audio_clips()

        if notes_count > 0 and not os.path.exists(SF3_PATH):
            QMessageBox.warning(
                self,
                "Missing SoundFont",
                f"Please put 'soundfont.sf3' in:\n{FLUID_DIR}"
                + ("\nOnly audio clips will play." if clips else ""),
            )
            if not clips:
                return

        if notes_count > 0 or clips:
            publish(
                Event.STATUS_UPDATE,
                text=f"Rendering Editor at {target_bpm} BPM...",
            )
            publish(Event.CONVERT_BUTTON_STATE, enabled=False)

            self.arranger.piano_roll.playback_offset = 0

            temp_mid = os.path.join(DATA_DIR, "editor_temp.mid")
            play_midi.write(temp_mid)
            preview_wav = os.path.join(DATA_DIR, "editor_preview.wav")

            if (
                hasattr(self, "preview_thread")
                and self.preview_thread.isRunning()
            ):
                self.preview_thread.terminate()

            self.preview_thread = MidiPreviewThread(
                temp_mid,
                FLUID_DIR,
                SF3_PATH,
                preview_wav,
                bpm=target_bpm,
                render_func=partial(render_editor_mix, clips=clips),
            )

            def on_ready(wav):
                publish(Event.CONVERT_BUTTON_STATE, enabled=True)
                publish(
                    Event.AUDIO_LOAD_REQUESTED,
                    path=wav,
                    analyze=True,
                    load_player=True,
                )
                self.control_bar.player.setPosition(0)
                self.control_bar.player.play()
                clip_info = f" + {len(clips)} clip(s)" if clips else ""
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Playing at {target_bpm} BPM{clip_info}",
                )

            def on_fail(err):
                publish(Event.CONVERT_BUTTON_STATE, enabled=True)
                publish(Event.STATUS_UPDATE, text="Editor Render Failed")
                QMessageBox.critical(
                    self,
                    "Render Error",
                    f"Could not render editor audio:\n{err}",
                )

            self.preview_thread.preview_ready.connect(on_ready)
            self.preview_thread.preview_failed.connect(on_fail)
            self.preview_thread.start()
        else:
            publish(Event.STATUS_UPDATE, text="Editor is empty.")

    def play_full_mix_action(self):
        if not self.current_output_dir or not os.path.exists(
            self.current_output_dir
        ):
            return QMessageBox.warning(
                self, "Error", "No output directory active."
            )

        files = [
            f for f in os.listdir(self.current_output_dir) if is_midi_path(f)
        ]
        if not files:
            return QMessageBox.warning(
                self, "Error", "No MIDI files found in current output."
            )

        publish(Event.STATUS_UPDATE, text=f"Mixing {len(files)} tracks...")
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)

        target_bpm = float(self.spin_bpm.value())
        combined = pretty_midi.PrettyMIDI(initial_tempo=target_bpm)
        for f in files:
            try:
                p = pretty_midi.PrettyMIDI(
                    os.path.join(self.current_output_dir, f)
                )

                for inst in p.instruments:
                    combined.instruments.append(inst)
            except Exception:
                pass

        mix_mid = os.path.join(self.current_output_dir, "full_mix_temp.mid")
        combined.instruments = merge_by_instrument(
            combined.instruments, keep_shared_names=True
        )
        combined.write(mix_mid)
        mix_wav = os.path.join(self.current_output_dir, "full_mix_preview.wav")

        if hasattr(self, "preview_thread") and self.preview_thread.isRunning():
            self.preview_thread.terminate()

        try:
            target_bpm = float(self.spin_bpm.value())
        except (ValueError, AttributeError):
            target_bpm = 120.0

        if target_bpm <= 0:
            target_bpm = 120.0

        self.preview_thread = MidiPreviewThread(
            mix_mid,
            FLUID_DIR,
            SF3_PATH,
            mix_wav,
            bpm=target_bpm,
            render_func=render_editor_mix,
        )

        def on_mix_ready(wav):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.OUTPUT_VIEW_REFRESH)
            publish(
                Event.AUDIO_LOAD_REQUESTED,
                path=wav,
                analyze=True,
                load_player=True,
            )
            self.control_bar.player.play()
            publish(Event.STATUS_UPDATE, text="Playing Full Mix")

        def on_mix_fail(err):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.STATUS_UPDATE, text="Full Mix Render Failed")
            QMessageBox.critical(
                self,
                "Render Error",
                f"Could not render full mix audio:\n{err}",
            )

        self.preview_thread.preview_ready.connect(on_mix_ready)
        self.preview_thread.preview_failed.connect(on_mix_fail)
        self.preview_thread.start()

    def on_volume_changed(self, volume):
        if hasattr(self, "visualizer"):
            self.visualizer.set_volume_level(volume)

    def batch_done(self, generated_files):
        try:
            self.reset_ui_state()
            publish(Event.OUTPUT_VIEW_REFRESH, output_dir=DATA_DIR)

            if generated_files:
                count = len(generated_files)
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Completed: {count} files generated",
                )
                session = self._find_generated_session(generated_files)
                if session is not None and hasattr(
                    self, "load_session_into_editor"
                ):
                    reply = QMessageBox.question(
                        self,
                        "Success",
                        f"Processing completed.\n{count} files generated.\n\n"
                        "Load the result into the editor now?\n"
                        "(vocals as a draggable audio clip, instruments as "
                        "MIDI lanes)",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        self.load_session_into_editor(session, replace=True)
                else:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Processing completed successfully.\n{count} files generated in the output folder.",
                    )
            else:
                publish(Event.STATUS_UPDATE, text="Process finished.")
                if self.is_processing:
                    QMessageBox.warning(
                        self,
                        "Info",
                        "Process finished but no files were returned. Check settings or input files.",
                    )
        finally:
            gc.collect()

    def _find_generated_session(self, generated_files):
        try:
            from korshmer.audio_processing.stem_session import StemSession
        except ImportError:
            return None
        seen = set()
        for path in generated_files:
            folder = os.path.dirname(str(path))
            if folder in seen:
                continue
            seen.add(folder)
            try:
                session = StemSession.find(folder)
            except Exception as e:
                logging.debug(f"Session lookup failed for {folder}: {e}")
                session = None
            if session is not None:
                return session
        return None

    def reset_ui_state(self):
        self.is_processing = False
        publish(Event.CONVERT_BUTTON_TEXT, text="Start Processing")
        publish(Event.CONVERT_BUTTON_STATE, enabled=True)
        publish(Event.CONVERT_BUTTON_ROLE, role="primary")
        publish(Event.PROGRESS_BAR_STATE, visible=False)

    def update_piano_scale_view(self):
        root = self.combo_root.currentText()
        scale = self.combo_scale.currentText()

        if hasattr(self, "arranger") and hasattr(self.arranger, "piano_roll"):
            self.arranger.piano_roll.set_scale_view(root, scale)

    def set_link_enabled(self, enabled):
        if not enabled:
            if self.link_sync is not None:
                self.link_sync.stop()
            return

        if self.link_sync is None:
            if not link_available():
                self._reject_link("Ableton Link needs the 'aalink' package.")
                return
            self.link_tempo_received.connect(self._apply_link_tempo)
            self.link_sync = LinkSync(self.link_tempo_received.emit)
        if not self.link_sync.start(self.spin_bpm.value()):
            self._reject_link("Ableton Link could not be started.")
            return
        publish(Event.STATUS_UPDATE, text="Ableton Link enabled")

    def set_osc_enabled(self, enabled):
        if not enabled:
            if self.osc_server is not None:
                self.osc_server.stop()
            return

        if self.osc_server is None:
            player = self.control_bar.player
            self.osc_play.connect(player.play)
            self.osc_pause.connect(player.pause)
            self.osc_stop.connect(player.stop)
            self.osc_tempo.connect(self._apply_osc_tempo)
            self.osc_server = OscServer(
                on_play=self.osc_play.emit,
                on_pause=self.osc_pause.emit,
                on_stop=self.osc_stop.emit,
                on_tempo=self.osc_tempo.emit,
            )
        if not self.osc_server.start():
            publish(
                Event.STATUS_UPDATE,
                text=f"OSC could not listen on port {self.osc_server.port}.",
            )
            self.chk_osc.blockSignals(True)
            self.chk_osc.setChecked(False)
            self.chk_osc.blockSignals(False)
            return
        publish(
            Event.STATUS_UPDATE,
            text=f"OSC listening on {self.osc_server.host}:"
            f"{self.osc_server.port}",
        )

    def set_obs_enabled(self, enabled):
        if not enabled:
            self._disconnect_obs_playback()
            if self.obs_client is not None:
                self.obs_client.stop()
            return

        if self.obs_client is not None:
            self.obs_client.stop()
        if not self._obs_signals_connected:
            self.obs_status.connect(
                lambda text: publish(Event.STATUS_UPDATE, text=text)
            )
            self.obs_connected.connect(self._on_obs_connected)
            self._obs_signals_connected = True
        self.obs_client = ObsClient(
            url=os.environ.get("KORSHMER_OBS_URL", OBS_DEFAULT_URL),
            password=get_secret("OBS_PASSWORD"),
            on_status=self.obs_status.emit,
        )
        client = self.obs_client
        publish(Event.STATUS_UPDATE, text="Connecting to OBS...")
        threading.Thread(
            target=lambda: self.obs_connected.emit(client.start()),
            name="korshmer-obs-connect",
            daemon=True,
        ).start()

    def _on_obs_connected(self, connected):
        if not connected:
            self.chk_obs.blockSignals(True)
            self.chk_obs.setChecked(False)
            self.chk_obs.blockSignals(False)
            return
        if not self.chk_obs.isChecked():
            self.obs_client.stop()
            return
        self._disconnect_obs_playback()
        self.control_bar.player.playbackStateChanged.connect(
            self._on_obs_playback_state
        )
        self._obs_playback_connected = True

    def _disconnect_obs_playback(self):
        if not self._obs_playback_connected:
            return
        self._obs_playback_connected = False
        try:
            self.control_bar.player.playbackStateChanged.disconnect(
                self._on_obs_playback_state
            )
        except (RuntimeError, TypeError):
            pass

    def _on_obs_playback_state(self, state):
        client = self.obs_client
        if client is None:
            return
        name = getattr(state, "name", str(state))
        if name in ("PlayingState", "1"):
            client.start_recording()
        elif name in ("StoppedState", "0"):
            client.stop_recording()

    def _apply_osc_tempo(self, bpm):
        self.spin_bpm.setValue(validate_bpm(bpm))

    def _reject_link(self, text):
        publish(Event.STATUS_UPDATE, text=text)
        self.chk_link.blockSignals(True)
        self.chk_link.setChecked(False)
        self.chk_link.blockSignals(False)

    def _apply_link_tempo(self, bpm):
        value = validate_bpm(bpm)
        self._applying_link_tempo = True
        try:
            self.spin_bpm.setValue(value)
        finally:
            self._applying_link_tempo = False

    def on_tap_tempo(self):
        bpm = self.tap_tempo.tap(time.monotonic())
        if bpm is None:
            return
        self.spin_bpm.setValue(bpm)

    def _restart_beat_timer(self, bpm):
        self.beat_timer.start(max(1, int(round(60000.0 / bpm))))

    def _on_beat(self):
        if hasattr(self, "beat_indicator"):
            self.beat_indicator.setStyleSheet(BEAT_INDICATOR_ON)
            self.beat_off_timer.start(80)

    def _on_beat_off(self):
        if hasattr(self, "beat_indicator"):
            self.beat_indicator.setStyleSheet(BEAT_INDICATOR_OFF)

    def update_editor_tempo(self, val):
        val = validate_bpm(val)
        self.pending_bpm_value = val
        self._restart_beat_timer(val)

        link = getattr(self, "link_sync", None)
        if link is not None and link.enabled and not self._applying_link_tempo:
            link.publish_tempo(val)

        if hasattr(self, "spin_bpm"):
            if self.bpm_animator is None:
                self.bpm_animator = BpmAnimator(self.spin_bpm)
            self.bpm_animator.flash()

        self.bpm_update_timer.start(100)

    def _apply_tempo_update(self):
        if self.pending_bpm_value is None:
            return

        new_bpm = float(self.pending_bpm_value)
        if hasattr(self, "arranger"):
            self.arranger.piano_roll.set_tempo(new_bpm)
            self.arranger.piano_roll.viewport().update()

        if hasattr(self, "control_bar") and hasattr(self.control_bar, "player"):
            self.control_bar.player.set_tempo(new_bpm)

        self.pending_bpm_value = None


def register_exit_cleanup():
    from korshmer.audio.io import register_temp_cleanup

    register_temp_cleanup()


def load_plugins():
    from korshmer.plugins import PluginRegistry

    try:
        PluginRegistry().load_all(plugin_dir=str(DATA_DIR / "plugins"))
    except Exception as e:
        logging.getLogger(__name__).error(f"Plugin loading failed: {e}")


def main():
    install_crash_handler()
    register_exit_cleanup()
    load_plugins()
    app = QApplication()
    f = QFont("Tahoma", 12)
    app.setFont(f)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
