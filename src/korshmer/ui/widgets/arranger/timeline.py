import os
import queue
from collections import deque

import numpy as np
import pretty_midi
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from korshmer.audio_processing.effect_chain import (
    apply_effect_chain,
    validate_effect_chain_depth,
)
from korshmer.constants import DATA_DIR, GRID_RESOLUTIONS, HELP_TEXTS
from korshmer.midi.performance_recorder import (
    PerformanceRecorder,
    open_input_port,
)
from korshmer.ui.viewmodels.arranger_vm import ArrangerViewModel
from korshmer.ui.viewmodels.editor_vm import compute_clip_waveform
from korshmer.ui.widgets.arranger.track_panel import TrackPanel
from korshmer.ui.widgets.material import MAT_ERROR, MAT_PRIMARY, MAT_TEXT
from korshmer.ui.widgets.material.button import (
    MaterialButton,
    MaterialToggleButton,
)
from korshmer.ui.widgets.midi.piano_roll import InteractivePianoRoll
from korshmer.utils.audio_host import (
    active_output_latency,
    compensation_frames,
    open_duplex_stream,
    open_input_stream,
    stream_input_latency,
)
from korshmer.utils.filesystem import sanitize_path
from korshmer.utils.tempo import midi_file_tempo, validate_bpm

DEFAULT_GRID_LABEL = "1/16"
ZOOM_STEP = 1.25
SNAP_HELP = (
    "Snaps note edits to the current grid. Hold Alt while dragging to bypass."
)


LEVEL_WINDOW_SECONDS = 0.05
PLACEHOLDER_MARGIN = 24


class LiveWaveformStrip(QWidget):
    MAX_BARS = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        self.levels = deque(maxlen=self.MAX_BARS)
        self.setFixedSize(160, 30)
        self.setVisible(False)

    def push_level(self, level):
        self.levels.append(min(1.0, max(0.0, float(level))))
        self.update()

    def clear(self):
        self.levels.clear()
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(MAT_ERROR))
        h = self.height()
        visible = list(self.levels)[-self.width() // 2 :]
        x = self.width() - 2 * len(visible)
        for level in visible:
            bar = max(1, int(level * 4 * h))
            bar = min(bar, h)
            painter.drawRect(x, (h - bar) // 2, 1, bar)
            x += 2


class _StreamingWriter:
    def __init__(self, filename, rate, skip, on_level):
        self.filename = filename
        self.rate = rate
        self.skip = max(0, int(skip))
        self.on_level = on_level
        self.window = max(1, int(rate * LEVEL_WINDOW_SECONDS))
        self._file = None
        self._head = []
        self._head_len = 0
        self._decided = self.skip == 0
        self._sq = 0.0
        self._n = 0
        self.received = 0

    def drain(self, blocks):
        while True:
            try:
                block = blocks.get_nowait()
            except queue.Empty:
                return
            self._add(np.asarray(block))

    def _add(self, block):
        self.received += len(block)
        self._track_level(block)
        if self._decided:
            self._write(block)
            return
        self._head.append(block)
        self._head_len += len(block)
        if self._head_len > self.skip:
            data = np.concatenate(self._head, axis=0)[self.skip :]
            self._head = []
            self._decided = True
            self._write(data)

    def _write(self, data):
        if not len(data):
            return
        if self._file is None:
            self._file = sf.SoundFile(
                self.filename,
                "w",
                samplerate=self.rate,
                channels=1 if data.ndim == 1 else data.shape[1],
            )
        self._file.write(data)

    def _track_level(self, block):
        flat = block.astype("float64").ravel()
        pos = 0
        while pos < len(flat):
            take = min(self.window - self._n, len(flat) - pos)
            part = flat[pos : pos + take]
            self._sq += float(np.dot(part, part))
            self._n += take
            pos += take
            if self._n >= self.window:
                self.on_level(float(np.sqrt(self._sq / self._n)))
                self._sq = 0.0
                self._n = 0

    def finish(self):
        if not self._decided and self._head:
            self._decided = True
            self._write(np.concatenate(self._head, axis=0))
            self._head = []
        self.close()
        return self.received > 0 and os.path.exists(self.filename)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


class AudioRecorderThread(QThread):
    recording_finished = Signal(str)
    level_ready = Signal(float)

    SAMPLE_RATE = 44100

    def __init__(
        self, filename, monitoring=False, monitor_effects=(), looper=None
    ):
        super().__init__()

        self.looper = looper
        self.filename = filename
        self.is_recording = True

        self.frames = []
        self._blocks = queue.SimpleQueue()
        self.compensation = 0
        self.monitoring = bool(monitoring)

        self.monitor_effects = list(
            validate_effect_chain_depth(list(monitor_effects))
        )

    def _open_stream(self):
        if self.monitoring:
            return open_duplex_stream(
                sd,
                samplerate=self.SAMPLE_RATE,
                channels=1,
                latency="low",
                callback=self._duplex_cb,
            )
        return open_input_stream(
            sd,
            samplerate=self.SAMPLE_RATE,
            channels=1,
            callback=self._cb,
        )

    def run(self):
        writer = None
        try:
            with self._open_stream() as stream:
                self.compensation = compensation_frames(
                    stream_input_latency(stream),
                    active_output_latency(),
                    self.SAMPLE_RATE,
                )
                writer = _StreamingWriter(
                    self.filename,
                    self.SAMPLE_RATE,
                    self.compensation,
                    self.level_ready.emit,
                )
                while self.is_recording:
                    writer.drain(self._blocks)
                    self.msleep(10)

            if writer is not None:
                writer.drain(self._blocks)
                if writer.finish():
                    self.recording_finished.emit(self.filename)
        except Exception as e:
            print(f"Rec Error: {e}")
        finally:
            if writer is not None:
                writer.close()

    def compensate(self, data):
        if 0 < self.compensation < len(data):
            return data[self.compensation :]
        return data

    def _cb(self, indata, _frames, _time, _status):
        if self.is_recording:
            self._blocks.put(indata.copy())
            if self.looper is not None:
                self.looper.feed_input(indata)

    def _duplex_cb(self, indata, outdata, _frames, _time, _status):
        if self.is_recording:
            self._blocks.put(indata.copy())
            if self.looper is not None:
                self.looper.feed_input(indata)
        try:
            wet = apply_effect_chain(indata.copy(), self.monitor_effects)
            wet = np.asarray(wet, dtype=outdata.dtype)
            if wet.shape != outdata.shape:
                wet = np.broadcast_to(wet, outdata.shape)
            outdata[:] = wet
        except Exception:
            outdata.fill(0)

    def stop(self):
        self.is_recording = False


class ClipWaveformWorker(QThread):
    waveform_ready = Signal(int, object, float)

    def __init__(self, clip_id, path, points=512, parent=None):
        super().__init__(parent)
        self.clip_id = clip_id
        self.path = path
        self.points = points

    def run(self):
        waveform, duration = compute_clip_waveform(self.path, self.points)
        self.waveform_ready.emit(self.clip_id, waveform, duration)


class ArrangerTimeline(QWidget):
    recording_saved = Signal(str)
    editor_play_request = Signal(object)
    tempo_changed = Signal(float)
    midi_exported = Signal(str)
    export_mix_requested = Signal()
    export_clip_requested = Signal()
    tracks_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = ArrangerViewModel()
        self._waveform_workers = []
        self._pending_waveform_ids = set()
        layout = QVBoxLayout(self)

        self.piano_roll = InteractivePianoRoll()
        self.model = self.piano_roll.model
        self.track_panel = TrackPanel(self.model)
        self.piano_roll.viewport_changed.connect(self._start_visible_waveforms)

        self._build_transport_toolbar(layout)
        self._build_edit_toolbar(layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.track_panel)
        self.splitter.addWidget(self.piano_roll)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([340, 820])

        self.lbl_placeholder = QLabel(
            "Drop MIDI or Audio Here, or Select from Output List",
            self.piano_roll,
        )
        self.lbl_placeholder.setWordWrap(True)
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet(
            f"font-size: 24px; color: {MAT_TEXT}; background: transparent;"
        )
        self.lbl_placeholder.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.piano_roll.installEventFilter(self)
        self._place_placeholder()

        self.piano_roll.notes_changed.connect(self.check_placeholder)
        self.piano_roll.model_changed.connect(self._on_model_changed)
        self.piano_roll.audio_dropped.connect(self._on_audio_dropped)
        self.track_panel.tracks_changed.connect(self._on_tracks_edited)
        self.track_panel.active_track_changed.connect(
            self._on_active_track_changed
        )
        self.track_panel.select_track_notes_requested.connect(
            self.piano_roll.select_track_notes
        )

        layout.addWidget(self.splitter, 1)
        self._update_history_buttons()

    def _build_transport_toolbar(self, layout):
        toolbar = QHBoxLayout()
        actions = QHBoxLayout()

        self.btn_rec = QPushButton("● REC")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setCursor(Qt.PointingHandCursor)
        self.btn_rec.setFixedHeight(32)
        self.btn_rec.setFixedWidth(80)
        self.btn_rec.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {MAT_ERROR}; border-radius: 16px; font-weight: bold;
            }}
            QPushButton:checked {{
                background-color: {MAT_ERROR}; color: white;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 0, 64, 0.1);
            }}
        """
        )
        self.btn_rec.clicked.connect(self.toggle_recording)
        self.recorder = None

        self.btn_perf = MaterialToggleButton("● Perform", MAT_ERROR)
        self.btn_perf.setFixedSize(92, 30)
        self.btn_perf.setToolTip(
            "Record the notes and controller moves of a connected MIDI "
            "device as a new editor track"
        )
        self.btn_perf.clicked.connect(self.toggle_performance_recording)
        self.performance = PerformanceRecorder()

        self.looper = None
        self.btn_loop = MaterialToggleButton("Loop", MAT_PRIMARY)
        self.btn_loop.setFixedSize(60, 30)
        self.btn_loop.setToolTip(
            "Send the recording into the looper: bars of the master BPM, "
            "endless overdub layers"
        )
        self.btn_layer = MaterialButton("+ Layer", "tonal")
        self.btn_layer.setToolTip("Record another layer at the next loop start")
        self.btn_layer.clicked.connect(self.add_loop_layer)
        self.btn_layer_undo = MaterialButton("Undo layer", "tonal")
        self.btn_layer_undo.clicked.connect(self.undo_loop_layer)
        self.btn_loop_clear = MaterialButton("Clear loop", "error")
        self.btn_loop_clear.clicked.connect(self.clear_loop_layers)
        self._perf_port = None

        self.btn_clear = MaterialButton("✕︎ Clear", "error")
        self.btn_clear.clicked.connect(self.clear_editor)
        self.btn_merge = MaterialButton("✚︎ Import/Merge MIDI", "tonal")
        self.btn_merge.clicked.connect(self.import_merge_midi)
        self.btn_sync_play = MaterialButton("▶︎ Play Editor", "primary")
        self.btn_sync_play.clicked.connect(self.sync_and_play)
        self.btn_export_midi = MaterialButton("⬇︎ Export MIDI", "tonal")
        self.btn_export_midi.clicked.connect(self.export_midi)
        self.btn_export_mix = MaterialButton("♫ Export Mix", "success")
        self.btn_export_mix.setToolTip(
            "Render the editor MIDI together with the audio clips"
        )
        self.btn_export_mix.clicked.connect(self.export_mix_requested.emit)
        self.btn_export_clip = MaterialButton("🎬 Export Clip", "success")
        self.btn_export_clip.setToolTip(
            "Export a video clip with the notes, waveform, instruments and "
            "technical specs, over the mastered MIDI audio"
        )
        self.btn_export_clip.clicked.connect(self.export_clip_requested.emit)

        self.btn_monitor = MaterialToggleButton("Monitor", MAT_PRIMARY)
        self.btn_monitor.setFixedSize(76, 30)
        self.btn_monitor.setToolTip(
            "Hear the input live while recording. Use headphones to avoid "
            "feedback. The saved recording stays dry."
        )

        toolbar.addWidget(self.btn_rec)
        toolbar.addWidget(self.btn_monitor)
        toolbar.addWidget(self.btn_perf)
        toolbar.addWidget(self.btn_loop)
        toolbar.addWidget(self.btn_layer)
        toolbar.addWidget(self.btn_layer_undo)
        toolbar.addWidget(self.btn_loop_clear)
        self.live_waveform = LiveWaveformStrip()
        toolbar.addWidget(self.live_waveform)
        toolbar.addStretch()
        for button in (
            self.btn_clear,
            self.btn_merge,
            self.btn_sync_play,
            self.btn_export_midi,
            self.btn_export_mix,
            self.btn_export_clip,
        ):
            actions.addWidget(button)
        for button in (
            self.btn_layer,
            self.btn_layer_undo,
            self.btn_loop_clear,
            *(actions.itemAt(i).widget() for i in range(actions.count())),
        ):
            button.setMinimumWidth(button.sizeHint().width())
        actions.addStretch()
        layout.addLayout(toolbar)
        layout.addLayout(actions)

    def _build_edit_toolbar(self, layout):
        toolbar = QHBoxLayout()

        self.btn_undo = MaterialButton("↶ Undo", "default")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = MaterialButton("↷ Redo", "default")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.redo)
        for button in (self.btn_undo, self.btn_redo):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        self.btn_snap = MaterialToggleButton("Snap", MAT_PRIMARY)
        self.btn_snap.setFixedSize(60, 30)
        self.btn_snap.setToolTip(SNAP_HELP)
        self.btn_snap.setChecked(True)
        self.btn_snap.toggled.connect(self.set_snap)

        lbl_grid = QLabel("Grid")
        lbl_grid.setStyleSheet(f"color: {MAT_TEXT};")
        self.combo_grid = QComboBox()
        self.combo_grid.addItems(list(GRID_RESOLUTIONS.keys()))
        self.combo_grid.setCurrentText(DEFAULT_GRID_LABEL)
        self.combo_grid.setToolTip(HELP_TEXTS.get("grid", "Grid resolution"))
        self.combo_grid.currentTextChanged.connect(self.set_grid)
        self.piano_roll.set_grid_divisions(GRID_RESOLUTIONS[DEFAULT_GRID_LABEL])

        self.btn_zoom_out = MaterialButton("−", "default")
        self.btn_zoom_out.setToolTip("Zoom out (Ctrl+Wheel)")
        self.btn_zoom_out.setFixedWidth(36)
        self.btn_zoom_out.clicked.connect(
            lambda: self.piano_roll.zoom(1.0 / ZOOM_STEP, 1.0)
        )
        self.btn_zoom_in = MaterialButton("+", "default")
        self.btn_zoom_in.setToolTip("Zoom in (Ctrl+Wheel)")
        self.btn_zoom_in.setFixedWidth(36)
        self.btn_zoom_in.clicked.connect(
            lambda: self.piano_roll.zoom(ZOOM_STEP, 1.0)
        )

        toolbar.addWidget(self.btn_undo)
        toolbar.addWidget(self.btn_redo)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.btn_snap)
        toolbar.addWidget(lbl_grid)
        toolbar.addWidget(self.combo_grid)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addStretch()
        layout.addLayout(toolbar)

    def eventFilter(self, obj, event):
        if obj is self.piano_roll and event.type() == QEvent.Resize:
            self._place_placeholder()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_placeholder()

    def _place_placeholder(self):
        label = getattr(self, "lbl_placeholder", None)
        if label is None:
            return
        area = self.piano_roll.rect()
        width = max(120, area.width() - 2 * PLACEHOLDER_MARGIN)
        label.setFixedWidth(width)
        label.setFixedHeight(label.heightForWidth(width))
        label.move(
            area.center().x() - label.width() // 2,
            area.center().y() - label.height() // 2,
        )

    def check_placeholder(self):
        self.lbl_placeholder.setVisible(not self.piano_roll.has_content())

    def _update_history_buttons(self):
        self.btn_undo.setEnabled(self.model.can_undo())
        self.btn_redo.setEnabled(self.model.can_redo())

    def _on_model_changed(self):
        if self.track_panel.refresh():
            self.tracks_changed.emit()
        self._update_history_buttons()
        self.check_placeholder()

    def _on_tracks_edited(self):
        self.piano_roll.load_model(keep_selection=True)
        self._update_history_buttons()
        self.check_placeholder()
        self.tracks_changed.emit()

    def _on_active_track_changed(self, track_id):
        self.piano_roll.current_track_id = track_id

    def _on_audio_dropped(self, path):
        self.add_audio_clip(path)

    def clear_editor(self):
        self.wait_for_waveforms()
        self.model.snapshot()
        self.model.clear()
        self.piano_roll.load_model()
        self._on_model_changed()

    def import_merge_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import MIDI", "", "MIDI Files (*.mid *.midi)"
        )
        if path:
            try:
                self.load_midi_file(path, replace=False)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Import failed: {e}")

    def load_midi_file(
        self,
        path,
        replace=True,
        tempo=None,
        file_tempo=True,
        merge_instruments=False,
    ):
        sanitize_path(path)
        declared = midi_file_tempo(path) if file_tempo else None
        if declared:
            tempo = declared
        if tempo and float(tempo) > 0:
            self.piano_roll.set_tempo(validate_bpm(tempo))
            self.tempo_changed.emit(float(self.piano_roll.tempo))
        base_tempo = float(self.piano_roll.tempo)
        pm = pretty_midi.PrettyMIDI(str(path), initial_tempo=base_tempo)
        return self.load_midi_object(
            pm, replace=replace, merge_instruments=merge_instruments
        )

    def load_midi_object(
        self, pm, replace=True, name_prefix=None, merge_instruments=False
    ):
        self.piano_roll.sync_model_from_scene()
        self.model.snapshot()
        created = self.model.load_midi(
            pm,
            replace=replace,
            name_prefix=name_prefix,
            merge_instruments=merge_instruments,
        )
        if created and (
            replace or self.piano_roll.current_track_id not in self.model.tracks
        ):
            self.piano_roll.current_track_id = created[0].id
            self.track_panel.active_track_id = created[0].id
        if replace:
            self.piano_roll.detect_grid_offset(pm)
        self.piano_roll.load_model()
        self._on_model_changed()
        return created

    def get_midi_from_editor(self):
        self.piano_roll.sync_model_from_scene()
        return self.model.to_pretty_midi(
            tempo=self.piano_roll.tempo, respect_mute_solo=True
        )

    def sync_and_play(self):
        pm = self.get_midi_from_editor()
        self.piano_roll.is_monitoring = True
        self.editor_play_request.emit(pm)

    def export_midi(self):
        pm = self.get_midi_from_editor()
        if not any(inst.notes for inst in pm.instruments):
            QMessageBox.information(self, "Export MIDI", "Editor is empty.")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MIDI",
            self.vm.export_midi_filename(),
            "MIDI Files (*.mid)",
        )
        if not path:
            return None
        path = self.vm.ensure_midi_suffix(path)
        try:
            sanitize_path(path)
            os.makedirs(os.path.dirname(path) or str(DATA_DIR), exist_ok=True)
            pm.write(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")
            return None
        self.midi_exported.emit(path)
        return path

    def add_audio_clip(
        self, path, start_sec=0.0, name=None, kind="audio", gain=1.0
    ):
        sanitize_path(path)
        self.piano_roll.sync_model_from_scene()
        self.model.snapshot()
        clip = self.model.add_clip(
            path, start=start_sec, name=name, kind=kind, gain=gain
        )
        self.piano_roll.add_clip(clip)
        self._queue_waveform(clip)
        self._start_visible_waveforms()
        self._on_model_changed()
        return clip

    def _queue_waveform(self, clip):
        self._pending_waveform_ids.add(clip.id)

    def _start_visible_waveforms(self):
        if not self._pending_waveform_ids:
            return
        self._pending_waveform_ids = {
            clip_id
            for clip_id in self._pending_waveform_ids
            if self.model.clip_by_id(clip_id) is not None
        }
        if self.piano_roll.isVisible():
            ready = set(self.piano_roll.visible_clip_ids())
        else:
            ready = set(self._pending_waveform_ids)
        for clip_id in sorted(self._pending_waveform_ids & ready):
            self._start_waveform_worker(self.model.clip_by_id(clip_id))
        self._pending_waveform_ids -= ready

    def _start_waveform_worker(self, clip):
        self._prune_finished_workers()
        worker = ClipWaveformWorker(clip.id, clip.path, parent=self)
        worker.waveform_ready.connect(self._on_waveform_ready)
        worker.finished.connect(self._prune_finished_workers)
        self._waveform_workers.append(worker)
        worker.start()
        return worker

    def _prune_finished_workers(self):
        still_running = []
        for worker in self._waveform_workers:
            if worker.isRunning():
                still_running.append(worker)
            else:
                worker.deleteLater()
        self._waveform_workers = still_running

    def _on_waveform_ready(self, clip_id, waveform, duration):
        if self.model.clip_by_id(clip_id) is None:
            return
        self.model.set_clip_waveform(clip_id, waveform, duration)
        self.piano_roll.refresh_clip(clip_id)
        self.track_panel.update_status()

    def wait_for_waveforms(self, timeout_ms=5000):
        for worker in list(self._waveform_workers):
            worker.wait(timeout_ms)
        QApplication.processEvents()

    def get_audio_clips(self):
        self.piano_roll.sync_model_from_scene()
        return self.model.active_clips()

    def get_all_clips(self):
        self.piano_roll.sync_model_from_scene()
        return [clip.to_dict() for clip in self.model.clips]

    def undo(self):
        return self.piano_roll.undo()

    def redo(self):
        return self.piano_roll.redo()

    def select_all(self):
        return self.piano_roll.select_all()

    def delete_selected(self):
        return self.piano_roll.delete_selected()

    def copy_selected(self):
        return self.piano_roll.copy_selected()

    def paste(self):
        return self.piano_roll.paste()

    def duplicate_selected(self):
        return self.piano_roll.duplicate_selected()

    def transpose_selected(self, semitones):
        return self.piano_roll.transpose_selected(semitones)

    def quantize_selected(self, grid_div=None, strength=1.0):
        divisions = grid_div or self.current_grid_divisions()
        return self.piano_roll.quantize_selected(divisions, strength)

    def humanize_selected(self, t=0.01, v=8):
        return self.piano_roll.humanize_selected(t, v)

    def set_snap(self, enabled):
        self.piano_roll.set_snap(enabled)
        if self.btn_snap.isChecked() != bool(enabled):
            self.btn_snap.blockSignals(True)
            self.btn_snap.setChecked(bool(enabled))
            self.btn_snap.blockSignals(False)

    def set_grid(self, name):
        if name not in GRID_RESOLUTIONS:
            return
        self.piano_roll.set_grid_divisions(GRID_RESOLUTIONS[name])
        if self.combo_grid.currentText() != name:
            self.combo_grid.blockSignals(True)
            self.combo_grid.setCurrentText(name)
            self.combo_grid.blockSignals(False)
        self.piano_roll.viewport().update()

    def current_grid_divisions(self):
        return GRID_RESOLUTIONS.get(
            self.combo_grid.currentText(), GRID_RESOLUTIONS[DEFAULT_GRID_LABEL]
        )

    def to_project_dict(self):
        self.piano_roll.sync_model_from_scene()
        data = self.model.to_dict()
        data["snap"] = self.piano_roll.snap_enabled
        data["grid"] = self.combo_grid.currentText()
        data["current_track_id"] = self.piano_roll.current_track_id
        return data

    def load_project_dict(self, data):
        self.wait_for_waveforms()
        self.model.load_state(data or {})
        self.piano_roll.tempo = self.model.tempo
        self.set_snap(bool((data or {}).get("snap", True)))
        self.set_grid((data or {}).get("grid", DEFAULT_GRID_LABEL))
        wanted = (data or {}).get("current_track_id")
        self.piano_roll.current_track_id = (
            wanted
            if wanted in self.model.tracks
            else self.model.first_track_id()
        )
        self.track_panel.active_track_id = self.piano_roll.current_track_id
        self.piano_roll.load_model()
        self._pending_waveform_ids = set()
        for clip in self.model.clips:
            if os.path.exists(clip.path):
                self._queue_waveform(clip)
        self._start_visible_waveforms()
        self.tempo_changed.emit(float(self.model.tempo))
        self._on_model_changed()

    def toggle_performance_recording(self):
        if self.btn_perf.isChecked():
            self.performance.start()
            try:
                self._perf_port = open_input_port(self.performance)
            except RuntimeError as e:
                self.performance.stop()
                self.btn_perf.setChecked(False)
                QMessageBox.warning(self, "Error", str(e))
            return
        port, self._perf_port = self._perf_port, None
        if port is not None:
            port.close()
        self.performance.stop()
        pm = self.performance.to_pretty_midi(tempo=float(self.piano_roll.tempo))
        if any(inst.notes for inst in pm.instruments):
            self.load_midi_object(pm, replace=False, name_prefix="Performance")

    def add_loop_layer(self):
        if self.looper is None or self.recorder is None:
            QMessageBox.information(
                self, "Looper", "Start REC with Loop on first."
            )
            return
        self.looper.arm_record()
        try:
            self.looper.start_output()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Loop playback failed: {e}")

    def undo_loop_layer(self):
        if self.looper is not None:
            self.looper.undo_layer()

    def clear_loop_layers(self):
        if self.looper is not None:
            self.looper.clear()

    def toggle_recording(self):
        if self.btn_rec.isChecked():
            try:
                sd.query_devices(kind="input")
            except Exception:
                self.btn_rec.setChecked(False)
                QMessageBox.warning(self, "Error", "No input device found.")
                return

            fname = self.vm.recording_filename()
            self.recorder = AudioRecorderThread(
                fname,
                monitoring=self.btn_monitor.isChecked(),
                looper=self.looper if self.btn_loop.isChecked() else None,
            )
            self.recorder.recording_finished.connect(self.on_rec_finished)
            self.live_waveform.clear()
            self.live_waveform.setVisible(True)
            self.recorder.level_ready.connect(self.live_waveform.push_level)
            self.recorder.start()
            self.btn_rec.setText("■ STOP")
        else:
            if self.recorder:
                self.recorder.stop()
                self.recorder.wait()
                self.recorder = None
            self.live_waveform.setVisible(False)
            self.btn_rec.setText("● REC")

    def on_rec_finished(self, path):
        self.recording_saved.emit(path)
        QMessageBox.information(
            self, "Recorded", f"Saved: {os.path.basename(path)}"
        )
