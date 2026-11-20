import logging
import os
import queue
from collections import deque

import numpy as np
import pretty_midi
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from korshmer.audio.loader import usable_audio_path
from korshmer.audio_processing.clip_notes import notes_audio
from korshmer.audio_processing.effect_chain import (
    apply_effect_chain,
    validate_effect_chain_depth,
)
from korshmer.constants import DATA_DIR, GRID_RESOLUTIONS, HELP_TEXTS
from korshmer.exceptions import AudioFormatError
from korshmer.midi.performance_recorder import (
    PerformanceRecorder,
    open_input_port,
)
from korshmer.ui.viewmodels.arranger_vm import ArrangerViewModel
from korshmer.ui.viewmodels.editor_vm import (
    TempoNormalizeError,
    compute_clip_waveform,
)
from korshmer.ui.widgets.arranger.track_panel import TrackPanel
from korshmer.ui.widgets.material import MAT_ERROR, MAT_PRIMARY, MAT_TEXT
from korshmer.ui.widgets.material.adaptive import FlowLayout
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
TOOLBAR_SPACING = 6
EVEN_BPM_HELP = (
    "Even BPM: when a song changes tempo, move every note and audio clip "
    "onto one average tempo so the whole song sits on a single grid. "
    "Turn it off to keep the tempo changes of the song; the grid then "
    "follows them."
)
ALIGNED_CLIPS_DIR = "tempo_aligned"
CLIP_ALIGN_LABEL = "Align Clip to the Beat"
CLIP_NOISE_LABEL = "Remove Background Noise"
CLIP_QUALITY_LABEL = "Improve Level and Quality"
CLIP_ALL_LABEL = "Fix Clip Automatically"
CLIP_EARLIER_LABEL = "Move Clip One Beat Earlier"
CLIP_LATER_LABEL = "Move Clip One Beat Later"
CLIP_MOVE_TOLERANCE_S = 1e-3
CLIP_FOLLOW_LABEL = "Following the tempo"
TEMPO_FOLLOW_DELAY_MS = 600
STEMS_FOLDER = "Stems"
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


class ClipEditWorker(QThread):
    edit_ready = Signal(int, str, object)
    edit_failed = Signal(int, str, str)
    edit_cancelled = Signal(int, str)
    edit_progress = Signal(int, str, str, int)

    def __init__(self, clip_id, label, job, parent=None):
        super().__init__(parent)
        self.clip_id = clip_id
        self.label = label
        self.job = job

    def _report(self, stage, done, total):
        percent = int(100 * done / total) if total else 0
        self.edit_progress.emit(self.clip_id, self.label, stage, percent)

    def run(self):
        from korshmer.audio_processing.clip_polish import ClipFixCancelled

        try:
            result = self.job(self._report, self.isInterruptionRequested)
        except ClipFixCancelled:
            self.edit_cancelled.emit(self.clip_id, self.label)
            return
        except Exception as problem:
            logging.getLogger(__name__).exception(
                f"{self.label} failed for clip {self.clip_id}"
            )
            self.edit_failed.emit(self.clip_id, self.label, str(problem))
            return
        self.edit_ready.emit(self.clip_id, self.label, result)


class ArrangerTimeline(QWidget):
    recording_saved = Signal(str)
    editor_play_request = Signal(object)
    tempo_changed = Signal(float)
    midi_exported = Signal(str)
    export_mix_requested = Signal()
    export_clip_requested = Signal()
    tracks_changed = Signal()
    clip_edit_status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = ArrangerViewModel()
        self._waveform_workers = []
        self._pending_waveform_ids = set()
        self._clip_edit_workers = []
        self._clip_edit_files = set()
        self.notes_audio = notes_audio
        self._saved_clip_files = set()
        layout = QVBoxLayout(self)

        self.piano_roll = InteractivePianoRoll()
        self.model = self.piano_roll.model
        self.track_panel = TrackPanel(self.model)
        self.piano_roll.viewport_changed.connect(self._start_visible_waveforms)
        self.piano_roll.tempo_followed.connect(self._on_tempo_followed)
        self._tempo_follow_timer = QTimer(self)
        self._tempo_follow_timer.setSingleShot(True)
        self._tempo_follow_timer.setInterval(TEMPO_FOLLOW_DELAY_MS)
        self._tempo_follow_timer.timeout.connect(self.stretch_clips_to_tempo)

        self._build_transport_toolbar(layout)
        self._build_edit_toolbar(layout)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.track_panel)
        self.splitter.addWidget(self.piano_roll)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([340, 820])

        self.lbl_placeholder = QLabel(
            "Drop MIDI or Audio Here, or pick it from My music",
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
        self.piano_roll.notes_changed.connect(self.track_panel.update_status)
        self.piano_roll.model_changed.connect(self._on_model_changed)
        self.piano_roll.audio_dropped.connect(self._on_audio_dropped)
        self.track_panel.tracks_changed.connect(self._on_tracks_edited)
        self.track_panel.active_track_changed.connect(
            self._on_active_track_changed
        )
        self.track_panel.select_track_notes_requested.connect(
            self.piano_roll.select_track_notes
        )

        self._build_clip_edit_bar(layout)
        layout.addWidget(self.splitter, 1)
        self._update_history_buttons()

    def _build_clip_edit_bar(self, layout):
        self.clip_edit_bar = QWidget()
        row = QHBoxLayout(self.clip_edit_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(TOOLBAR_SPACING)
        self.clip_edit_label = QLabel("")
        self.clip_edit_progress = QProgressBar()
        self.clip_edit_progress.setRange(0, 100)
        self.clip_edit_progress.setAccessibleName("Clip fix progress")
        self.btn_cancel_clip_edit = QPushButton("Cancel")
        self.btn_cancel_clip_edit.setToolTip(
            "Stop the clip fix that is running and leave the clip as it was"
        )
        self.btn_cancel_clip_edit.setAccessibleName("Cancel clip fix")
        self.btn_cancel_clip_edit.clicked.connect(self.cancel_clip_edits)
        row.addWidget(self.clip_edit_label)
        row.addWidget(self.clip_edit_progress, 1)
        row.addWidget(self.btn_cancel_clip_edit)
        self.clip_edit_bar.setVisible(False)
        layout.addWidget(self.clip_edit_bar)

    def _build_transport_toolbar(self, layout):
        toolbar = FlowLayout(spacing=TOOLBAR_SPACING)
        self._toolbar = toolbar

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
        for button in (
            self.btn_layer,
            self.btn_layer_undo,
            self.btn_loop_clear,
            self.btn_clear,
            self.btn_merge,
            self.btn_sync_play,
            self.btn_export_midi,
            self.btn_export_mix,
            self.btn_export_clip,
        ):
            button.setMinimumWidth(button.sizeHint().width())
        for button in (
            self.btn_sync_play,
            self.btn_merge,
            self.btn_clear,
            self.btn_export_midi,
            self.btn_export_mix,
            self.btn_export_clip,
        ):
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

    def _build_edit_toolbar(self, _layout):
        toolbar = self._toolbar

        self.btn_undo = MaterialButton("↶ Undo", "default")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo = MaterialButton("↷ Redo", "default")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.clicked.connect(self.redo)
        for button in (self.btn_undo, self.btn_redo):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            button.setMinimumWidth(button.sizeHint().width())

        self.btn_snap = MaterialToggleButton("Snap", MAT_PRIMARY)
        self.btn_snap.setFixedSize(60, 30)
        self.btn_snap.setToolTip(SNAP_HELP)
        self.btn_snap.setChecked(True)
        self.btn_snap.toggled.connect(self.set_snap)

        self.btn_even_bpm = MaterialToggleButton("Even BPM", MAT_PRIMARY)
        self.btn_even_bpm.setFixedSize(90, 30)
        self.btn_even_bpm.setToolTip(EVEN_BPM_HELP)
        self.btn_even_bpm.setChecked(True)
        self.btn_even_bpm.toggled.connect(self.set_even_bpm)
        self._tempo_problem = None
        self.piano_roll.clips_changed.connect(self._retry_even_bpm)

        lbl_grid = QLabel("Grid")
        lbl_grid.setStyleSheet(f"color: {MAT_TEXT};")
        lbl_grid.setFixedHeight(30)
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
        toolbar.addWidget(self.btn_snap)
        toolbar.addWidget(self.btn_even_bpm)
        toolbar.addWidget(lbl_grid)
        toolbar.addWidget(self.combo_grid)
        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.live_waveform)

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
        self._drop_unused_clip_files()

    def _on_tracks_edited(self):
        self.piano_roll.load_model(keep_selection=True)
        self._update_history_buttons()
        self.check_placeholder()
        self.tracks_changed.emit()

    def _on_active_track_changed(self, track_id):
        self.piano_roll.current_track_id = track_id

    def _on_audio_dropped(self, path):
        try:
            self.add_audio_clip(path)
        except AudioFormatError as e:
            QMessageBox.warning(self, "Audio Clip", str(e))

    def set_song_folder(self, folder):
        from korshmer.project_management.song_folder import set_open_song_folder

        self.vm.song_folder = str(folder) if folder else None
        set_open_song_folder(folder)

    def repoint_clips(self, old_folder, new_folder):
        old_root = os.path.abspath(str(old_folder))
        for clip in self.model.clips:
            paths = [clip.path] + ([clip.source[0]] if clip.source else [])
            moved = []
            for path in paths:
                inside = os.path.abspath(path).startswith(old_root + os.sep)
                moved.append(
                    os.path.join(
                        str(new_folder), os.path.relpath(path, old_root)
                    )
                    if inside
                    else path
                )
            clip.path = moved[0]
            if clip.source:
                clip.source = (moved[1], *clip.source[1:])
            if clip.tempo_base:
                base = clip.tempo_base[0]
                if os.path.abspath(base).startswith(old_root + os.sep):
                    base = os.path.join(
                        str(new_folder), os.path.relpath(base, old_root)
                    )
                clip.tempo_base = (base, clip.tempo_base[1])

    def keep_clips_in(self, song_folder):
        from korshmer.project_management.song_folder import kept_in_song

        self.piano_roll.sync_model_from_scene()
        self.set_song_folder(song_folder)
        moved = 0
        for clip in self.model.clips:
            kept = kept_in_song(clip.path, song_folder)
            moved += kept != clip.path
            clip.path = kept
            if clip.tempo_base is not None:
                clip.tempo_base = (
                    kept_in_song(clip.tempo_base[0], song_folder),
                    clip.tempo_base[1],
                )
            if clip.source is not None:
                clip.source = (
                    kept_in_song(clip.source[0], song_folder),
                    *clip.source[1:],
                )
        return moved

    def missing_clip_names(self):
        return [
            clip.name
            for clip in self.model.clips
            if not os.path.isfile(clip.path)
        ]

    def clear_editor(self):
        self.wait_for_waveforms()
        self.model.snapshot()
        self.model.clear()
        self.piano_roll.load_model()
        self.piano_roll.focus_notes()
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
        evened = self.model.normalized is not None and not replace
        if tempo and float(tempo) > 0 and not evened:
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
        was_empty = not self.model.notes
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
        if replace or was_empty:
            self.piano_roll.focus_notes()
        self._on_model_changed()
        self._tempo_problem = None
        self.apply_even_bpm()
        return created

    def _aligned_clip(self, clip, tempo_map, bpm):
        from korshmer.audio_processing.tempo_align import align_clip_to_tempo

        folder = os.path.dirname(os.path.abspath(clip.path))
        if os.path.basename(folder) != STEMS_FOLDER:
            folder = os.path.join(str(DATA_DIR), ALIGNED_CLIPS_DIR)
        origin = getattr(clip, "origin", clip.start)
        return align_clip_to_tempo(
            clip.path, max(0.0, origin), tempo_map, bpm, folder
        )

    def aligned_song_audio(self, path):
        state = self.model.normalized
        if state is None:
            return path
        from korshmer.audio_processing.tempo_align import align_clip_to_tempo

        aligned, _duration = align_clip_to_tempo(
            path,
            0.0,
            state["map"],
            state["bpm"],
            os.path.join(str(DATA_DIR), ALIGNED_CLIPS_DIR),
        )
        return aligned

    def _after_tempo_switch(self):
        self.piano_roll.tempo = self.model.tempo
        self.piano_roll.load_model()
        for clip in self.model.clips:
            if not clip.waveform:
                self._queue_waveform(clip)
        self._start_visible_waveforms()
        self.tempo_changed.emit(float(self.model.tempo))
        self._on_model_changed()

    def _report_tempo_problem(self, text):
        if text == self._tempo_problem:
            return
        self._tempo_problem = text
        logging.getLogger(__name__).error(text)
        QMessageBox.warning(self, "Even BPM", text)

    def apply_even_bpm(self):
        if not self.btn_even_bpm.isChecked() or self.model.tempo_map is None:
            return False
        self.piano_roll.sync_model_from_scene()
        try:
            self.model.normalize_tempo(self._aligned_clip)
        except TempoNormalizeError as problem:
            self._report_tempo_problem(str(problem))
            return False
        except Exception as problem:
            self._report_tempo_problem(
                f"Even BPM was not applied: {problem}. Turn off Even BPM "
                "to keep the tempo changes of the song."
            )
            return False
        self._tempo_problem = None
        self._after_tempo_switch()
        return True

    def align_clip_to_song(self, clip):
        if clip is None or self.model.normalized is None:
            return False
        try:
            if not self.model.normalize_clip(clip, self._aligned_clip):
                return False
        except TempoNormalizeError as problem:
            self._report_tempo_problem(str(problem))
            return False
        except Exception as problem:
            self._report_tempo_problem(
                f"The clip '{clip.name}' was not moved onto the even "
                f"tempo: {problem}. Turn off Even BPM or remove that clip."
            )
            return False
        self.piano_roll.load_model(keep_selection=True)
        self._queue_waveform(clip)
        self._start_visible_waveforms()
        self._on_model_changed()
        return True

    def _retry_even_bpm(self):
        if self._tempo_problem is not None:
            self.apply_even_bpm()

    def set_even_bpm(self, enabled):
        if enabled:
            return self.apply_even_bpm()
        self._tempo_problem = None
        self.piano_roll.sync_model_from_scene()
        if not self.model.restore_tempo():
            return False
        self._after_tempo_switch()
        return True

    def to_editor_time(self, seconds):
        return self.model.to_editor_time(seconds)

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
        path = usable_audio_path(path)
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

    def clip_menu_actions(self, position=None):
        clip_id = self.piano_roll.clip_id_for_menu(position)
        if clip_id is None or self.model.clip_by_id(clip_id) is None:
            return []
        return [
            (label, lambda _=False, run=run: run(clip_id))
            for label, run in (
                (CLIP_ALIGN_LABEL, self.align_clip_to_beat),
                (CLIP_NOISE_LABEL, self.remove_clip_noise),
                (CLIP_QUALITY_LABEL, self.improve_clip_quality),
                (CLIP_ALL_LABEL, self.fix_clip_automatically),
                (CLIP_EARLIER_LABEL, self.move_clip_one_beat_earlier),
                (CLIP_LATER_LABEL, self.move_clip_one_beat_later),
            )
        ]

    def move_clip_one_beat_earlier(self, clip_id):
        return self.move_clip_by_beats(clip_id, -1.0)

    def move_clip_one_beat_later(self, clip_id):
        return self.move_clip_by_beats(clip_id, 1.0)

    def move_clip_by_beats(self, clip_id, beats):
        self.piano_roll.sync_model_from_scene()
        clip = self.model.clip_by_id(clip_id)
        if clip is None:
            return False
        start = self._shifted_clip_start(clip, beats)
        if start < 0.0:
            self.clip_edit_status.emit(
                f"'{clip.name}' is at the start of the song and cannot move earlier"
            )
            return False
        self.model.snapshot()
        clip.start = start
        self.piano_roll.load_model(keep_selection=True)
        self._on_model_changed()
        self.clip_edit_status.emit(
            f"'{clip.name}' moved one beat "
            f"{'later' if beats > 0 else 'earlier'}"
        )
        return True

    def _clip_beats_at(self):
        tempo_map = self.model.tempo_map
        if tempo_map is not None:
            return tempo_map.beats_at
        tempo = float(self.model.tempo)
        origin = float(self.model.grid_offset)
        return lambda seconds: (seconds - origin) * tempo / 60.0

    def _clip_edit_job(self, clip, steps):
        from korshmer.audio_processing import clip_polish

        audio = clip_polish.ClipAudio(
            clip.path,
            clip.start,
            clip.offset,
            clip.duration,
            clip.source[0] if clip.source is not None else None,
        )
        beats_at = self._clip_beats_at()
        steady = self.model.tempo_map is None and self.model.normalized is None
        song_bpm = float(self.model.tempo) if steady else None

        song = self.get_midi_from_editor() if "quality" in steps else None
        grid_steps = self.current_grid_divisions()
        tempo = float(self.piano_roll.tempo)
        render = self.notes_audio
        heard = {}

        def notes(sample_rate):
            if sample_rate not in heard:
                heard[sample_rate] = render(song, tempo, sample_rate)
            return heard[sample_rate]

        fix = (
            clip_polish.fix_clip_parts
            if set(steps) == {"noise", "quality", "align"}
            else clip_polish.fix_clip
        )

        def job(report, cancelled):
            return fix(
                audio,
                steps,
                beats_at,
                song_bpm,
                report,
                cancelled,
                notes=notes if song is not None else None,
                grid_steps=grid_steps,
            )

        return job

    def _start_clip_edit(self, clip_id, label, steps):
        self.piano_roll.sync_model_from_scene()
        clip = self.model.clip_by_id(clip_id)
        if clip is None or self.clip_edit_running(clip_id):
            return None
        worker = ClipEditWorker(
            clip_id, label, self._clip_edit_job(clip, steps), parent=self
        )
        worker.edit_ready.connect(self._on_clip_edit_ready)
        worker.edit_failed.connect(self._on_clip_edit_failed)
        worker.edit_cancelled.connect(self._on_clip_edit_cancelled)
        worker.edit_progress.connect(self._on_clip_edit_progress)
        worker.finished.connect(self._refresh_clip_edit_bar)
        self._clip_edit_workers.append(worker)
        self.clip_edit_status.emit(f"{label}: working on '{clip.name}'")
        self._show_clip_edit_progress(label, clip.name, "Starting", 0)
        worker.start()
        return worker

    def _show_clip_edit_progress(self, label, name, stage, percent):
        self.clip_edit_label.setText(f"{label} on '{name}': {stage}")
        self.clip_edit_progress.setValue(percent)
        self.clip_edit_bar.setVisible(True)

    def _on_clip_edit_progress(self, clip_id, label, stage, percent):
        clip = self.model.clip_by_id(clip_id)
        name = clip.name if clip is not None else "the clip"
        self._show_clip_edit_progress(label, name, stage, percent)

    def _refresh_clip_edit_bar(self):
        self.clip_edit_bar.setVisible(self.clip_edit_running())

    def cancel_clip_edits(self):
        for worker in self._clip_edit_workers:
            worker.requestInterruption()

    def _on_clip_edit_cancelled(self, clip_id, label):
        clip = self.model.clip_by_id(clip_id)
        name = clip.name if clip is not None else "the clip"
        self.clip_edit_status.emit(
            f"{label}: cancelled, '{name}' was left as it was"
        )
        self._refresh_clip_edit_bar()

    def _on_tempo_followed(self, earlier_bpm, bpm):
        if earlier_bpm <= 0 or not self.model.clips_to_follow_tempo(
            earlier_bpm
        ):
            return
        self._tempo_follow_timer.start()

    def stretch_clips_to_tempo(self):
        from korshmer.audio_processing import clip_polish

        self._tempo_follow_timer.stop()
        if (
            self.model.tempo_map is not None
            or self.model.normalized is not None
        ):
            return []
        bpm = float(self.model.tempo)
        started = []
        for clip in list(self.model.clips):
            if clip.tempo_base is None or clip.source is not None:
                continue
            if self.clip_edit_running(clip.id):
                self._tempo_follow_timer.start()
                continue
            base_path, recorded_bpm = clip.tempo_base
            if not os.path.isfile(base_path):
                continue
            if not (
                clip_polish.SLOWEST_FOLLOW
                <= recorded_bpm / bpm
                <= clip_polish.FASTEST_FOLLOW
            ):
                self.clip_edit_status.emit(
                    f"{CLIP_FOLLOW_LABEL}: '{clip.name}' was recorded at "
                    f"{recorded_bpm:g} BPM, too far from {bpm:g} BPM to stretch, "
                    "so it keeps its recorded speed"
                )
                continue

            def job(
                report,
                cancelled,
                base_path=base_path,
                recorded_bpm=recorded_bpm,
            ):
                report(clip_polish.FOLLOW_STAGE, 0, 1)
                path, length = clip_polish.follow_tempo_file(
                    base_path, recorded_bpm, bpm
                )
                return {
                    "follow": bpm,
                    "path": path,
                    "length": length,
                    "made": [path] if path != base_path else [],
                }

            worker = ClipEditWorker(
                clip.id, CLIP_FOLLOW_LABEL, job, parent=self
            )
            worker.edit_ready.connect(self._on_clip_edit_ready)
            worker.edit_failed.connect(self._on_clip_edit_failed)
            worker.edit_progress.connect(self._on_clip_edit_progress)
            worker.finished.connect(self._refresh_clip_edit_bar)
            self._clip_edit_workers.append(worker)
            self._show_clip_edit_progress(
                CLIP_FOLLOW_LABEL, clip.name, "Starting", 0
            )
            worker.start()
            started.append(worker)
        return started

    def _apply_tempo_follow(self, clip, result):
        if abs(result["follow"] - float(self.model.tempo)) > 1e-6:
            self._tempo_follow_timer.start()
            return
        if not self.model.retune_clip(
            clip.id, result["path"], result["length"]
        ):
            return
        self.piano_roll.load_model(keep_selection=True)
        self._queue_waveform(clip)
        self._start_visible_waveforms()
        self._on_model_changed()
        self._refresh_clip_edit_bar()
        self.clip_edit_status.emit(
            f"{CLIP_FOLLOW_LABEL}: '{clip.name}' now plays at "
            f"{float(self.model.tempo):g} BPM"
        )

    def _drop_unused_clip_files(self):
        in_use = self.model.clip_files_in_use() | self._saved_clip_files
        for path in sorted(self._clip_edit_files - in_use):
            self._clip_edit_files.discard(path)
            try:
                os.remove(path)
            except OSError as problem:
                logging.getLogger(__name__).warning(
                    f"Could not remove the unused clip file {path}: {problem}"
                )

    def clip_edit_running(self, clip_id=None):
        self._clip_edit_workers = [
            worker
            for worker in self._clip_edit_workers
            if worker.isRunning() or not worker.isFinished()
        ]
        return any(
            clip_id is None or worker.clip_id == clip_id
            for worker in self._clip_edit_workers
        )

    def wait_for_clip_edits(self, timeout_ms=600000):
        for worker in list(self._clip_edit_workers):
            worker.wait(timeout_ms)
        QApplication.processEvents()

    def align_clip_to_beat(self, clip_id):
        return self._start_clip_edit(clip_id, CLIP_ALIGN_LABEL, ("align",))

    def remove_clip_noise(self, clip_id):
        return self._start_clip_edit(clip_id, CLIP_NOISE_LABEL, ("noise",))

    def improve_clip_quality(self, clip_id):
        return self._start_clip_edit(clip_id, CLIP_QUALITY_LABEL, ("quality",))

    def fix_clip_automatically(self, clip_id):
        return self._start_clip_edit(
            clip_id, CLIP_ALL_LABEL, ("noise", "quality", "align")
        )

    def _shifted_clip_start(self, clip, beats):
        tempo_map = self.model.tempo_map
        if tempo_map is not None:
            return tempo_map.seconds_at(tempo_map.beats_at(clip.start) + beats)
        return clip.start + beats * 60.0 / float(self.model.tempo)

    def _fixed_clip_start(self, clip, result):
        start = clip.start
        if result.get("shift"):
            start = self._shifted_clip_start(clip, result["shift"])
            if start < 0.0:
                start = self._shifted_clip_start(clip, result["shift"] + 1.0)
        return start

    def _clip_fix_notes(self, result, seconds, moved):
        from korshmer.audio_processing import clip_level

        notes = []
        if "readings" in result:
            notes.append(clip_level.summary(result["readings"]))
        if "tempo" in result:
            notes.append(
                "tempo matched from {:.1f} to {:.1f} BPM".format(
                    *result["tempo"]
                )
            )
        placement = self._clip_placement_text(result, seconds, moved)
        if placement:
            notes.append(placement)
        if "on_grid" in result:
            count, largest = result["on_grid"]
            notes.append(
                f"{count} sounds inside it were stretched onto the "
                f"{self.combo_grid.currentText()} grid "
                f"(largest move {largest * 1000.0:.0f} ms)"
            )
        return notes

    def _replace_with_parts(self, clip, label, result):
        from korshmer.audio_processing import clip_level

        self.model.snapshot()
        self.model.remove_clip(clip.id)
        said = []
        for number, fixed in enumerate(result["parts"], 1):
            readings = fixed.get("readings")
            improved = bool(readings) and not clip_level.unchanged(readings)
            part = self.model.add_clip(
                fixed["path"],
                start=fixed["start"],
                name=f"{clip.name} {number}",
                kind=clip.kind,
                gain=1.0 if improved else clip.gain,
            )
            part.color = clip.color
            part.muted = clip.muted
            start = self._fixed_clip_start(part, fixed)
            moved = abs(start - part.start) >= CLIP_MOVE_TOLERANCE_S
            notes = self._clip_fix_notes(fixed, start - part.start, moved)
            part.start = start
            self._queue_waveform(part)
            said.append(f"part {number}: " + "; ".join(notes))
        self.piano_roll.load_model()
        self._start_visible_waveforms()
        self._on_model_changed()
        self._drop_unused_clip_files()
        self._refresh_clip_edit_bar()
        self.clip_edit_status.emit(
            f"{label}: '{clip.name}' was split into {len(said)} parts and "
            "each was fixed on its own. " + " | ".join(said)
        )

    def _on_clip_edit_ready(self, clip_id, label, result):
        from korshmer.audio_processing import clip_level

        self._clip_edit_files.update(result.get("made", ()))
        self.piano_roll.sync_model_from_scene()
        clip = self.model.clip_by_id(clip_id)
        if clip is None:
            self._drop_unused_clip_files()
            return
        if "follow" in result:
            self._apply_tempo_follow(clip, result)
            return
        if "parts" in result:
            self._replace_with_parts(clip, label, result)
            return
        trimmed = "trim" in result
        if trimmed:
            self.model.snapshot()
            clip.trim(*result["trim"])
        start = self._fixed_clip_start(clip, result)
        moved = abs(start - clip.start) >= CLIP_MOVE_TOLERANCE_S
        notes = self._clip_fix_notes(result, start - clip.start, moved)
        if trimmed:
            notes.insert(0, "silence at its ends was trimmed off")
        new_path = result["path"]
        if new_path == clip.path and not moved and not trimmed:
            self.clip_edit_status.emit(
                f"{label}: nothing changed in '{clip.name}': "
                + "; ".join(notes)
            )
            self._refresh_clip_edit_bar()
            return
        if not trimmed:
            self.model.snapshot()
        if new_path != clip.path:
            clip.path = new_path
            clip.waveform = []
            clip.tempo_base = None
            if clip.source is not None and "source" in result:
                clip.source = (result["source"], *clip.source[1:])
            if "length" in result:
                clip.rescale(result["length"])
            readings = result.get("readings")
            if readings and not clip_level.unchanged(readings):
                clip.gain = 1.0
        if moved:
            clip.start = start
        self.piano_roll.load_model(keep_selection=True)
        if not clip.waveform:
            self._queue_waveform(clip)
            self._start_visible_waveforms()
        self._on_model_changed()
        self._drop_unused_clip_files()
        self._refresh_clip_edit_bar()
        ready = f"{label}: '{clip.name}' is ready"
        self.clip_edit_status.emit(
            f"{ready}: {'; '.join(notes)}" if notes else ready
        )

    @staticmethod
    def _clip_placement_text(result, seconds, moved):
        from korshmer.audio_processing import clip_polish

        anchor = result.get("anchor")
        if anchor is None:
            return ""
        if anchor == clip_polish.SILENT:
            return "it is silent, so it stayed where it was"
        if not moved:
            return "it is already on the beat"
        if anchor == clip_polish.ON_NEIGHBOUR:
            return (
                f"it is too short to read a beat from, so it moved "
                f"{seconds:+.3f} s together with the part next to it"
            )
        if anchor == clip_polish.ON_SOUNDS:
            return (
                f"it has no steady beat, so its sounds were moved "
                f"{seconds:+.3f} s onto the nearest grid lines"
            )
        if anchor == clip_polish.ON_FIRST_SOUND:
            return (
                f"it has no steady beat, so its first sound was moved "
                f"{seconds:+.3f} s onto the nearest grid line"
            )
        return f"it was moved {seconds:+.3f} s onto the beat"

    def _on_clip_edit_failed(self, clip_id, label, problem):
        clip = self.model.clip_by_id(clip_id)
        name = clip.name if clip is not None else "the clip"
        text = f"{label} could not finish for '{name}': {problem}"
        self._refresh_clip_edit_bar()
        self.clip_edit_status.emit(text)
        if label != CLIP_FOLLOW_LABEL:
            QMessageBox.warning(self, label, text)

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
        clip = self.model.clip_by_id(clip_id)
        if clip is None:
            return
        measured = getattr(self.sender(), "path", clip.path)
        if measured != clip.path:
            self._queue_waveform(clip)
            self._start_visible_waveforms()
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

    def selected_slide_count(self):
        return self.piano_roll.selected_slide_count()

    def remove_slide_selected(self):
        return self.piano_roll.remove_slide_selected()

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
        self._saved_clip_files |= {clip.path for clip in self.model.clips}
        self._saved_clip_files |= {
            clip.source[0] for clip in self.model.clips if clip.source
        }
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
        self.piano_roll.focus_notes()
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
