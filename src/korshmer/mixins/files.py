import gc
import logging
import os
import shutil
import subprocess
import sys
import time

import pretty_midi
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QWidget,
)

from korshmer.audio.loader import check_audio_file_type
from korshmer.constants import DATA_DIR, FLUID_DIR, SF3_PATH
from korshmer.event_bus import Event, publish
from korshmer.exceptions import AudioFormatError, InvalidFileError
from korshmer.mixins.path_guard import reject_unsafe_path
from korshmer.services.file_management import FileManagementService
from korshmer.ui.theming.color_utils import MAT_PRIMARY
from korshmer.utils.filesystem import is_midi_path
from korshmer.workers.midi_preview import MidiPreviewThread

_fs = FileManagementService


class FilesMixin:
    def release_file_lock(self):
        try:
            if hasattr(self.control_bar, "player"):
                self.control_bar.player.stop()
                self.control_bar.player.current_path = None
            if hasattr(self.spectrogram, "spectrogram_image"):
                self.spectrogram.spectrogram_image = None
                self.spectrogram.update()
        except Exception:
            pass

    def export_file_action(self, source_path):
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "Error", "File or folder not found.")
            return

        base_name = os.path.basename(source_path)

        if os.path.isfile(source_path):
            _, ext = os.path.splitext(base_name)
            target, _ = QFileDialog.getSaveFileName(
                self, "Export File", base_name, f"File (*{ext})"
            )
            if target:
                if reject_unsafe_path(self, target):
                    return
                try:
                    shutil.copy2(source_path, target)
                    QMessageBox.information(
                        self, "Success", f"File exported to:\n{target}"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")

        elif os.path.isdir(source_path):
            target_dir = QFileDialog.getExistingDirectory(
                self, "Export Folder Destination"
            )
            if target_dir:
                if reject_unsafe_path(self, target_dir):
                    return
                dest_path = os.path.join(target_dir, base_name)
                try:
                    if os.path.exists(dest_path):
                        if (
                            QMessageBox.question(
                                self,
                                "Overwrite",
                                "Destination folder exists. Overwrite?",
                                QMessageBox.Yes | QMessageBox.No,
                            )
                            != QMessageBox.Yes
                        ):
                            return
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                    QMessageBox.information(
                        self, "Success", f"Folder exported to:\n{dest_path}"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def on_info_click(self, path):
        try:
            info = _fs.get_file_stats(path)
            if not info:
                QMessageBox.warning(
                    self, "Error", "Could not retrieve information."
                )
                return
            type_label = "Folder" if info["is_dir"] else "File"
            info_text = (
                f"<div><h3 style='color:{MAT_PRIMARY}'>{info['basename']}</h3>"
                f"<p><b>Type:</b> {type_label}</p><p><b>Path:</b> {path}</p><hr>"
                f"<p><b>Size:</b> {info['size_formatted']}</p><p><b>Created:</b> {info['created']}</p>"
                f"<p><b>Last Modified:</b> {info['modified']}</p></div>"
            )
            msg = QMessageBox(self)
            msg.setWindowTitle("File Details")
            msg.setTextFormat(Qt.RichText)
            msg.setText(info_text)
            msg.setIcon(QMessageBox.Information)
            msg.exec()
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Could not retrieve information: {e}"
            )

    def on_delete_click(self, path, list_widget, item_ref, is_output):
        current_src = getattr(self.control_bar.player, "current_path", None)
        is_active = current_src == path
        if is_midi_path(path):
            temp_wav = os.path.join(os.path.dirname(path), "temp_preview.wav")
            if current_src == temp_wav:
                is_active = True
            if is_active:
                self.release_file_lock()
                self.file_label.setText("No file selected")
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass
        elif is_active:
            self.release_file_lock()
            self.file_label.setText("No file selected")
        if not is_output:
            row = list_widget.row(item_ref)
            list_widget.takeItem(row)
            if path in self.file_list_paths:
                self.file_list_paths.remove(path)
            if list_widget.count() == 0:
                publish(Event.CONVERT_BUTTON_STATE, enabled=False)
            self.save_input_list()
        else:
            res = QMessageBox.question(
                self,
                "Delete",
                f"Move '{os.path.basename(path)}' to recycle bin?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                recycle_bin = os.path.join(DATA_DIR, "_recycle_bin")
                os.makedirs(recycle_bin, exist_ok=True)
                try:
                    base_name = os.path.basename(path)
                    new_name = f"{int(time.time())}_{base_name}"
                    shutil.move(path, os.path.join(recycle_bin, new_name))
                    row = list_widget.row(item_ref)
                    list_widget.takeItem(row)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete: {e}")

    def save_input_list(self):
        _fs.save_file_list(self.file_list_paths)

    def load_input_list(self):
        for p in _fs.load_file_list():
            self.add_file(p, save=False)

    def sync_convert_button(self):
        publish(Event.CONVERT_BUTTON_STATE, enabled=bool(self.file_list_paths))

    def refresh_lists(self):
        for i in range(self.input_list.count() - 1, -1, -1):
            item = self.input_list.item(i)
            path = item.data(Qt.UserRole)
            if not os.path.exists(path):
                self.input_list.takeItem(i)
                if path in self.file_list_paths:
                    self.file_list_paths.remove(path)
        self.save_input_list()
        self.load_output_view()

    def load_output_view(self):
        self.output_list.clear()
        if self.current_output_dir != DATA_DIR:
            back_item = QListWidgetItem()
            widget = QWidget()
            lay = QHBoxLayout(widget)
            lay.setContentsMargins(4, 4, 4, 4)
            btn = QPushButton("📁 .. (Back to main output folder)")
            btn.setMinimumHeight(35)
            btn.setStyleSheet("text-align: right;")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda: [
                    setattr(self, "current_output_dir", DATA_DIR),
                    self.load_output_view(),
                ]
            )
            lay.addWidget(btn)
            back_item.setSizeHint(widget.sizeHint())
            self.output_list.addItem(back_item)
            self.output_list.setItemWidget(back_item, widget)
        for name, full_path in _fs.list_output_files(self.current_output_dir):
            item = QListWidgetItem()
            widget = self._create_list_item_widget(
                name, full_path, self.output_list, item, is_output=True
            )
            sz = widget.sizeHint()
            sz.setHeight(46)
            item.setSizeHint(sz)
            item.setData(Qt.UserRole, full_path)
            self.output_list.addItem(item)
            self.output_list.setItemWidget(item, widget)

    def add_file(self, path, save=True):
        try:
            path = _fs.resolve_path(path)
        except InvalidFileError as e:
            logging.warning(f"Skipping unsafe file path: {e}")
            publish(
                Event.STATUS_UPDATE, text="Skipped file with an unsafe path"
            )
            return False
        if _fs.is_valid_audio(path) and os.path.isfile(path):
            try:
                check_audio_file_type(path)
            except AudioFormatError as e:
                logging.warning(f"Skipping non-audio file {path}: {e}")
                publish(
                    Event.STATUS_UPDATE,
                    text=f"Skipped non-audio file: {os.path.basename(path)}",
                )
                return False
        if path not in self.file_list_paths:
            self.file_list_paths.append(path)
            item = QListWidgetItem()
            widget = self._create_list_item_widget(
                os.path.basename(path),
                path,
                self.input_list,
                item,
                is_output=False,
            )
            sz = widget.sizeHint()
            sz.setHeight(46)
            item.setSizeHint(sz)
            item.setData(Qt.UserRole, path)
            self.input_list.addItem(item)
            self.input_list.setItemWidget(item, widget)
            if len(self.file_list_paths) == 1:
                self.input_list.setCurrentItem(item)
                self._last_selected_list = "input"
                publish(
                    Event.AUDIO_LOAD_REQUESTED,
                    path=path,
                    analyze=True,
                    load_player=True,
                )
                self.file_label.setText(os.path.basename(path))
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            if save:
                self.save_input_list()

            self.input_list.scrollToItem(item)
            return True
        return False

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def on_files_dropped_list(self, files):
        count = 0
        for f in files:
            if _fs.is_valid_file(f) and self.add_file(f):
                count += 1
        if count > 0:
            publish(
                Event.STATUS_UPDATE, text=f"Added {count} files via Drag & Drop"
            )

    def dropEvent(self, event: QDropEvent):
        pass

    def clear_files(self):
        publish(Event.FILE_LOCK_RELEASED)
        self.file_list_paths = []
        self.input_list.clear()
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)
        self.file_label.setText("No file selected")
        publish(Event.STATUS_UPDATE, text="Ready")
        self.save_input_list()

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "Audio/MIDI (*.mp3 *.wav *.flac *.ogg *.aiff *.aif *.mid *.midi)",
        )
        for f in files:
            self.add_file(f)

    def handle_midi_selection(self, midi_path, load_editor=False):
        try:
            publish(Event.FILE_LOCK_RELEASED)
            QApplication.processEvents()

            current_bpm = float(self.spin_bpm.value())

            if load_editor:
                try:
                    if hasattr(self.arranger, "load_midi_file"):
                        self.arranger.load_midi_file(
                            midi_path, replace=True, tempo=current_bpm
                        )
                    else:
                        pm = pretty_midi.PrettyMIDI(
                            midi_path, initial_tempo=current_bpm
                        )
                        self.arranger.piano_roll.load_notes(pm)
                    self.tabs.setCurrentWidget(self.arranger)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Error", f"Could not load MIDI file: {e}"
                    )

            preview_wav = os.path.join(
                os.path.dirname(midi_path), "temp_preview.wav"
            )
            if os.path.exists(preview_wav):
                try:
                    os.remove(preview_wav)
                except PermissionError:
                    pass

            if os.path.exists(SF3_PATH):
                publish(Event.STATUS_UPDATE, text="Generating preview...")
                publish(Event.CONVERT_BUTTON_STATE, enabled=False)

                if (
                    hasattr(self, "preview_thread")
                    and self.preview_thread.isRunning()
                ):
                    self.preview_thread.terminate()
                    self.preview_thread.wait()

                self.preview_thread = MidiPreviewThread(
                    midi_path, FLUID_DIR, SF3_PATH, preview_wav, bpm=current_bpm
                )

                def on_preview_success(wav_path):
                    publish(Event.CONVERT_BUTTON_STATE, enabled=True)
                    publish(
                        Event.AUDIO_LOAD_REQUESTED,
                        path=wav_path,
                        analyze=True,
                        load_player=True,
                    )
                    publish(
                        Event.STATUS_UPDATE,
                        text=f"Ready: {os.path.basename(midi_path)}",
                    )

                def on_preview_fail(err_msg):
                    publish(Event.CONVERT_BUTTON_STATE, enabled=True)
                    publish(
                        Event.STATUS_UPDATE, text="Preview Generation Failed"
                    )
                    QMessageBox.critical(
                        self, "Preview Error", f"An error occurred:\n{err_msg}"
                    )
                    publish(
                        Event.AUDIO_LOAD_REQUESTED,
                        path=midi_path,
                        analyze=False,
                        load_player=True,
                    )

                self.preview_thread.preview_ready.connect(on_preview_success)
                self.preview_thread.preview_failed.connect(on_preview_fail)
                self.preview_thread.start()
            else:
                publish(
                    Event.AUDIO_LOAD_REQUESTED,
                    path=midi_path,
                    analyze=False,
                    load_player=True,
                )
        finally:
            gc.collect()

    def add_midi_lanes_to_editor(self, midi_path):
        if not os.path.exists(midi_path):
            QMessageBox.warning(self, "Error", "MIDI file not found.")
            return
        try:
            current_bpm = float(self.spin_bpm.value())
            self.arranger.load_midi_file(
                midi_path, replace=False, tempo=current_bpm
            )
            self.tabs.setCurrentWidget(self.arranger)
            publish(
                Event.STATUS_UPDATE,
                text=f"Added lanes from {os.path.basename(midi_path)}",
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not add MIDI lanes: {e}")

    def on_play_click(self, path, is_output):
        self.control_bar.btn_ab.setChecked(False)
        self.arranger.piano_roll.is_monitoring = False

        if os.path.isdir(path):
            if is_output:
                self.current_output_dir = path
                self.load_output_view()
            return

        if is_midi_path(path):
            self.handle_midi_selection(path, load_editor=False)
        elif path.lower().endswith(
            (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
        ):
            publish(
                Event.AUDIO_LOAD_REQUESTED,
                path=path,
                analyze=True,
                load_player=True,
            )
            publish(
                Event.STATUS_UPDATE,
                text=f"Ready to play: {os.path.basename(path)}",
            )

    def on_double_click(self, item):
        path = item.data(Qt.UserRole)
        if not path:
            return
        is_output = self.sender() == self.output_list
        self.on_play_click(path, is_output)

    def open_file_location_action(self, explicit_path=None):
        path = explicit_path
        if not path:
            item = self.output_list.currentItem()
            if not item:
                return
            path = item.data(Qt.UserRole)

        folder = os.path.dirname(path) if os.path.isfile(path) else path

        if sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.call(["open", folder])

    def on_input_file_select(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.file_label.setText(os.path.basename(path))

    def on_input_file_double_click(self, item):
        self.on_double_click(item)

    def on_output_file_double_click(self, item):
        self.on_double_click(item)
