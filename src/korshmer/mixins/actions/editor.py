import logging
import numbers
import os
import shutil
import time
from functools import partial

import pretty_midi
from PySide6.QtWidgets import QFileDialog, QMessageBox

from korshmer.constants import DATA_DIR, FLUID_DIR, NOTE_NAMES, SF3_PATH
from korshmer.event_bus import Event, publish
from korshmer.mixins.path_guard import reject_unsafe_path
from korshmer.utils.tempo import validate_bpm


def _layers_dir():
    out_dir = os.path.join(DATA_DIR, "Processed_MIDI")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class EditorActionsMixin:
    def _editor(self):
        return getattr(self, "arranger", None)

    def _editor_tab_active(self):
        tabs = getattr(self, "tabs", None)
        arranger = self._editor()
        if tabs is None or arranger is None:
            return False
        return tabs.currentWidget() is arranger

    def _show_editor_tab(self):
        tabs = getattr(self, "tabs", None)
        arranger = self._editor()
        if tabs is not None and arranger is not None:
            tabs.setCurrentWidget(arranger)

    def _current_bpm(self):
        try:
            value = self.spin_bpm.value()
        except AttributeError:
            return 120.0
        return validate_bpm(value)

    def _editor_midi(self):
        arranger = self._editor()
        if arranger is None:
            return None
        try:
            return arranger.get_midi_from_editor()
        except Exception as e:
            logging.error(f"Could not read editor content: {e}")
            return None

    def _editor_grid_offset(self):
        roll = getattr(self._editor(), "piano_roll", None)
        value = getattr(roll, "grid_offset", None)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return float(value) if value > 0 else None
        return None

    def _resolve_key(self, pm=None):
        root = self.combo_root.currentText()
        scale = self.combo_scale.currentText()
        needs_root = "Auto" in root
        needs_scale = "Auto" in scale
        if not needs_root and not needs_scale:
            return root, scale
        detected_root, detected_scale = "C", "Major"
        if pm is not None and any(
            n for inst in pm.instruments if not inst.is_drum for n in inst.notes
        ):
            try:
                from korshmer.analysis import detect_midi_key

                os.makedirs(DATA_DIR, exist_ok=True)
                tmp = os.path.join(DATA_DIR, "_key_probe.mid")
                pm.write(tmp)
                detected_root, detected_scale = detect_midi_key(tmp)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            except Exception as e:
                logging.debug(f"Key detection fallback: {e}")
        if needs_root:
            root = detected_root
        if needs_scale:
            scale = detected_scale
        if root not in NOTE_NAMES:
            root = "C"
        return root, scale

    def on_piano_roll_quantize(self):
        arranger = self._editor()
        if arranger is None:
            return
        strength = 1.0
        slider = getattr(self, "slider_quantize", None)
        if slider is not None:
            strength = max(0.05, slider.value() / 100.0)
        arranger.quantize_selected(strength=strength)
        publish(Event.STATUS_UPDATE, text="Quantized selection")

    def on_piano_roll_humanize(self):
        arranger = self._editor()
        if arranger is None:
            return
        timing = 0.01
        velocity = 8
        slider_t = getattr(self, "slider_human_time", None)
        slider_v = getattr(self, "slider_human_vel", None)
        if slider_t is not None:
            timing = max(0.001, slider_t.value() / 100.0)
        if slider_v is not None:
            velocity = max(1, int(slider_v.value()))
        arranger.humanize_selected(timing, velocity)
        publish(Event.STATUS_UPDATE, text="Humanized selection")

    def on_piano_roll_transpose_up(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.transpose_selected(1)

    def on_piano_roll_transpose_down(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.transpose_selected(-1)

    def on_piano_roll_octave_up(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.transpose_selected(12)

    def on_piano_roll_octave_down(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.transpose_selected(-12)

    def on_piano_roll_duplicate(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.duplicate_selected()

    def on_piano_roll_delete(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.delete_selected()

    def on_piano_roll_copy(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.copy_selected()

    def on_piano_roll_paste(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.paste()

    def on_piano_roll_select_all(self):
        arranger = self._editor()
        if arranger is not None:
            arranger.select_all()

    def on_piano_roll_add_audio_clip(self):
        self.editor_add_audio_clip_dialog()

    def editor_add_audio_clip_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Audio Clip to Editor",
            str(DATA_DIR),
            "Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.aif)",
        )
        if path and hasattr(self, "send_audio_to_editor"):
            self.send_audio_to_editor(path)

    def pick_stems_folder_for_editor(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select a stems output folder", str(DATA_DIR)
        )
        if not folder:
            return
        if reject_unsafe_path(self, folder):
            return
        if hasattr(self, "load_output_folder_session"):
            loaded = self.load_output_folder_session(folder)
            if loaded is False:
                QMessageBox.information(
                    self,
                    "Stems",
                    "No stems session was found in that folder.\n"
                    "Run Separate Stems first, then pick its output folder.",
                )

    def editor_zoom(self, factor):
        arranger = self._editor()
        if arranger is not None:
            arranger.piano_roll.zoom(factor, 1.0)

    def editor_toggle_snap(self):
        arranger = self._editor()
        if arranger is None:
            return
        enabled = not bool(getattr(arranger.piano_roll, "snap_enabled", True))
        arranger.set_snap(enabled)
        publish(
            Event.STATUS_UPDATE,
            text="Snap on" if enabled else "Snap off",
        )

    def _generator_roles(self):
        boxes = getattr(self, "gen_role_boxes", {})
        return tuple(key for key, box in boxes.items() if box.isChecked())

    def _generator_seed(self):
        spin = getattr(self, "spin_gen_seed", None)
        if spin is None:
            return None
        value = int(spin.value())
        if value <= 0:
            value = int(time.time() * 1000) % 100000
            spin.setValue(value)
        return value

    def run_midi_generator(self):
        try:
            from korshmer.midi_generation import ComposeOptions, compose
        except ImportError as e:
            QMessageBox.warning(
                self, "MIDI Generator", f"Generation engine unavailable: {e}"
            )
            return

        arranger = self._editor()
        if arranger is None:
            return
        pm = self._editor_midi()
        root, scale = self._resolve_key(pm)
        bpm = self._current_bpm()
        mode = self.combo_gen_mode.currentText()
        replace = self.chk_gen_replace.isChecked() and mode == "New song"
        options = ComposeOptions(
            root=root,
            scale=scale,
            bpm=bpm,
            minutes=self.spin_gen_minutes.value(),
            style=self.combo_gen_style.currentText(),
            mood=self.combo_gen_mood.currentText(),
            seed=self._generator_seed(),
            roles=self._generator_roles(),
            mode=mode,
            existing_tracks=0 if replace else len(arranger.model.tracks),
            grid_offset=self._editor_grid_offset(),
        )
        publish(
            Event.STATUS_UPDATE,
            text=(
                f"Layering over the editor in {root} {scale} "
                f"at {bpm:.0f} BPM..."
                if mode == "Layer over editor"
                else f"Generating {options.minutes:g} min in {root} {scale} "
                f"at {bpm:.0f} BPM..."
            ),
        )
        try:
            composition = compose(options, existing=pm)
        except ValueError as e:
            publish(Event.STATUS_UPDATE, text="MIDI Generator needs a setting")
            QMessageBox.warning(self, "MIDI Generator", str(e))
            return
        except Exception as e:
            logging.exception("MIDI Generator failed")
            publish(Event.STATUS_UPDATE, text="MIDI Generator failed")
            QMessageBox.critical(
                self, "MIDI Generator", f"Generation failed:\n{e}"
            )
            return

        instruments = [i for i in composition.tracks if i.notes]
        if not instruments:
            QMessageBox.information(
                self, "MIDI Generator", "The generator produced no notes."
            )
            return
        result = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        result.instruments = instruments
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            _layers_dir(), f"Generated_{stamp}_seed{options.seed}.mid"
        )
        try:
            result.write(path)
        except Exception as e:
            logging.exception("Could not save the generated MIDI")
            QMessageBox.critical(
                self, "MIDI Generator", f"Could not save the result:\n{e}"
            )
            return
        arranger.load_midi_file(
            path, replace=replace, tempo=bpm, merge_instruments=True
        )
        roll = getattr(arranger, "piano_roll", None)
        if (
            options.grid_offset is None
            and composition.grid_offset > 0
            and hasattr(roll, "set_grid_offset")
        ):
            roll.set_grid_offset(composition.grid_offset)
        publish(Event.OUTPUT_VIEW_REFRESH, output_dir=os.path.dirname(path))
        self._show_editor_tab()
        parts = ""
        if mode == "Layer over editor":
            parts = " | " + ", ".join(
                f"{s.name} {s.bars}" for s in composition.sections
            )
        publish(
            Event.STATUS_UPDATE,
            text=(
                f"Generated {len(instruments)} tracks, "
                f"{composition.total_bars} bars (seed {options.seed}){parts}"
            ),
        )

    def export_editor_mix_action(self):
        from korshmer.audio_processors import render_editor_mix
        from korshmer.workers.midi_preview import MidiPreviewThread

        arranger = self._editor()
        if arranger is None:
            return
        pm = self._editor_midi()
        clips = arranger.get_audio_clips()
        notes_count = (
            sum(len(i.notes) for i in pm.instruments) if pm is not None else 0
        )
        if notes_count == 0 and not clips:
            QMessageBox.information(
                self, "Export Mix", "The editor is empty, nothing to export."
            )
            return
        if notes_count > 0 and not os.path.exists(SF3_PATH):
            QMessageBox.warning(
                self,
                "Missing SoundFont",
                f"MIDI lanes need 'soundfont.sf3' in:\n{FLUID_DIR}\n"
                "Only audio clips will be exported.",
            )

        export_dir = os.path.join(DATA_DIR, "Exports")
        os.makedirs(export_dir, exist_ok=True)
        default_name = f"EditorMix_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Editor Mix",
            os.path.join(export_dir, default_name),
            "WAV Audio (*.wav)",
        )
        if not save_path:
            return
        if reject_unsafe_path(self, save_path):
            return
        if not save_path.lower().endswith(".wav"):
            save_path += ".wav"

        bpm = self._current_bpm()
        temp_mid = os.path.join(export_dir, "_editor_export.mid")
        if pm is None:
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        pm.write(temp_mid)

        publish(Event.STATUS_UPDATE, text="Exporting editor mix...")
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)

        thread = getattr(self, "export_thread", None)
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait()

        self.export_thread = MidiPreviewThread(
            temp_mid,
            FLUID_DIR,
            SF3_PATH,
            save_path,
            bpm=bpm,
            render_func=partial(render_editor_mix, clips=clips),
        )

        def on_done(path):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.OUTPUT_VIEW_REFRESH, output_dir=export_dir)
            publish(Event.STATUS_UPDATE, text="Export complete")
            QMessageBox.information(
                self, "Export Mix", f"Mix exported to:\n{path}"
            )

        def on_fail(err):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.STATUS_UPDATE, text="Export failed")
            QMessageBox.critical(self, "Export Mix", f"Export failed:\n{err}")

        self.export_thread.preview_ready.connect(on_done)
        self.export_thread.preview_failed.connect(on_fail)
        self.export_thread.start()

    def export_music_clip_action(self):
        from korshmer.video.music_clip import export_music_clip
        from korshmer.workers.base import CallableWorker

        arranger = self._editor()
        if arranger is None:
            return
        pm = self._editor_midi()
        notes_count = (
            sum(len(i.notes) for i in pm.instruments) if pm is not None else 0
        )
        clips = arranger.get_audio_clips()
        if notes_count == 0 and not clips:
            QMessageBox.information(
                self, "Export Clip", "The editor has no notes to show."
            )
            return
        if notes_count > 0 and not os.path.exists(SF3_PATH):
            QMessageBox.warning(
                self,
                "Missing SoundFont",
                f"The clip needs 'soundfont.sf3' in:\n{FLUID_DIR}",
            )
            return
        if shutil.which("ffmpeg") is None:
            QMessageBox.warning(
                self,
                "Missing ffmpeg",
                "Install ffmpeg to export a music clip.",
            )
            return

        export_dir = os.path.join(DATA_DIR, "Exports")
        os.makedirs(export_dir, exist_ok=True)
        default_name = f"MusicClip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Music Clip",
            os.path.join(export_dir, default_name),
            "MP4 Video (*.mp4)",
        )
        if not save_path:
            return
        if reject_unsafe_path(self, save_path):
            return
        if not save_path.lower().endswith(".mp4"):
            save_path += ".mp4"

        bpm = float(arranger.piano_roll.tempo)
        temp_mid = os.path.join(export_dir, "_clip_export.mid")
        if pm is None:
            pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        pm.write(temp_mid)

        publish(Event.STATUS_UPDATE, text="Exporting music clip...")
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)

        thread = getattr(self, "clip_thread", None)
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait()

        worker = CallableWorker(
            lambda: export_music_clip(
                temp_mid,
                save_path,
                FLUID_DIR,
                SF3_PATH,
                bpm=bpm,
                clips=clips,
                title=os.path.splitext(os.path.basename(save_path))[0],
                stop_callback=lambda: not worker.is_running,
                progress_callback=worker.report_progress,
            )
        )
        self.clip_thread = worker

        def on_progress(percent, text):
            publish(Event.STATUS_UPDATE, text=f"Music clip: {text}")

        def on_done(path):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.OUTPUT_VIEW_REFRESH, output_dir=export_dir)
            publish(Event.STATUS_UPDATE, text="Clip export complete")
            QMessageBox.information(
                self,
                "Export Clip",
                f"Clip exported to:\n{path}\n"
                "The mastered WAV was saved next to it.",
            )

        def on_fail(err):
            publish(Event.CONVERT_BUTTON_STATE, enabled=True)
            publish(Event.STATUS_UPDATE, text="Clip export failed")
            if err != "Cancelled":
                QMessageBox.critical(
                    self, "Export Clip", f"Export failed:\n{err}"
                )

        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_done)
        worker.error_signal.connect(on_fail)
        worker.start()
