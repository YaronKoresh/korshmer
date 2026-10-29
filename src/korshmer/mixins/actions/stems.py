import logging
import numbers
import os

from PySide6.QtWidgets import QMessageBox

from korshmer.event_bus import Event, publish
from korshmer.utils.filesystem import is_safe_path
from korshmer.utils.tempo import BPM_DECIMALS, validate_bpm

logger = logging.getLogger(__name__)

SEPARABLE_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
EDITOR_TAB_INDEX = 0


def _publish_named_event(event_name, **data):
    event = getattr(Event, event_name, None)
    if event is None:
        logger.debug(f"Event {event_name} is not defined; skipped")
        return False
    publish(event, **data)
    return True


class StemsActionsMixin:
    def _stem_thread_is_running(self):
        thread = getattr(self, "stem_thread", None)
        return thread is not None and thread.isRunning()

    def run_stem_separation(self):
        if self._stem_thread_is_running():
            answer = QMessageBox.question(
                self,
                "Stem Separation",
                "A stem separation is already running.\nStop it?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self.stop_stem_separation()
            return
        if not hasattr(self, "_get_selected_file"):
            return
        path = self._get_selected_file(is_input=None)
        if not path:
            return
        if not path.lower().endswith(SEPARABLE_AUDIO_EXTENSIONS):
            QMessageBox.warning(
                self,
                "Error",
                "Please select an audio file (WAV, MP3, FLAC, OGG, or AIFF).",
            )
            return
        from korshmer.workers.stem_separation import StemSeparationThread

        float32 = False
        if hasattr(self, "chk_wav32"):
            float32 = bool(self.chk_wav32.isChecked())
        publish(
            Event.STATUS_UPDATE,
            text=f"Separating stems: {os.path.basename(path)}...",
        )
        publish(Event.CONVERT_BUTTON_STATE, enabled=False)
        publish(Event.PROGRESS_BAR_STATE, visible=True, value=0)
        options = None
        if hasattr(self, "combo_separation"):
            from korshmer.audio_processing.separation_engine import (
                SeparationOptions,
            )

            options = SeparationOptions(
                quality=self.combo_separation.currentText().lower()
            )
        self.stem_thread = StemSeparationThread(
            path, float32=float32, options=options
        )
        self.stem_thread.progress_update.connect(self.on_stem_progress)
        self.stem_thread.session_ready.connect(self.on_stems_session_ready)
        self.stem_thread.separation_failed.connect(
            self.on_stem_separation_failed
        )
        self.stem_thread.finished.connect(self._finish_stem_ui)
        self.stem_thread.start()

    def stop_stem_separation(self):
        if not self._stem_thread_is_running():
            return
        self.stem_thread.stop()
        publish(Event.STATUS_UPDATE, text="Stopping stem separation...")

    def on_stem_progress(self, value, text):
        publish(Event.PROGRESS_UPDATE, value=int(value), text=str(text))

    def _finish_stem_ui(self):
        publish(Event.CONVERT_BUTTON_STATE, enabled=True)
        publish(Event.PROGRESS_BAR_STATE, visible=False)

    def on_stem_separation_failed(self, message):
        self._finish_stem_ui()
        if message == "Cancelled":
            publish(Event.STATUS_UPDATE, text="Stem separation cancelled.")
            return
        publish(Event.STATUS_UPDATE, text="Stem separation failed.")
        QMessageBox.critical(
            self, "Stem Separation", f"Separation failed:\n{message}"
        )

    def on_stems_session_ready(self, session):
        self._finish_stem_ui()
        publish(Event.OUTPUT_VIEW_REFRESH, output_dir=session.output_dir)
        stem_names = ", ".join(name for name, _ in session.ordered_stems())
        publish(
            Event.STATUS_UPDATE,
            text=f"Stems ready ({stem_names}) at {round(session.tempo)} BPM",
        )
        _publish_named_event("STEMS_READY", session=session)
        answer = QMessageBox.question(
            self,
            "Stems Ready",
            f"Separated {len(session.stems)} stems into:\n"
            f"{session.output_dir}\n\n"
            "Load the vocals and stem MIDI into the editor?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.load_session_into_editor(session)

    def _editor_or_none(self):
        arranger = getattr(self, "arranger", None)
        if arranger is None:
            publish(Event.STATUS_UPDATE, text="Editor is not available.")
        return arranger

    def _show_editor_tab(self):
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(EDITOR_TAB_INDEX)

    def _apply_session_tempo(self, session):
        bpm = round(validate_bpm(getattr(session, "tempo", None)), BPM_DECIMALS)
        spin_bpm = getattr(self, "spin_bpm", None)
        if spin_bpm is not None:
            bpm = max(spin_bpm.minimum(), min(spin_bpm.maximum(), bpm))
            spin_bpm.setValue(bpm)
        arranger = getattr(self, "arranger", None)
        roll = getattr(arranger, "piano_roll", None)
        if roll is not None and hasattr(roll, "set_tempo"):
            roll.set_tempo(float(bpm))
        if hasattr(self, "_apply_tempo_update"):
            self.pending_bpm_value = float(bpm)
            self._apply_tempo_update()

    def _apply_session_grid(self, session):
        arranger = getattr(self, "arranger", None)
        roll = getattr(arranger, "piano_roll", None)
        offset = getattr(session, "grid_offset", None)
        if isinstance(offset, numbers.Real) and not isinstance(offset, bool):
            if hasattr(roll, "set_grid_offset"):
                roll.set_grid_offset(float(offset))
        elif hasattr(roll, "detect_grid_offset") and hasattr(
            arranger, "get_midi_from_editor"
        ):
            try:
                roll.detect_grid_offset(arranger.get_midi_from_editor())
            except Exception as exc:
                logger.error(f"Could not detect the grid offset: {exc}")

    def load_session_into_editor(self, session, replace=False):
        arranger = self._editor_or_none()
        if arranger is None or session is None:
            return False
        if replace and hasattr(arranger, "clear_editor"):
            arranger.clear_editor()
        self._apply_session_tempo(session)
        loaded = 0
        from korshmer.audio_processing.stem_session import (
            cleaned_vocals_path,
            vocal_gain_for_file,
        )

        vocals = cleaned_vocals_path(session)
        if vocals and hasattr(arranger, "add_audio_clip"):
            try:
                arranger.add_audio_clip(
                    vocals,
                    0.0,
                    name="Vocals",
                    kind="vocals",
                    gain=vocal_gain_for_file(vocals),
                )
                loaded += 1
            except Exception as exc:
                logger.error(f"Could not add vocals clip: {exc}")
        if hasattr(arranger, "load_midi_file"):
            for midi_path in session.midi_files:
                if not os.path.isfile(midi_path):
                    logger.warning(f"Session MIDI missing: {midi_path}")
                    continue
                try:
                    arranger.load_midi_file(
                        midi_path,
                        replace=False,
                        tempo=float(session.tempo),
                        file_tempo=False,
                    )
                    loaded += 1
                except Exception as exc:
                    logger.error(f"Could not load {midi_path}: {exc}")
        self._apply_session_grid(session)
        if loaded:
            self._show_editor_tab()
            publish(
                Event.STATUS_UPDATE,
                text=f"Editor loaded {loaded} items from "
                f"{os.path.basename(session.output_dir)}",
            )
        else:
            publish(
                Event.STATUS_UPDATE,
                text="Nothing to load from this stems session.",
            )
        return loaded > 0

    def send_audio_to_editor(self, path, start_sec=0.0):
        arranger = self._editor_or_none()
        if arranger is None or not path or not is_safe_path(path):
            return False
        if not os.path.isfile(str(path)):
            return False
        if not hasattr(arranger, "add_audio_clip"):
            publish(Event.STATUS_UPDATE, text="Editor has no clip lane.")
            return False
        from korshmer.audio_processing.stem_session import classify_stem_name

        name = os.path.splitext(os.path.basename(str(path)))[0]
        kind = classify_stem_name(path) or "audio"
        try:
            arranger.add_audio_clip(
                str(path), max(0.0, float(start_sec)), name=name, kind=kind
            )
        except Exception as exc:
            logger.error(f"Could not add clip {path}: {exc}")
            publish(Event.STATUS_UPDATE, text="Could not add audio clip.")
            return False
        self._show_editor_tab()
        publish(Event.STATUS_UPDATE, text=f"Added clip to editor: {name}")
        return True

    def load_output_folder_session(self, folder):
        from korshmer.audio_processing.stem_session import StemSession

        if not is_safe_path(folder):
            publish(
                Event.STATUS_UPDATE, text="Skipped folder with an unsafe path"
            )
            return False
        session = StemSession.find(folder)
        if session is None:
            publish(
                Event.STATUS_UPDATE,
                text="No stems session found in this folder.",
            )
            return False
        return self.load_session_into_editor(session)
