import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from korshmer.constants import DATA_DIR
from korshmer.event_bus import Event, publish
from korshmer.utils.filesystem import is_midi_path
from korshmer.workers.base import CallableWorker


class BaseActionsMixin:
    def _get_selected_file(self, is_input=False, requires_midi=False):

        if hasattr(self, "_last_selected_list"):
            if (
                self._last_selected_list == "input"
                and self.input_list.currentItem()
            ):
                list_widget = self.input_list
            elif (
                self._last_selected_list == "output"
                and self.output_list.currentItem()
            ):
                list_widget = self.output_list
            elif self.input_list.currentItem():
                list_widget = self.input_list
            elif self.output_list.currentItem():
                list_widget = self.output_list
            else:
                list_widget = None
        else:
            if not is_input:
                list_widget = (
                    self.output_list
                    if self.output_list.currentItem()
                    else self.input_list
                )
            else:
                list_widget = (
                    self.input_list
                    if self.input_list.currentItem()
                    else self.output_list
                )

        if not list_widget:
            QMessageBox.warning(
                self,
                "Error",
                "No file selected. Please select a file from either the input or output list.",
            )
            return None

        item = list_widget.currentItem()
        if not item:
            QMessageBox.warning(
                self,
                "Error",
                "No file selected. Please select a file from either the input or output list.",
            )
            return None
        path = item.data(Qt.UserRole)
        if not path:
            QMessageBox.warning(self, "Error", "No file was selected.")
            return None
        if requires_midi and not is_midi_path(path):
            QMessageBox.warning(self, "Error", "MIDI file is required.")
            return None
        return path

    def get_selected_midi(self):
        return self._get_selected_file(is_input=None, requires_midi=True)

    def _run_action(
        self, action_func, *args, success_msg, on_done=None, **kwargs
    ):
        def finished(result):
            try:
                value = self._finish_action(result, success_msg)
            except Exception as e:
                self._fail_action(str(e))
                return
            if on_done is not None and value is not None:
                on_done(value)

        kwargs.pop("is_audio", None)
        if self._run_in_background(
            action_func,
            *args,
            on_result=finished,
            on_error=self._fail_action,
            **kwargs,
        ):
            publish(Event.STATUS_UPDATE, text="Processing...")
        return None

    def _run_in_background(self, func, *args, on_result, on_error, **kwargs):
        busy = getattr(self, "_action_worker", None)
        if busy is not None and busy.isRunning():
            publish(
                Event.STATUS_UPDATE,
                text="Another action is still running. Please wait.",
            )
            return False

        worker = CallableWorker(func, *args, **kwargs)
        self._action_worker = worker

        def finished(result):
            self._release_action_worker(worker)
            on_result(result)

        def failed(message):
            self._release_action_worker(worker)
            on_error(message)

        worker.finished_signal.connect(finished, Qt.QueuedConnection)
        worker.error_signal.connect(failed, Qt.QueuedConnection)
        worker.start()
        return True

    def _release_action_worker(self, worker):
        if getattr(self, "_action_worker", None) is worker:
            self._action_worker = None
        worker.wait()
        worker.deleteLater()

    def _fail_action(self, message):
        publish(Event.STATUS_UPDATE, text="Error.")
        QMessageBox.critical(self, "Error", f"Action failed: {message}")

    def _finish_action(self, result, success_msg):
        if result is None:
            publish(Event.STATUS_UPDATE, text="Action canceled or irrelevant.")
            return None
        if isinstance(result, str) and (
            is_midi_path(result)
            or result.lower().endswith(".wav")
            or result.lower().endswith(".json")
            or result.lower().endswith(".txt")
        ):
            output_dir = os.path.dirname(result)
            if os.path.exists(output_dir):
                publish(Event.OUTPUT_VIEW_REFRESH, output_dir=output_dir)
            else:
                publish(Event.OUTPUT_VIEW_REFRESH)
            final_name = os.path.basename(result)
            publish(Event.STATUS_UPDATE, text=f"{success_msg.split(':')[0]}.")
            QMessageBox.information(
                self, "Success", f"{success_msg}: {final_name}"
            )
            return final_name
        elif isinstance(result, list):
            slice_path = os.path.dirname(result[0]) if result else DATA_DIR
            publish(
                Event.STATUS_UPDATE,
                text=f"{success_msg.split(':')[0]}. {len(result)} files created.",
            )
            QMessageBox.information(
                self,
                "Success",
                f"{success_msg}: {len(result)} files created in folder:\n{slice_path}",
            )
            return result
        else:
            publish(Event.STATUS_UPDATE, text=f"{success_msg.split(':')[0]}.")
            QMessageBox.information(
                self, "Success", f"{success_msg} (saved directly)."
            )
            return result
