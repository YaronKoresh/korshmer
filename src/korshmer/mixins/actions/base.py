import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from korshmer.constants import DATA_DIR
from korshmer.event_bus import Event, publish
from korshmer.utils.filesystem import is_midi_path
from korshmer.workers.base import CallableWorker

ACTION_CANCELLED_MESSAGE = "Cancelled"


class BaseActionsMixin:
    _action_progress_label: str | None = None
    _action_owns_progress_bar = False

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
        self,
        action_func,
        *args,
        success_msg,
        on_done=None,
        progress_label=None,
        **kwargs,
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
            progress_label=progress_label,
            **kwargs,
        ):
            publish(
                Event.STATUS_UPDATE,
                text=f"{progress_label}: starting..."
                if progress_label
                else "Processing...",
            )
        return None

    def _run_in_background(
        self,
        func,
        *args,
        on_result,
        on_error,
        progress_label=None,
        **kwargs,
    ):
        busy = getattr(self, "_action_worker", None)
        if busy is not None and busy.isRunning():
            publish(
                Event.STATUS_UPDATE,
                text="Another action is still running. Please wait.",
            )
            return False

        worker = CallableWorker(
            func, *args, feeds_progress=progress_label is not None, **kwargs
        )
        self._action_worker = worker
        settled = []

        def settle():
            settled.append(True)
            self._release_action_worker(worker)
            if progress_label is not None:
                self._hide_action_progress()

        def finished(result):
            settle()
            on_result(result)

        def failed(message):
            settle()
            if progress_label is not None and worker.stop_requested():
                publish(
                    Event.STATUS_UPDATE, text=f"{progress_label} cancelled."
                )
                return
            on_error(message)

        def ended():
            if not settled:
                failed(ACTION_CANCELLED_MESSAGE)

        worker.finished_signal.connect(finished, Qt.QueuedConnection)
        worker.error_signal.connect(failed, Qt.QueuedConnection)
        if progress_label is not None:
            self._action_progress_label = progress_label
            worker.progress_signal.connect(
                self._show_action_progress, Qt.QueuedConnection
            )
            worker.finished.connect(ended, Qt.QueuedConnection)
            self._begin_action_progress()
        worker.start()
        return True

    def cancellable_action_is_running(self):
        worker = getattr(self, "_action_worker", None)
        return (
            worker is not None
            and worker.isRunning()
            and getattr(self, "_action_progress_label", None) is not None
        )

    def cancel_running_action(self):
        if not self.cancellable_action_is_running():
            return False
        self._action_worker.stop()
        publish(
            Event.STATUS_UPDATE,
            text=f"Stopping {self._action_progress_label.lower()}...",
        )
        if self._action_owns_progress_bar:
            publish(Event.CONVERT_BUTTON_STATE, enabled=False)
            publish(Event.CONVERT_BUTTON_TEXT, text="Stopping...")
        return True

    def _other_progress_is_shown(self):
        stems_running = getattr(self, "_stem_thread_is_running", None)
        return bool(getattr(self, "is_processing", False)) or bool(
            stems_running is not None and stems_running()
        )

    def _begin_action_progress(self):
        self._action_owns_progress_bar = not self._other_progress_is_shown()
        if not self._action_owns_progress_bar:
            return
        publish(Event.PROGRESS_BAR_STATE, visible=True, value=0)
        publish(
            Event.CONVERT_BUTTON_TEXT,
            text=f"Stop {self._action_progress_label}",
        )
        publish(Event.CONVERT_BUTTON_ROLE, role="danger")
        publish(Event.CONVERT_BUTTON_STATE, enabled=True)

    def _show_action_progress(self, percent, stage):
        label = getattr(self, "_action_progress_label", None)
        worker = getattr(self, "_action_worker", None)
        if label is None or worker is None or worker.stop_requested():
            return
        text = f"{label} {percent}%: {stage}"
        if self._action_owns_progress_bar:
            publish(Event.PROGRESS_UPDATE, value=percent, text=text)
        else:
            publish(Event.STATUS_UPDATE, text=text)

    def _hide_action_progress(self):
        self._action_progress_label = None
        if not self._action_owns_progress_bar:
            return
        self._action_owns_progress_bar = False
        publish(Event.PROGRESS_BAR_STATE, visible=False)
        publish(Event.CONVERT_BUTTON_TEXT, text="Start Processing")
        publish(Event.CONVERT_BUTTON_ROLE, role="primary")
        publish(
            Event.CONVERT_BUTTON_STATE,
            enabled=bool(getattr(self, "file_list_paths", True)),
        )

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
