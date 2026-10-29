import gc

from PySide6.QtCore import Signal

from korshmer.constants import STEM_SEPARATION_TIMEOUT_SECONDS
from korshmer.workers.base import TaskWorker


class StemSeparationThread(TaskWorker):
    progress_update = Signal(int, str)
    session_ready = Signal(object)
    separation_failed = Signal(str)

    def __init__(
        self,
        audio_path,
        output_root=None,
        model_name="htdemucs_6s",
        float32=False,
        timeout_seconds=STEM_SEPARATION_TIMEOUT_SECONDS,
        options=None,
    ):
        super().__init__(timeout_seconds=timeout_seconds)
        self.audio_path = str(audio_path)
        self.output_root = output_root
        self.model_name = model_name
        self.float32 = bool(float32)
        self.options = options

    def _emit_progress(self, percent, message):
        if not self.is_running:
            return
        self.progress_update.emit(int(percent), str(message))
        self.report_progress(percent, message)

    def _stop_requested(self):
        return not self.is_running or self.is_timed_out()

    def execute(self):
        from korshmer.audio_processing.stem_session import separate_stems

        extra = {} if self.options is None else {"options": self.options}
        try:
            session = separate_stems(
                self.audio_path,
                output_root=self.output_root,
                model_name=self.model_name,
                progress_callback=self._emit_progress,
                check_stop_callback=self._stop_requested,
                float32=self.float32,
                **extra,
            )
        except InterruptedError:
            if self.is_running and self.is_timed_out():
                message = self._timeout_message()
                self.separation_failed.emit(message)
                raise TimeoutError(message) from None
            self.separation_failed.emit("Cancelled")
            raise
        except Exception as exc:
            self.separation_failed.emit(str(exc) or exc.__class__.__name__)
            raise
        finally:
            gc.collect()
        if self.is_running:
            self.session_ready.emit(session)
        return session
