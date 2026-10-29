from korshmer.core.di_container import DIContainer as Container
from korshmer.event_bus import Event, EventBus

try:
    from korshmer.workers.generic import WorkerThread

    _WORKER_AVAILABLE = True
except ImportError:
    WorkerThread = None
    _WORKER_AVAILABLE = False


class SessionController:
    def __init__(self, container: Container):
        self.container = container
        self._current_worker = None
        self._event_bus = EventBus()

    def start_conversion(self, file_paths, params, process_track_func):
        if self._current_worker is not None:
            return None

        if not _WORKER_AVAILABLE:
            return None

        self._current_worker = WorkerThread(
            file_paths, params, process_track_func
        )
        self._current_worker.started.connect(
            lambda: self._event_bus.publish(Event.PROCESS_STARTED)
        )
        self._current_worker.finished.connect(
            lambda: self._event_bus.publish(Event.PROCESS_FINISHED)
        )
        self._current_worker.start()

        return self._current_worker

    def stop_conversion(self, worker):
        if worker is not None:
            worker.requestInterruption()
            worker.wait()
            self._current_worker = None

    @property
    def is_processing(self):
        return (
            self._current_worker is not None
            and self._current_worker.isRunning()
        )
