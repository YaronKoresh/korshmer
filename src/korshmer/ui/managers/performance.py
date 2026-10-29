import logging
import time
from collections import deque

from korshmer.constants import (
    PERFORMANCE_CPU_SPIKE_PERCENT,
    PERFORMANCE_LOGGER_NAME,
)

perf_logger = logging.getLogger(PERFORMANCE_LOGGER_NAME)


class PerformanceMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.last_update = time.time()
        self.update_interval = 0.5
        self.track_performance = {}
        self._in_cpu_spike = False

    def _log_cpu_spike(self, cpu_percent):
        is_spike = cpu_percent > PERFORMANCE_CPU_SPIKE_PERCENT
        if is_spike and not self._in_cpu_spike:
            perf_logger.warning(
                f"CPU spike: {cpu_percent:.1f}% "
                f"(threshold {PERFORMANCE_CPU_SPIKE_PERCENT:.1f}%)"
            )
        self._in_cpu_spike = is_spike

    def update(self):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return None

        self.last_update = current_time

        try:
            import psutil

            process = psutil.Process()

            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.cpu_history.append(cpu_percent)
            self._log_cpu_spike(cpu_percent)
            self.memory_history.append(memory_mb)

            return {
                "cpu_current": cpu_percent,
                "cpu_avg": sum(self.cpu_history) / len(self.cpu_history)
                if self.cpu_history
                else 0,
                "memory_current": memory_mb,
                "memory_avg": (
                    sum(self.memory_history) / len(self.memory_history)
                    if self.memory_history
                    else 0
                ),
                "threads": process.num_threads(),
            }
        except Exception as e:
            logging.debug(f"Performance monitoring error: {e}")
            return None

    def record_track_performance(self, track_id, cpu_usage, processing_time):
        max_tracked_items = 100
        if len(self.track_performance) >= max_tracked_items:
            oldest_track = min(
                self.track_performance.keys(),
                key=lambda k: len(self.track_performance[k]["cpu_samples"]),
            )
            del self.track_performance[oldest_track]

        if track_id not in self.track_performance:
            self.track_performance[track_id] = {
                "cpu_samples": deque(maxlen=50),
                "time_samples": deque(maxlen=50),
            }

        self.track_performance[track_id]["cpu_samples"].append(cpu_usage)
        self.track_performance[track_id]["time_samples"].append(processing_time)

    def clear_track_performance(self, track_id):
        if track_id in self.track_performance:
            del self.track_performance[track_id]

    def get_track_stats(self, track_id):
        if track_id in self.track_performance:
            data = self.track_performance[track_id]
            return {
                "avg_cpu": (
                    sum(data["cpu_samples"]) / len(data["cpu_samples"])
                    if data["cpu_samples"]
                    else 0
                ),
                "avg_time": (
                    sum(data["time_samples"]) / len(data["time_samples"])
                    if data["time_samples"]
                    else 0
                ),
            }
        return {"avg_cpu": 0, "avg_time": 0}

    def get_summary(self):
        return {
            "cpu_current": self.cpu_history[-1] if self.cpu_history else 0,
            "cpu_peak": max(self.cpu_history) if self.cpu_history else 0,
            "memory_current": self.memory_history[-1]
            if self.memory_history
            else 0,
            "memory_peak": max(self.memory_history)
            if self.memory_history
            else 0,
        }
