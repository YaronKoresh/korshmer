import logging
import math

__all__ = ["validate_loop_points"]


def validate_loop_points(start_ms, end_ms, duration_ms):
    try:
        start = float(start_ms)
        end = float(end_ms)
    except (TypeError, ValueError):
        logging.warning(
            f"Rejected loop points (not numbers): "
            f"start={start_ms!r}, end={end_ms!r}"
        )
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        logging.warning(
            f"Rejected loop points (not finite): "
            f"start={start_ms!r}, end={end_ms!r}"
        )
        return None

    start = max(0.0, start)
    try:
        duration = float(duration_ms)
    except (TypeError, ValueError):
        duration = float("nan")
    if math.isfinite(duration) and duration > 0:
        end = min(end, duration)

    if start >= end:
        logging.warning(
            f"Rejected loop points (start >= end after clamping): "
            f"start={start}, end={end}"
        )
        return None
    return (start, end)
