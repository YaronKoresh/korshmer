import logging
import math

__all__ = ["validate_timeline_time"]


def validate_timeline_time(time):
    try:
        value = float(time)
    except (TypeError, ValueError):
        logging.warning(f"Rejected timeline time (not a number): {time!r}")
        return None
    if not math.isfinite(value):
        logging.warning(f"Rejected timeline time (not finite): {time!r}")
        return None
    if value < 0.0:
        logging.warning(f"Clamped negative timeline time to 0.0: {time!r}")
        return 0.0
    return value
