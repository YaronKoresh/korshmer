from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from korshmer.project_management.file_io_log import logged_file_op, logger

CORRUPT_SUFFIX = ".corrupt"


def validate_recovery_data(recovery_data: Any) -> str | None:
    if not isinstance(recovery_data, dict):
        return "recovery data is not an object"
    timestamp = recovery_data.get("timestamp")
    if not isinstance(timestamp, str):
        return "timestamp is missing or not text"
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        return "timestamp is not an ISO date"
    if not isinstance(recovery_data.get("project_data"), dict):
        return "project_data is missing or not an object"
    project_path = recovery_data.get("project_path")
    if project_path is not None and not isinstance(project_path, str):
        return "project_path is not text"
    return None


def quarantine_recovery_file(recovery_file: Path) -> Path | None:
    target = recovery_file.with_name(recovery_file.name + CORRUPT_SUFFIX)
    try:
        os.replace(recovery_file, target)
        logger.info(
            "Move corrupt recovery file done: %s -> %s", recovery_file, target
        )
        return target
    except OSError as e:
        logging.error(f"Failed to set aside corrupt recovery file: {e}")
        return None


def load_recovery_file(recovery_file: Path) -> dict[str, Any] | None:
    if not recovery_file.exists():
        return None

    try:
        with logged_file_op("Load recovery file", recovery_file):
            with gzip.open(recovery_file, "rt", encoding="utf-8") as f:
                recovery_data = json.load(f)
        problem = validate_recovery_data(recovery_data)
    except Exception as e:
        problem = str(e) or type(e).__name__

    if problem is not None:
        kept = quarantine_recovery_file(recovery_file)
        logging.error(
            f"Recovery file is corrupt ({problem}); "
            + (f"kept as {kept}" if kept else "it could not be set aside")
        )
        return None

    logging.info(f"Found recovery file from {recovery_data.get('timestamp')}")
    return recovery_data


def save_recovery_file(
    recovery_file: Path,
    current_project_path: Path | None,
    current_project_data: dict[str, Any],
) -> bool:
    temporary_path = None
    try:
        recovery_data = {
            "timestamp": datetime.now().isoformat(),
            "project_path": (
                str(current_project_path) if current_project_path else None
            ),
            "project_data": current_project_data,
        }

        handle, temporary_name = tempfile.mkstemp(
            dir=recovery_file.parent,
            prefix=recovery_file.name + ".",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with logged_file_op("Save recovery file", recovery_file):
            with os.fdopen(handle, "wb") as raw:
                with gzip.open(raw, "wt", encoding="utf-8") as f:
                    json.dump(recovery_data, f)
            os.replace(temporary_path, recovery_file)
            temporary_path = None
        return True
    except Exception as e:
        logging.error(f"Failed to save recovery state: {e}")
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def clear_recovery_file(recovery_file: Path) -> bool:
    try:
        if recovery_file.exists():
            with logged_file_op(
                "Clear recovery file", recovery_file, report_size=False
            ):
                recovery_file.unlink()
        return True
    except Exception as e:
        logging.error(f"Failed to clear recovery state: {e}")
        return False
