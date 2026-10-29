import datetime
import faulthandler
import itertools
import logging
import os
import platform
import sys
import threading
import traceback
from pathlib import Path

from korshmer import __version__
from korshmer.constants import CRASH_REPORTS_DIR

logger = logging.getLogger(__name__)

MAX_CRASH_REPORTS = 20
CRASH_REPORT_PREFIX = "crash-"
CRASH_REPORT_SUFFIX = ".txt"
FATAL_ERRORS_FILE = "fatal-errors.log"

_lock = threading.Lock()
_counter = itertools.count(1)
_state = {
    "installed": False,
    "reports_dir": None,
    "sys_hook": None,
    "thread_hook": None,
    "fault_file": None,
}


def _format_report(exc_type, exc_value, exc_tb, thread_name, timestamp):
    lines = [
        "KORSHMER crash report",
        "=" * 60,
        f"Time: {timestamp.isoformat(timespec='seconds')}",
        f"Version: {__version__}",
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"Executable: {sys.executable}",
        f"Process ID: {os.getpid()}",
        f"Thread: {thread_name}",
        f"Exception: {getattr(exc_type, '__name__', exc_type)}: {exc_value}",
        "=" * 60,
        "",
        "Traceback:",
        "".join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        ).rstrip(),
        "",
    ]
    return "\n".join(lines)


def _prune_reports(reports_dir, keep):
    try:
        reports = sorted(
            p
            for p in Path(reports_dir).glob(
                f"{CRASH_REPORT_PREFIX}*{CRASH_REPORT_SUFFIX}"
            )
            if p.is_file()
        )
    except OSError:
        return
    for old in reports[: max(0, len(reports) - keep)]:
        try:
            old.unlink()
        except OSError:
            pass


def write_crash_report(
    exc_type, exc_value, exc_tb, thread_name=None, reports_dir=None
):
    try:
        if thread_name is None:
            thread_name = threading.current_thread().name
        target_dir = Path(
            reports_dir or _state["reports_dir"] or CRASH_REPORTS_DIR
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now()
        with _lock:
            seq = next(_counter)
        name = (
            f"{CRASH_REPORT_PREFIX}{now.strftime('%Y%m%d-%H%M%S')}"
            f"-{os.getpid()}-{seq:03d}{CRASH_REPORT_SUFFIX}"
        )
        path = target_dir / name
        tmp_path = target_dir / (name + ".tmp")
        text = _format_report(exc_type, exc_value, exc_tb, thread_name, now)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
        _prune_reports(target_dir, MAX_CRASH_REPORTS)
    except Exception as e:
        try:
            logger.error(f"Could not write crash report: {e}")
        except Exception:
            pass
        return None
    try:
        logger.critical(
            f"Unhandled {getattr(exc_type, '__name__', exc_type)} in thread "
            f"{thread_name}; crash report saved to {path}",
            exc_info=(exc_type, exc_value, exc_tb),
        )
    except Exception:
        pass
    return path


def _sys_excepthook(exc_type, exc_value, exc_tb):
    if not issubclass(exc_type, KeyboardInterrupt):
        write_crash_report(exc_type, exc_value, exc_tb)
    previous = _state["sys_hook"] or sys.__excepthook__
    previous(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    if args.exc_type is not None and not issubclass(args.exc_type, SystemExit):
        thread_name = args.thread.name if args.thread is not None else None
        write_crash_report(
            args.exc_type, args.exc_value, args.exc_traceback, thread_name
        )
    previous = _state["thread_hook"] or threading.__excepthook__
    previous(args)


def _enable_faulthandler(reports_dir):
    try:
        fault_file = open(
            Path(reports_dir) / FATAL_ERRORS_FILE, "a", encoding="utf-8"
        )
    except OSError as e:
        logger.warning(f"Could not open fatal error log: {e}")
        return
    try:
        fault_file.write(
            f"\n--- KORSHMER {__version__} started "
            f"{datetime.datetime.now().isoformat(timespec='seconds')} "
            f"(pid {os.getpid()}) ---\n"
        )
        fault_file.flush()
        faulthandler.enable(file=fault_file, all_threads=True)
    except Exception as e:
        logger.warning(f"Could not enable fatal error handler: {e}")
        fault_file.close()
        return
    _state["fault_file"] = fault_file


def install_crash_handler(reports_dir=None, enable_faulthandler=True):
    with _lock:
        if _state["installed"]:
            return Path(_state["reports_dir"])
        target_dir = Path(reports_dir or CRASH_REPORTS_DIR)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create crash report directory: {e}")
        _state["reports_dir"] = target_dir
        _state["sys_hook"] = sys.excepthook
        _state["thread_hook"] = threading.excepthook
        sys.excepthook = _sys_excepthook
        threading.excepthook = _threading_excepthook
        _state["installed"] = True
    if enable_faulthandler:
        _enable_faulthandler(target_dir)
    logger.info(f"Crash reports are saved to {target_dir}")
    return target_dir


def uninstall_crash_handler():
    with _lock:
        if not _state["installed"]:
            return
        if sys.excepthook is _sys_excepthook:
            sys.excepthook = _state["sys_hook"] or sys.__excepthook__
        if threading.excepthook is _threading_excepthook:
            threading.excepthook = (
                _state["thread_hook"] or threading.__excepthook__
            )
        fault_file = _state["fault_file"]
        if fault_file is not None:
            try:
                faulthandler.disable()
            except Exception:
                pass
            try:
                fault_file.close()
            except Exception:
                pass
        _state.update(
            installed=False,
            reports_dir=None,
            sys_hook=None,
            thread_hook=None,
            fault_file=None,
        )


def is_installed():
    return _state["installed"]
