import atexit
import hashlib
import os
import platform
import shutil
import sys
import tempfile
import uuid
from importlib import metadata

CACHE_DIR_VARIABLE = "NUMBA_CACHE_DIR"
OWNER_VARIABLE = "KORSHMER_NUMBA_CACHE_OWNER"
CACHE_FOLDER = "korshmer-numba-cache"
SHARED_FOLDER = "shared"
CACHE_FILE_ENDINGS = (".nbi", ".nbc")
VERSIONED_PACKAGES = ("numba", "llvmlite", "numpy", "librosa")


def _remove(folder):
    shutil.rmtree(folder, ignore_errors=True)


def _versions_key():
    parts = [sys.version, platform.machine()]
    for package in VERSIONED_PACKAGES:
        try:
            parts.append(f"{package}={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{package}=none")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _cache_files(folder):
    count = 0
    for _root, _dirs, files in os.walk(folder):
        count += sum(name.endswith(CACHE_FILE_ENDINGS) for name in files)
    return count


def snapshot_folder(base):
    return os.path.join(base, SHARED_FOLDER, _versions_key())


def _seed(private, snapshot):
    try:
        shutil.copytree(snapshot, private, dirs_exist_ok=True)
    except (OSError, shutil.Error):
        _remove(private)
        os.makedirs(private, exist_ok=True)


def publish(private, snapshot):
    compiled = _cache_files(private)
    if compiled == 0 or compiled <= _cache_files(snapshot):
        return False
    parent = os.path.dirname(snapshot)
    os.makedirs(parent, exist_ok=True)
    unique = uuid.uuid4().hex
    staged = os.path.join(parent, f".staged-{unique}")
    retired = os.path.join(parent, f".retired-{unique}")
    try:
        shutil.copytree(private, staged)
        if os.path.isdir(snapshot):
            os.rename(snapshot, retired)
        os.rename(staged, snapshot)
        return True
    except OSError:
        return False
    finally:
        _remove(staged)
        _remove(retired)


def _owner_gone(name):
    process = name.split("-", 1)[0]
    if not process.isdigit():
        return False
    try:
        import psutil
    except ImportError:
        return False
    return not psutil.pid_exists(int(process))


def remove_abandoned(base):
    try:
        names = os.listdir(base)
    except OSError:
        return
    for name in names:
        if name != SHARED_FOLDER and _owner_gone(name):
            _remove(os.path.join(base, name))


def _finish(private, snapshot):
    try:
        publish(private, snapshot)
    finally:
        _remove(private)


def use_private_cache(environ=None):
    environ = os.environ if environ is None else environ
    owner = environ.get(OWNER_VARIABLE)
    if environ.get(CACHE_DIR_VARIABLE) and owner is None:
        return environ[CACHE_DIR_VARIABLE]
    process = str(os.getpid())
    if owner == process:
        return environ[CACHE_DIR_VARIABLE]
    base = os.path.join(tempfile.gettempdir(), CACHE_FOLDER)
    os.makedirs(base, exist_ok=True)
    remove_abandoned(base)
    folder = tempfile.mkdtemp(prefix=f"{process}-", dir=base)
    snapshot = snapshot_folder(base)
    if os.path.isdir(snapshot):
        _seed(folder, snapshot)
    environ[CACHE_DIR_VARIABLE] = folder
    environ[OWNER_VARIABLE] = process
    atexit.register(_finish, folder, snapshot)
    return folder
