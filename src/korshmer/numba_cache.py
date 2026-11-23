import atexit
import os
import shutil
import tempfile

CACHE_DIR_VARIABLE = "NUMBA_CACHE_DIR"
OWNER_VARIABLE = "KORSHMER_NUMBA_CACHE_OWNER"
CACHE_FOLDER = "korshmer-numba-cache"


def _remove(folder):
    shutil.rmtree(folder, ignore_errors=True)


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
    folder = tempfile.mkdtemp(prefix=f"{process}-", dir=base)
    environ[CACHE_DIR_VARIABLE] = folder
    environ[OWNER_VARIABLE] = process
    atexit.register(_remove, folder)
    return folder
