import os
import subprocess
import sys
import textwrap

import pytest

from korshmer.numba_cache import (
    CACHE_DIR_VARIABLE,
    OWNER_VARIABLE,
    use_private_cache,
)


def test_each_process_gets_its_own_cache_folder():
    environ = {}
    folder = use_private_cache(environ)
    assert os.path.isdir(folder)
    assert environ[CACHE_DIR_VARIABLE] == folder
    assert environ[OWNER_VARIABLE] == str(os.getpid())


def test_calling_again_in_the_same_process_keeps_the_folder():
    environ = {}
    assert use_private_cache(environ) == use_private_cache(environ)


def test_a_folder_inherited_from_another_process_is_replaced():
    environ = {CACHE_DIR_VARIABLE: "/parent/cache", OWNER_VARIABLE: "1"}
    folder = use_private_cache(environ)
    assert folder != "/parent/cache"
    assert environ[OWNER_VARIABLE] == str(os.getpid())


def test_a_folder_the_user_chose_is_kept():
    environ = {CACHE_DIR_VARIABLE: "/chosen/cache"}
    assert use_private_cache(environ) == "/chosen/cache"
    assert OWNER_VARIABLE not in environ


def test_importing_the_package_sets_a_private_cache_folder():
    code = "import os, korshmer; print(os.environ['NUMBA_CACHE_DIR'])"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (CACHE_DIR_VARIABLE, OWNER_VARIABLE)
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    folder = out.stdout.strip()
    assert "korshmer-numba-cache" in folder
    assert not os.path.exists(folder)


@pytest.mark.slow
def test_tempo_survives_processes_compiling_at_the_same_time():
    pytest.importorskip("librosa")
    code = textwrap.dedent(
        """
        import korshmer
        import numpy as np
        from korshmer.audio_processing.tempo_estimation import estimate_tempo
        sr = 22050
        y = np.zeros(sr * 8, np.float32)
        for i in range(16):
            y[int(i * 0.5 * sr) : int(i * 0.5 * sr) + 200] = 1.0
        print(estimate_tempo(y, sr))
        """
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (CACHE_DIR_VARIABLE, OWNER_VARIABLE)
    }
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    for run in runs:
        out, err = run.communicate(timeout=600)
        assert run.returncode == 0, err
        assert abs(float(out) - 120.0) < 2.0
