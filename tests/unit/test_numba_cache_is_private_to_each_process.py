import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

from korshmer import numba_cache
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


def _compiled(folder, names):
    for name in names:
        path = os.path.join(folder, "librosa", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.write(name)


def test_a_process_publishes_what_it_compiled(tmp_path):
    private, snapshot = tmp_path / "private", tmp_path / "shared" / "key"
    _compiled(private, ["f.nbi", "f.1.nbc"])

    assert numba_cache.publish(str(private), str(snapshot))
    assert sorted(os.listdir(snapshot / "librosa")) == ["f.1.nbc", "f.nbi"]
    assert sorted(os.listdir(tmp_path / "shared")) == ["key"]


def test_a_smaller_cache_does_not_replace_a_fuller_snapshot(tmp_path):
    private, snapshot = tmp_path / "private", tmp_path / "shared" / "key"
    _compiled(snapshot, ["f.nbi", "f.1.nbc", "g.nbi", "g.1.nbc"])
    _compiled(private, ["f.nbi", "f.1.nbc"])

    assert not numba_cache.publish(str(private), str(snapshot))
    assert len(os.listdir(snapshot / "librosa")) == 4


def test_a_fuller_cache_replaces_the_snapshot(tmp_path):
    private, snapshot = tmp_path / "private", tmp_path / "shared" / "key"
    _compiled(snapshot, ["f.nbi", "f.1.nbc"])
    _compiled(private, ["f.nbi", "f.1.nbc", "g.nbi", "g.1.nbc"])

    assert numba_cache.publish(str(private), str(snapshot))
    assert len(os.listdir(snapshot / "librosa")) == 4
    assert sorted(os.listdir(tmp_path / "shared")) == ["key"]


def test_a_new_process_starts_from_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    base = tmp_path / numba_cache.CACHE_FOLDER
    _compiled(numba_cache.snapshot_folder(str(base)), ["f.nbi", "f.1.nbc"])

    folder = use_private_cache({})

    assert sorted(os.listdir(os.path.join(folder, "librosa"))) == [
        "f.1.nbc",
        "f.nbi",
    ]


def test_the_snapshot_changes_with_the_numba_version(monkeypatch):
    first = numba_cache.snapshot_folder("base")
    real = numba_cache.metadata.version
    monkeypatch.setattr(
        numba_cache.metadata,
        "version",
        lambda name: "0.0" if name == "numba" else real(name),
    )
    assert numba_cache.snapshot_folder("base") != first


@pytest.mark.slow
def test_a_later_process_reuses_the_numba_code_compiled_before(tmp_path):
    pytest.importorskip("librosa")
    code = textwrap.dedent(
        """
        import os, korshmer
        folder = os.environ["NUMBA_CACHE_DIR"]
        print(sum(len(files) for _r, _d, files in os.walk(folder)))
        import numpy as np, librosa
        y = np.zeros(22050 * 4, np.float32)
        y[::11025] = 1.0
        e = librosa.onset.onset_strength(y=y, sr=22050)
        librosa.beat.beat_track(onset_envelope=e, sr=22050)
        """
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (CACHE_DIR_VARIABLE, OWNER_VARIABLE)
    }
    env["TMPDIR"] = str(tmp_path)
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        for _ in range(2)
    ]
    snapshots = os.listdir(tmp_path / numba_cache.CACHE_FOLDER / "shared")
    assert len(snapshots) == 1
    assert int(runs[0].stdout) == 0
    assert int(runs[1].stdout) > 0


def test_folders_left_by_processes_that_died_are_removed(tmp_path):
    gone = tmp_path / "999999999-abc"
    mine = tmp_path / f"{os.getpid()}-abc"
    for folder in (gone, mine, tmp_path / "shared"):
        folder.mkdir()

    numba_cache.remove_abandoned(str(tmp_path))

    assert sorted(os.listdir(tmp_path)) == sorted([mine.name, "shared"])
