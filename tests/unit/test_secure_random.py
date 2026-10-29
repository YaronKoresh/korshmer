import re
import zipfile
from pathlib import Path

import pytest

from korshmer.collaboration.operations import Operation, OpType
from korshmer.collaboration.server import CollaborationServer
from korshmer.project_management import project_saver
from korshmer.project_management.project_loader import load_compressed
from korshmer.project_management.project_saver import save_compressed
from korshmer.utils import secure_random

SRC = Path(secure_random.__file__).resolve().parents[1]
UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def test_tokens_have_requested_size_and_differ():
    assert len(secure_random.token_bytes(8)) == 8
    assert len(secure_random.token_hex(8)) == 16
    assert secure_random.token_urlsafe(16) != secure_random.token_urlsafe(16)
    assert secure_random.token_hex() != secure_random.token_hex()


@pytest.mark.parametrize(
    "func",
    [
        secure_random.token_bytes,
        secure_random.token_hex,
        secure_random.token_urlsafe,
    ],
)
def test_tokens_reject_non_positive_size(func):
    with pytest.raises(ValueError):
        func(0)


def test_new_id_is_a_version_4_uuid_and_unique():
    ids = {secure_random.new_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(UUID4.match(i) for i in ids)


def test_make_private_dir_creates_distinct_folders(tmp_path):
    a = secure_random.make_private_dir(tmp_path / "temp", "x_")
    b = secure_random.make_private_dir(tmp_path / "temp", "x_")
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert a.name.startswith("x_")


def test_load_ignores_leftover_extract_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)
    stale = tmp_path / "temp" / "extract_1700000000" / "assets"
    stale.mkdir(parents=True)
    (stale / "stale.wav").write_bytes(b"old")
    kproj = save_compressed(
        {"name": "p", "assets": []}, tmp_path / "proj", tmp_path / "temp"
    )
    out = tmp_path / "out"
    load_compressed(kproj, tmp_path / "temp", out)
    assert not list(out.glob("*"))


def test_overlapping_saves_do_not_share_a_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)
    original = project_saver._write_archive
    calls = []

    def overlapping(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            save_compressed(
                {"name": "inner", "assets": []},
                tmp_path / "inner",
                tmp_path / "temp",
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(project_saver, "_write_archive", overlapping)
    save_compressed(
        {"name": "outer", "assets": []}, tmp_path / "outer", tmp_path / "temp"
    )
    for name in ("inner", "outer"):
        with zipfile.ZipFile(tmp_path / f"{name}.kproj") as zf:
            assert "manifest.json" in zf.namelist()
    assert list((tmp_path / "temp").iterdir()) == []


def test_session_and_operation_ids_come_from_secure_random():
    server = CollaborationServer()
    session_id = server.create_session()
    assert UUID4.match(session_id)
    op = Operation(
        op_type=OpType.TEMPO_CHANGE.value,
        session_id=session_id,
        peer_id="p",
        data={},
    )
    assert UUID4.match(op.op_id)


SECURITY_SENSITIVE = [
    "collaboration",
    "network",
    "utils/credential_store.py",
    "utils/secure_random.py",
    "project_management/project_saver.py",
    "project_management/project_loader.py",
    "project_management/project_recovery.py",
]


def _python_files():
    for entry in SECURITY_SENSITIVE:
        path = SRC / entry
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: p.name)
def test_security_sensitive_modules_do_not_use_random(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import random|from random import)", text, re.M)
    assert "uuid.uuid4" not in text
