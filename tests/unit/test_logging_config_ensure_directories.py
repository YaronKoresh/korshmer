from korshmer.utils import logging_config


def _patch_dirs(monkeypatch, root):
    data = root / "saved_data"
    monkeypatch.setattr(logging_config, "DATA_DIR", data)
    monkeypatch.setattr(logging_config, "TEMP_DIR", root / "temp")
    monkeypatch.setattr(logging_config, "MODELS_DIR", root / "models")
    monkeypatch.setattr(logging_config, "LOGS_DIR", data / "logs")


def test_creates_all_directories(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    logging_config.ensure_directories()
    for name in ("temp", "models", "saved_data", "saved_data/logs"):
        assert (tmp_path / name).is_dir()


def test_blocked_data_dir_does_not_skip_temp_and_models(tmp_path, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    (tmp_path / "saved_data").write_text("blocker")
    logging_config.ensure_directories()
    assert (tmp_path / "temp").is_dir()
    assert (tmp_path / "models").is_dir()
