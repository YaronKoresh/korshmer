from unittest.mock import MagicMock

import pytest

from korshmer.audio_processing import stem_separator, stem_session as ss
from korshmer.audio_processing.stem_session import (
    KNOWN_STEM_MODELS,
    separate_stems,
    validate_stem_model_name,
)


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "song.wav"
    path.write_bytes(b"RIFF")
    return path


class TestValidateStemModelName:
    @pytest.mark.parametrize("name", KNOWN_STEM_MODELS)
    def test_known_models_accepted(self, name):
        assert validate_stem_model_name(name) == name

    @pytest.mark.parametrize(
        "name", ["not_a_real_model", "", "demucs", "htdemucs/../x", "a\\b"]
    )
    def test_unknown_models_rejected(self, name):
        with pytest.raises(ValueError, match="Allowed models"):
            validate_stem_model_name(name)


class TestSeparateStemsModelCheck:
    def test_unknown_model_raises_before_demucs_call(
        self, audio_file, tmp_path, monkeypatch
    ):
        run_job = MagicMock()
        monkeypatch.setattr(ss, "_run_demucs_job", run_job)
        with pytest.raises(ValueError, match="not_a_real_model"):
            separate_stems(
                audio_file,
                output_root=tmp_path / "out",
                model_name="not_a_real_model",
            )
        run_job.assert_not_called()

    def test_known_model_is_passed_to_demucs(
        self, audio_file, tmp_path, monkeypatch
    ):
        run_job = MagicMock(side_effect=RuntimeError("stop after call"))
        monkeypatch.setattr(ss, "_run_demucs_job", run_job)
        with pytest.raises(RuntimeError, match="stop after call"):
            separate_stems(
                audio_file,
                output_root=tmp_path / "out",
                model_name="htdemucs_6s",
            )
        assert run_job.call_args[0][2] == "htdemucs_6s"


class TestStemSeparatorModelCheck:
    def _patch_demucs(self, monkeypatch):
        get_model = MagicMock()
        monkeypatch.setattr(stem_separator, "torch", MagicMock())
        monkeypatch.setattr(stem_separator, "get_model", get_model)
        monkeypatch.setattr(stem_separator, "apply_model", MagicMock())
        return get_model

    def test_init_rejects_unknown_model_before_get_model(self, monkeypatch):
        get_model = self._patch_demucs(monkeypatch)
        with pytest.raises(ValueError):
            stem_separator.StemSeparator("not_a_real_model", device="cpu")
        get_model.assert_not_called()

    def test_set_model_rejects_unknown_model_before_get_model(
        self, monkeypatch
    ):
        get_model = self._patch_demucs(monkeypatch)
        sep = stem_separator.StemSeparator("htdemucs", device="cpu")
        get_model.reset_mock()
        with pytest.raises(ValueError):
            sep.set_model("../evil")
        get_model.assert_not_called()
        assert sep.model_name == "htdemucs"


class TestStemSeparatorSampleRateIsolation:
    def test_export_uses_rate_of_separate_on_same_thread(
        self, monkeypatch, tmp_path
    ):
        import os
        import threading

        import numpy as np

        monkeypatch.setattr(stem_separator, "torch", MagicMock())
        monkeypatch.setattr(stem_separator, "get_model", MagicMock())
        sources = MagicMock()
        sources.shape = (1, 4, 2, 8)
        monkeypatch.setattr(
            stem_separator, "apply_model", MagicMock(return_value=sources)
        )
        written = {}

        def fake_write(path, data, rate):
            written[os.path.basename(os.path.dirname(path))] = rate

        monkeypatch.setattr(stem_separator.sf, "write", fake_write)

        sep = stem_separator.StemSeparator("htdemucs", device="cpu")
        both_separated = threading.Barrier(2)
        errors = []

        def run(rate, name):
            try:
                audio = np.zeros((2, 8), dtype="float32")
                sep.separate(audio, sample_rate=rate)
                both_separated.wait(timeout=10)
                sep.export_stems(
                    {"vocals": np.zeros((2, 8))},
                    str(tmp_path / name),
                    "wav",
                )
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run, args=(44100, "a")),
            threading.Thread(target=run, args=(48000, "b")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == []
        assert written == {"a": 44100, "b": 48000}
