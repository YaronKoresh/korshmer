from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def test_smart_mastering_plugin_unpacks_tuple():
    from korshmer.plugins.builtin import SmartMasteringPlugin

    plugin = SmartMasteringPlugin()
    sr = 44100
    audio = (
        np.random.default_rng(0)
        .uniform(-0.5, 0.5, (2, 4410))
        .astype(np.float32)
    )

    fake_af = MagicMock()
    fake_af.y = audio
    fake_af.sr = sr
    fake_af.to_write_format.return_value = audio.T

    processed = audio * 0.9
    mock_processor = MagicMock()
    mock_processor.process.return_value = (processed, sr)

    mock_sm_cls = MagicMock(return_value=mock_processor)

    with (
        patch("korshmer.plugins.builtin.AudioFile") as mock_af_cls,
        patch("korshmer.audio_processing.SmartMastering", mock_sm_cls),
        patch("korshmer.audio_processing.save_audio") as mock_save,
    ):
        mock_af_cls.load.return_value.ensure_2d.return_value = fake_af
        plugin.process("/in.wav", "/out.wav")

    mock_processor.process.assert_called_once_with(audio, sr)
    assert isinstance(fake_af.y, np.ndarray)
    mock_save.assert_called_once()
