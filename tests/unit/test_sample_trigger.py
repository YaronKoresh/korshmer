import numpy as np
import pytest

from korshmer.core.audio_engine import AudioEngine
from korshmer.core.sample_trigger import MAX_VOICES, SampleBank, SampleMode
from korshmer.midi.controller_mapping import ControllerAction


def render(bank, frames, channels=1):
    out = np.zeros((frames, channels), dtype=np.float32)
    bank.mix_into(out)
    return out[:, 0]


def bank_with(**samples):
    bank = SampleBank(1000, 1)
    for note, (data, mode) in samples.items():
        bank.assign(int(note[1:]), np.asarray(data, dtype=np.float32), mode)
    return bank


def test_two_one_shots_sum_until_each_ends():
    bank = bank_with(
        n60=([1.0] * 6, SampleMode.ONE_SHOT),
        n62=([0.5] * 3, SampleMode.ONE_SHOT),
    )
    bank.trigger(60)
    first = render(bank, 2)
    bank.trigger(62)
    rest = render(bank, 6)
    assert list(first) == [1.0, 1.0]
    assert list(rest) == [1.5, 1.5, 1.5, 1.0, 0.0, 0.0]
    assert bank.active_voices == 0


def test_one_shot_ignores_note_off_and_loop_stops_on_it():
    bank = bank_with(
        n60=([1.0] * 4, SampleMode.ONE_SHOT),
        n61=([2.0, 3.0, 4.0], SampleMode.LOOP),
    )
    bank.trigger(60)
    bank.trigger(61)
    bank.release(60)
    out = render(bank, 8)

    assert list(out) == [3.0, 4.0, 5.0, 3.0, 3.0, 4.0, 2.0, 3.0]
    bank.release(61)
    assert not render(bank, 4).any()
    assert bank.active_voices == 0


def test_actions_from_controller_trigger_and_release():
    bank = bank_with(n64=([1.0, 1.0], SampleMode.LOOP))
    assert bank.handle_action(ControllerAction("note_on", 0, 64))
    assert list(render(bank, 5)) == [1.0] * 5
    assert bank.handle_action(ControllerAction("note_off", 0, 64))
    assert not render(bank, 2).any()
    assert not bank.handle_action(ControllerAction("transport_play", 0, 1))


def test_voice_limit_unassigned_note_and_validation():
    bank = bank_with(n10=([1.0] * 100, SampleMode.ONE_SHOT))
    for _ in range(MAX_VOICES + 5):
        bank.trigger(10)
    bank.trigger(99)
    render(bank, 1)
    assert bank.active_voices == MAX_VOICES
    with pytest.raises(ValueError):
        bank.assign(128, [1.0])
    with pytest.raises(ValueError):
        bank.assign(5, [])


def test_mono_sample_is_spread_over_stereo_and_rate_is_converted():
    bank = SampleBank(2000, 2)
    bank.assign(
        60,
        np.ones(100, dtype=np.float32),
        SampleMode.ONE_SHOT,
        sample_rate=1000,
    )
    bank.trigger(60)
    out = np.zeros((300, 2), dtype=np.float32)
    bank.mix_into(out)
    assert out[:200].min() == 1.0 and not out[200:].any()
    bank.configure(1000, 1)
    bank.trigger(60)
    out1 = np.zeros((300, 1), dtype=np.float32)
    bank.mix_into(out1)
    assert out1[:100].min() == 1.0 and not out1[100:].any()


def test_engine_callback_mixes_samples_over_playback_with_volume():
    eng = AudioEngine()
    eng.data = np.full((1000, 1), 0.25, dtype="float32")
    eng.fs = 1000
    eng.samples.configure(1000, 1)
    eng.samples.assign(
        60, np.full(10, 0.5, dtype=np.float32), SampleMode.ONE_SHOT
    )
    eng.samples.trigger(60)
    eng._volume = 0.5
    eng._stream_start_time = None
    out = np.zeros((20, 1), dtype=np.float32)
    eng._cb(out, 20, None, None)
    assert np.allclose(out[:10], (0.25 + 0.5) * 0.5)
    assert np.allclose(out[10:], 0.125)
