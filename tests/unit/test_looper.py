import numpy as np

from korshmer.core.looper import Looper


def record_layer(lp, value, block=4410):
    lp.arm_record()
    out = np.zeros((block, lp.channels), dtype=np.float32)
    n = lp.loop_frames
    fed = 0
    while lp.recording or lp.armed:
        if lp.armed:
            lp.mix_out(out)
            continue
        take = min(block, n - fed) if n - fed > 0 else block
        lp.feed_input(np.full((take, lp.channels), value, dtype=np.float32))
        fed += take
    return lp


def play(lp, frames):
    out = np.zeros((frames, lp.channels), dtype=np.float32)
    lp.mix_out(out)
    return out


def test_loop_length_follows_bpm_bars_and_rate():
    lp = Looper(bpm=120, bars=2, beats_per_bar=4, sample_rate=44100)
    assert lp.loop_frames == 176400
    assert lp.set_bpm(60)
    assert lp.loop_frames == 352800


def test_playback_is_the_sum_of_layers_within_one_sample():
    lp = Looper(bpm=120, bars=1, sample_rate=1000)
    assert lp.loop_frames == 2000
    record_layer(lp, 0.25)
    assert lp.playing and lp.layer_count == 1
    record_layer(lp, 0.125)
    assert lp.layer_count == 2
    out = play(lp, 5000)
    assert np.allclose(out, 0.375, atol=1e-6)
    lp.close()


def test_twenty_overdubs_in_a_row_keep_playing_and_undo_clear_work():
    lp = Looper(bpm=240, bars=1, sample_rate=1000)
    for _ in range(20):
        record_layer(lp, 0.01)
    assert lp.layer_count == 20
    assert np.allclose(play(lp, 300), 0.2, atol=1e-5)
    assert lp.undo_layer() and lp.layer_count == 19
    assert np.allclose(play(lp, 300), 0.19, atol=1e-5)
    lp.clear()
    assert lp.layer_count == 0 and not lp.playing
    assert not play(lp, 10).any()
    assert not lp.undo_layer()


def test_output_is_clipped_but_layers_are_not_normalised_away():
    lp = Looper(bpm=240, bars=1, sample_rate=1000)
    record_layer(lp, 0.8)
    record_layer(lp, 0.8)
    assert play(lp, 10).max() == 1.0
    lp.undo_layer()
    assert np.allclose(play(lp, 10), 0.8)


def test_bpm_change_with_layers_waits_until_clear():
    lp = Looper(bpm=240, bars=1, sample_rate=1000)
    record_layer(lp, 0.1)
    frames = lp.loop_frames
    assert not lp.set_bpm(120)
    assert lp.loop_frames == frames and lp.pending_bpm == 120.0
    lp.clear()
    assert lp.bpm == 120.0 and lp.loop_frames == 2 * frames


def test_input_latency_is_compensated_and_mono_input_fills_stereo():
    lp = Looper(
        bpm=240, bars=1, sample_rate=1000, channels=2, latency_frames=10
    )
    lp.arm_record()
    ramp = np.arange(lp.loop_frames, dtype=np.float32).reshape(-1, 1) / 1000
    lp.feed_input(ramp)
    out = play(lp, lp.loop_frames)
    assert out.shape[1] == 2
    assert np.isclose(out[0, 0], ramp[10, 0]) and np.isclose(
        out[0, 1], ramp[10, 0]
    )
    assert out[-1].max() == 0.0
