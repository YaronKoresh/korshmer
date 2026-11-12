import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from korshmer.midi.section_dynamics import (
    apply_section_dynamics,
    apply_section_dynamics_from_file,
    drop_silent_notes,
    drop_silent_notes_from_file,
    song_loud_level,
    song_loud_level_from_files,
)

SR = 22050


def _song(levels_db, seconds=10.0):
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220 * t)
    return np.concatenate([tone * 10 ** (db / 20) for db in levels_db])


def _midi(seconds, sections, velocity=100, drum=False):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, is_drum=drum)
    for start in np.arange(0.5, seconds * sections, 1.0):
        inst.notes.append(pretty_midi.Note(velocity, 60, start, start + 0.4))
    pm.instruments.append(inst)
    return pm


def _section_means(pm, seconds, sections):
    notes = pm.instruments[0].notes
    return [
        np.mean(
            [
                n.velocity
                for n in notes
                if i * seconds + 3 <= n.start < (i + 1) * seconds - 3
            ]
        )
        for i in range(sections)
    ]


def test_quiet_verse_and_loud_chorus_get_different_velocities():
    audio = _song([-20, -8, -20, -8])
    pm = _midi(10.0, 4)
    assert apply_section_dynamics(pm, audio, SR) > 0
    verse, chorus, verse2, chorus2 = _section_means(pm, 10.0, 4)
    assert chorus == pytest.approx(100, abs=2)
    assert chorus2 == pytest.approx(100, abs=2)

    assert verse == pytest.approx(50, abs=4)
    assert verse2 == pytest.approx(verse, abs=2)


def test_note_to_note_detail_is_kept():
    audio = _song([-10])
    pm = _midi(10.0, 1)
    for i, n in enumerate(pm.instruments[0].notes):
        n.velocity = 60 if i % 2 else 110
    apply_section_dynamics(pm, audio, SR)
    velocities = [n.velocity for n in pm.instruments[0].notes]
    for loud, soft in zip(velocities[::2], velocities[1::2], strict=True):
        assert loud / soft == pytest.approx(110 / 60, rel=0.05)
    assert max(velocities) == 110


def test_drop_is_capped_and_never_reaches_zero():
    audio = _song([-60, 0])
    pm = _midi(10.0, 2, velocity=20)
    apply_section_dynamics(pm, audio, SR, max_drop_db=18.0)
    quiet = min(n.velocity for n in pm.instruments[0].notes)
    assert quiet == round(20 * 10 ** (-18 / 40))
    assert quiet >= 1


def test_drums_and_zero_amount_are_left_alone():
    audio = _song([-20, -8])
    drums = _midi(10.0, 2, drum=True)
    assert apply_section_dynamics(drums, audio, SR) == 0
    plain = _midi(10.0, 2)
    assert apply_section_dynamics(plain, audio, SR, amount=0) == 0
    assert {n.velocity for n in plain.instruments[0].notes} == {100}


def test_silence_and_empty_midi_do_not_fail():
    assert apply_section_dynamics(_midi(1.0, 1), np.zeros(10), SR) == 0
    assert apply_section_dynamics(pretty_midi.PrettyMIDI(), _song([0]), SR) == 0


def test_from_file_matches_in_memory(tmp_path):
    audio = _song([-20, -8]).astype(np.float32)
    path = tmp_path / "stem.wav"
    sf.write(path, np.stack([audio, audio], 1), SR)
    a, b = _midi(10.0, 2), _midi(10.0, 2)
    apply_section_dynamics(a, audio, SR)
    apply_section_dynamics_from_file(b, str(path))
    va = [n.velocity for n in a.instruments[0].notes]
    vb = [n.velocity for n in b.instruments[0].notes]
    assert np.allclose(va, vb, atol=1)


def test_backing_stem_follows_the_song_and_not_its_own_peak():
    lead, backing = _song([-6, -6]), _song([-18, -6])
    song = song_loud_level([lead, backing], SR)
    lead_midi, backing_midi = _midi(10.0, 2), _midi(10.0, 2)
    apply_section_dynamics(lead_midi, lead, SR, reference_db=song)
    apply_section_dynamics(backing_midi, backing, SR, reference_db=song)
    assert _section_means(lead_midi, 10.0, 2) == pytest.approx(
        [100, 100], abs=2
    )
    assert _section_means(backing_midi, 10.0, 2) == pytest.approx(
        [50, 100], abs=4
    )


def test_stem_that_is_soft_all_song_long_stays_soft():
    lead, backing = _song([-8]), _song([-20])
    song = song_loud_level([backing, lead], SR)
    alone, in_song = _midi(10.0, 1), _midi(10.0, 1)
    apply_section_dynamics(alone, backing, SR)
    apply_section_dynamics(in_song, backing, SR, reference_db=song)
    assert _section_means(alone, 10.0, 1)[0] == pytest.approx(100, abs=2)
    assert _section_means(in_song, 10.0, 1)[0] == pytest.approx(50, abs=4)


def test_song_level_skips_silence_and_matches_files(tmp_path):
    loud, soft = _song([-6]), _song([-18])
    song = song_loud_level([soft, np.zeros(10), loud], SR)
    assert song == pytest.approx(song_loud_level([loud], SR))
    assert song_loud_level([], SR) is None
    assert song_loud_level([np.zeros(10)], SR) is None
    paths = [tmp_path / "bass.wav", tmp_path / "other.wav"]
    for path, audio in zip(paths, (loud, soft), strict=True):
        sf.write(path, audio.astype(np.float32), SR)
    assert song_loud_level_from_files(paths) == pytest.approx(song, abs=0.1)


def test_reference_level_must_be_a_finite_number():
    with pytest.raises(ValueError):
        apply_section_dynamics(
            _midi(10.0, 1), _song([-6]), SR, reference_db=float("nan")
        )


def test_notes_over_a_silent_part_of_the_stem_are_dropped():
    lead, stem = _song([-8, -8, -8]), _song([-20, -70, -30])
    song = song_loud_level([lead, stem], SR)
    pm = _midi(10.0, 3)
    dropped = drop_silent_notes(pm, stem, SR, song)
    starts = [n.start for n in pm.instruments[0].notes]
    assert dropped == 10
    assert len(starts) == 20
    assert not [s for s in starts if 10.0 <= s < 20.0]


def test_silent_note_check_keeps_drums_and_matches_files(tmp_path):
    stem = _song([-70, -8]).astype(np.float32)
    song = song_loud_level([stem], SR)
    drums = _midi(10.0, 2, drum=True)
    assert drop_silent_notes(drums, stem, SR, song) == 0
    assert len(drums.instruments[0].notes) == 20
    path = tmp_path / "piano.wav"
    sf.write(path, stem, SR)
    a, b = _midi(10.0, 2), _midi(10.0, 2)
    assert drop_silent_notes(a, stem, SR, song) == 10
    assert drop_silent_notes_from_file(b, str(path), song) == 10
    assert drop_silent_notes(_midi(1.0, 1), np.zeros(10), SR, song) == 0
    with pytest.raises(ValueError):
        drop_silent_notes(_midi(1.0, 1), stem, SR, float("inf"))


def test_voices_stay_nearly_dry_and_bass_tight():
    from korshmer.audio_processors import effect_sends

    vocal_reverb, vocal_chorus = effect_sends(53)
    default_reverb, _ = effect_sends(0)
    assert vocal_reverb < default_reverb
    assert vocal_chorus == 0
    assert effect_sends(33)[0] < default_reverb
    assert effect_sends(0, is_drum=True)[1] == 0


def test_vocals_use_a_fast_attack_voice_and_dynamics_are_on_by_default():
    from korshmer import track_processor
    from korshmer.services.preset_service import PresetService

    assert track_processor.VOCAL_PROGRAM == 53
    assert PresetService.normalize_velocity_mode("Original") == "Original"
    assert PresetService.normalize_velocity_mode(None) == "Original"
