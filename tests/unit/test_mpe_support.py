import pytest

from korshmer.midi_processing.mpe_support import (
    MPEChannelAllocator,
    MPEChannelExhaustedError,
    MPEConfiguration,
    MPEControllerMapper,
    MPENote,
    MPEProcessor,
    MPEZone,
)


def test_zone_channels_exclude_master_channels():
    processor = MPEProcessor()
    config = processor.config
    assert config.master_channel_lower not in config.lower_zone_channels
    assert config.master_channel_upper not in config.upper_zone_channels


def test_custom_configuration_drops_master_channels():
    config = MPEConfiguration(
        lower_zone_channels=[0, 1, 2],
        upper_zone_channels=[13, 14, 15],
    )
    assert config.lower_zone_channels == [1, 2]
    assert config.upper_zone_channels == [13, 14]


@pytest.mark.parametrize("zone", [MPEZone.LOWER, MPEZone.UPPER])
def test_channels_are_not_reused_until_exhausted(zone):
    processor = MPEProcessor()
    members = (
        processor.config.lower_zone_channels
        if zone == MPEZone.LOWER
        else processor.config.upper_zone_channels
    )
    channels = [
        processor.create_mpe_note(60, 0.0, 1.0, 100, zone).channel
        for _ in members
    ]
    assert sorted(channels) == sorted(members)
    assert len(set(channels)) == len(channels)
    with pytest.raises(MPEChannelExhaustedError):
        processor.create_mpe_note(60, 0.0, 1.0, 100, zone)


def test_released_channel_can_be_allocated_again():
    processor = MPEProcessor()
    notes = [
        processor.create_mpe_note(60, 0.0, 1.0, 100, MPEZone.LOWER)
        for _ in processor.config.lower_zone_channels
    ]
    processor.release_note(notes[2])
    again = processor.create_mpe_note(62, 0.0, 1.0, 100, MPEZone.LOWER)
    assert again.channel == notes[2].channel


def test_allocator_without_config_still_returns_a_channel():
    assert MPEChannelAllocator().allocate_channel(MPEZone.LOWER) == 1


def test_zone_setup_sends_rpn_6_on_master_channels():
    processor = MPEProcessor()
    messages = processor.zone_setup_messages
    assert len(messages) == 6
    lower = [m for m in messages if m["channel"] == 0]
    upper = [m for m in messages if m["channel"] == 15]
    assert [(m["control"], m["value"]) for m in lower] == [
        (101, 0),
        (100, 6),
        (6, 7),
    ]
    assert [(m["control"], m["value"]) for m in upper] == [
        (101, 0),
        (100, 6),
        (6, 6),
    ]
    assert all(m["type"] == "control_change" for m in messages)


def test_zone_setup_skips_empty_zone():
    config = MPEConfiguration(
        lower_zone_channels=[1, 2, 3], upper_zone_channels=[]
    )
    messages = MPEProcessor(config).zone_setup_messages
    assert {m["channel"] for m in messages} == {0}


def test_processor_from_linnstrument_preset_uses_its_bend_range():
    mapper = MPEControllerMapper("linnstrument")
    processor = MPEProcessor.from_controller_mapper(mapper)
    assert processor.config.pitch_bend_range == 24
    note = MPENote(
        pitch=60,
        start_time=0.0,
        duration=1.0,
        velocity=100,
        channel=1,
        pressure=[],
        pitch_bend=[(0.5, 24.0)],
        timbre=[],
        slide=[],
    )
    bends = [
        m["value"]
        for m in processor.convert_to_midi(note)
        if m["type"] == "pitch_bend"
    ]
    assert bends == [16383]


def test_negative_full_bend_stays_in_range():
    processor = MPEProcessor()
    note = MPENote(60, 0.0, 1.0, 100, 1, [], [(0.1, -48.0)], [], [])
    bends = [
        m["value"]
        for m in processor.convert_to_midi(note)
        if m["type"] == "pitch_bend"
    ]
    assert bends == [0]


def _gliss_note():
    from korshmer.midi_processing.mpe_support import MPEGlissandoGenerator

    return MPEGlissandoGenerator(), MPEGlissandoGenerator().generate_glissando(
        60, 72, 0.0, 1.0
    )


@pytest.mark.parametrize("steps", [0, -3, 2.5, True, 10**9])
def test_glissando_rejects_invalid_step_count(steps):
    from korshmer.midi_processing.mpe_support import MPEGlissandoGenerator

    with pytest.raises(ValueError):
        MPEGlissandoGenerator().generate_glissando(
            60, 72, 0.0, 1.0, num_steps=steps
        )


@pytest.mark.parametrize(
    "start_time,duration",
    [(0.0, -1.0), (0.0, 0.0), (0.0, float("nan")), (float("inf"), 1.0)],
)
def test_glissando_rejects_invalid_time_values(start_time, duration):
    from korshmer.midi_processing.mpe_support import MPEGlissandoGenerator

    with pytest.raises(ValueError):
        MPEGlissandoGenerator().generate_glissando(60, 72, start_time, duration)


def test_glissando_rejects_invalid_start_pitch():
    from korshmer.midi_processing.mpe_support import MPEGlissandoGenerator

    with pytest.raises(ValueError):
        MPEGlissandoGenerator().generate_glissando(200, 72, 0.0, 1.0)


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -5.0, 0.0, 1e9])
def test_vibrato_rejects_invalid_rate(rate):
    generator, note = _gliss_note()
    note.pitch_bend = []
    with pytest.raises(ValueError):
        generator.generate_vibrato(note, rate_hz=rate)


def test_vibrato_rejects_non_finite_depth():
    generator, note = _gliss_note()
    with pytest.raises(ValueError):
        generator.generate_vibrato(note, depth_semitones=float("nan"))


def test_vibrato_defaults_still_add_points():
    generator, note = _gliss_note()
    note.pitch_bend = []
    generator.generate_vibrato(note)
    assert len(note.pitch_bend) == 50
