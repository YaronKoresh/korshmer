"""Tests for swapping the bundled SoundFont for a different GM/GS bank.

The app is built around GeneralUser GS, which ships every General MIDI
program plus the GS drum kits the genres use (including kit 26, "Dance").
A user may replace ``fluidsynth/soundfont.sf3`` with a higher quality bank
(or point ``KORSHMER_SOUNDFONT`` at one). Many excellent free banks -- e.g.
MuseScore General / FluidR3 -- do not carry every GS drum kit, so the render
must fall back to a present kit of the same family instead of silently
dropping to whatever the synth defaults to.
"""

import importlib
import pathlib

import pytest

import korshmer.audio_processors as ap

BUNDLED_SF3 = (
    pathlib.Path(__file__).resolve().parents[2] / "fluidsynth" / "soundfont.sf3"
)


class TestDrumKitFallbackOrder:
    def test_requested_kit_comes_first(self):
        assert ap.drum_kit_fallback_order(26)[0] == 26

    def test_dance_kit_falls_through_the_electronic_family_then_standard(self):
        assert ap.drum_kit_fallback_order(26) == (26, 25, 24, 0)

    def test_jazz_falls_back_to_brush_then_standard(self):
        assert ap.drum_kit_fallback_order(32) == (32, 40, 0)

    def test_room_falls_back_to_power_then_standard(self):
        assert ap.drum_kit_fallback_order(8) == (8, 16, 0)

    def test_every_order_ends_at_the_standard_kit(self):
        for kit in (0, 8, 16, 24, 25, 26, 32, 40):
            assert ap.drum_kit_fallback_order(kit)[-1] == 0

    def test_order_has_no_duplicates(self):
        for kit in (0, 8, 16, 24, 25, 26, 32, 40):
            order = ap.drum_kit_fallback_order(kit)
            assert len(order) == len(set(order))


class TestResolveDrumKit:
    def test_present_kit_is_used_unchanged(self):
        available = {0, 8, 16, 24, 25, 26, 32, 40}
        assert ap.resolve_drum_kit(26, available) == 26

    def test_dance_kit_resolves_to_tr808_when_dance_is_absent(self):
        # MuseScore General / FluidR3 lack kit 26 but carry 24 and 25.
        available = {0, 8, 16, 24, 25, 32, 40}
        assert ap.resolve_drum_kit(26, available) == 25

    def test_dance_kit_resolves_to_electronic_when_808_also_absent(self):
        available = {0, 24, 32}
        assert ap.resolve_drum_kit(26, available) == 24

    def test_falls_back_to_standard_when_no_family_kit_exists(self):
        assert ap.resolve_drum_kit(26, {0}) == 0

    def test_jazz_resolves_to_brush_when_jazz_is_absent(self):
        assert ap.resolve_drum_kit(32, {0, 40}) == 40


class TestReadSoundfontKits:
    def test_bundled_soundfont_carries_every_kit_the_genres_use(self):
        if not BUNDLED_SF3.exists():
            pytest.skip("bundled soundfont missing")
        kits = ap.read_soundfont_kits(str(BUNDLED_SF3))
        for kit in (0, 8, 16, 24, 25, 26, 32, 40):
            assert kit in kits

    def test_missing_file_yields_empty_set_without_raising(self, tmp_path):
        assert ap.read_soundfont_kits(str(tmp_path / "nope.sf3")) == set()

    def test_non_riff_file_yields_empty_set(self, tmp_path):
        junk = tmp_path / "not-a-soundfont.sf3"
        junk.write_bytes(b"this is not a soundfont")
        assert ap.read_soundfont_presets(str(junk)) == set()

    def test_bad_path_type_is_handled_not_raised(self):
        # Inspection must never break rendering, even on a bad argument.
        assert ap.read_soundfont_kits(None) == set()


class TestSoundfontPathOverride:
    def test_env_var_overrides_the_default_soundfont_path(self, monkeypatch):
        monkeypatch.setenv("KORSHMER_SOUNDFONT", "/tmp/custom-bank.sf3")
        import korshmer.constants as constants

        constants = importlib.reload(constants)
        try:
            assert str(constants.SF3_PATH) == "/tmp/custom-bank.sf3"
        finally:
            monkeypatch.delenv("KORSHMER_SOUNDFONT", raising=False)
            importlib.reload(constants)

    def test_default_path_is_the_bundled_soundfont_without_the_env_var(
        self, monkeypatch
    ):
        monkeypatch.delenv("KORSHMER_SOUNDFONT", raising=False)
        import korshmer.constants as constants

        constants = importlib.reload(constants)
        assert (
            str(constants.SF3_PATH)
            .replace("\\", "/")
            .endswith("fluidsynth/soundfont.sf3")
        )
