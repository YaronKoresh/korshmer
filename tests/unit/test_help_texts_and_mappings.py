from korshmer.constants import DRUM_MAPPINGS, HELP_TEXTS, STEM_TYPE_PATTERNS


class TestHelpTexts:
    def test_help_texts_is_dict(self):
        assert isinstance(HELP_TEXTS, dict)

    def test_help_texts_not_empty(self):
        assert len(HELP_TEXTS) > 0

    def test_all_values_are_strings(self):
        for key, value in HELP_TEXTS.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value for {key!r} is not a string"

    def test_all_values_not_empty(self):
        for key, value in HELP_TEXTS.items():
            assert len(value) > 0, f"Help text for {key!r} is empty"

    def test_common_keys_present(self):
        expected_keys = [
            "bpm",
            "quantize",
            "swing",
            "transpose",
            "velocity",
            "grid",
            "scale",
            "root",
        ]
        for key in expected_keys:
            assert key in HELP_TEXTS, f"Expected key {key!r} not in HELP_TEXTS"

    def test_all_keys_lowercase(self):
        for key in HELP_TEXTS:
            assert key == key.lower(), f"Key {key!r} should be lowercase"


class TestDrumMappings:
    def test_drum_mappings_is_dict(self):
        assert isinstance(DRUM_MAPPINGS, dict)

    def test_drum_mappings_has_known_plugins(self):
        expected_plugins = [
            "Addictive Drums 2",
            "EZDrummer / Superior",
            "Steven Slate",
        ]
        for plugin in expected_plugins:
            assert plugin in DRUM_MAPPINGS, (
                f"Expected plugin {plugin!r} not in DRUM_MAPPINGS"
            )

    def test_all_mappings_are_dicts(self):
        for plugin, mapping in DRUM_MAPPINGS.items():
            assert isinstance(mapping, dict), (
                f"Mapping for {plugin!r} is not a dict"
            )

    def test_all_mapping_values_in_midi_range(self):
        for plugin, mapping in DRUM_MAPPINGS.items():
            for source, target in mapping.items():
                assert 0 <= source <= 127, (
                    f"Source note {source} in {plugin!r} out of MIDI range"
                )
                assert 0 <= target <= 127, (
                    f"Target note {target} in {plugin!r} out of MIDI range"
                )


class TestStemTypePatterns:
    def test_stem_type_patterns_is_dict(self):
        assert isinstance(STEM_TYPE_PATTERNS, dict)

    def test_has_common_stems(self):
        expected = ["drums", "bass", "vocals", "guitar", "piano", "other"]
        for stem in expected:
            assert stem in STEM_TYPE_PATTERNS, (
                f"Expected stem type {stem!r} not in STEM_TYPE_PATTERNS"
            )

    def test_all_patterns_are_lists(self):
        for stem, patterns in STEM_TYPE_PATTERNS.items():
            assert isinstance(patterns, list), (
                f"Patterns for {stem!r} should be a list"
            )

    def test_all_patterns_not_empty(self):
        for stem, patterns in STEM_TYPE_PATTERNS.items():
            assert len(patterns) > 0, (
                f"Patterns for {stem!r} should not be empty"
            )
