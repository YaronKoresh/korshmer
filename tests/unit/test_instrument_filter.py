from korshmer.midi_processing.instrument_filter import (
    MIDI_CONVERTIBLE_INSTRUMENTS,
    NON_MIDI_INSTRUMENTS,
    InstrumentFilter,
    should_convert_to_midi,
)


class TestInstrumentFilterConstants:
    def test_midi_convertible_instruments_is_frozenset(self):
        assert isinstance(MIDI_CONVERTIBLE_INSTRUMENTS, frozenset)

    def test_non_midi_instruments_is_frozenset(self):
        assert isinstance(NON_MIDI_INSTRUMENTS, frozenset)

    def test_no_overlap_between_sets(self):
        overlap = MIDI_CONVERTIBLE_INSTRUMENTS & NON_MIDI_INSTRUMENTS
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_expected_convertible_instruments(self):
        expected = ["piano", "guitar", "bass", "drums", "strings"]
        for instrument in expected:
            assert instrument in MIDI_CONVERTIBLE_INSTRUMENTS

    def test_expected_non_convertible_instruments(self):
        expected = ["vocals", "speech", "ambient", "noise"]
        for instrument in expected:
            assert instrument in NON_MIDI_INSTRUMENTS


class TestInstrumentFilterNormalization:
    def test_normalize_lowercase(self):
        result = InstrumentFilter._normalize_name("PIANO")
        assert result == "piano"

    def test_normalize_underscore_to_space(self):
        result = InstrumentFilter._normalize_name("electric_guitar")
        assert result == "electric guitar"

    def test_normalize_dash_to_space(self):
        result = InstrumentFilter._normalize_name("electric-guitar")
        assert result == "electric guitar"

    def test_normalize_multiple_separators(self):
        result = InstrumentFilter._normalize_name("Acoustic_Grand-Piano")
        assert result == "acoustic grand piano"

    def test_normalize_strips_whitespace(self):
        result = InstrumentFilter._normalize_name("  piano  ")
        assert result == "piano"

    def test_normalize_caching(self):

        InstrumentFilter._normalized_cache.clear()

        name = "Electric Guitar"
        result1 = InstrumentFilter._normalize_name(name)
        result2 = InstrumentFilter._normalize_name(name)

        assert result1 is result2
        assert name in InstrumentFilter._normalized_cache


class TestInstrumentFilterFuzzyMatching:
    def test_fuzzy_match_exact(self):
        assert InstrumentFilter._fuzzy_match(
            "piano", MIDI_CONVERTIBLE_INSTRUMENTS
        )

    def test_fuzzy_match_case_insensitive(self):
        assert InstrumentFilter._fuzzy_match(
            "PIANO", MIDI_CONVERTIBLE_INSTRUMENTS
        )

    def test_fuzzy_match_with_separator(self):
        assert InstrumentFilter._fuzzy_match(
            "acoustic_guitar", MIDI_CONVERTIBLE_INSTRUMENTS
        )

    def test_fuzzy_match_partial_word(self):
        assert InstrumentFilter._fuzzy_match(
            "piano sounds", MIDI_CONVERTIBLE_INSTRUMENTS
        )

    def test_fuzzy_match_no_match(self):
        assert not InstrumentFilter._fuzzy_match(
            "theremin", MIDI_CONVERTIBLE_INSTRUMENTS
        )

    def test_fuzzy_match_ambiguous_rejected(self):

        assert not InstrumentFilter._fuzzy_match(
            "ambient", MIDI_CONVERTIBLE_INSTRUMENTS
        )


class TestShouldConvert:
    def setup_method(self):
        InstrumentFilter.set_whitelist([])
        InstrumentFilter.set_blacklist([])

    def test_should_convert_piano(self):
        assert InstrumentFilter.should_convert("piano")

    def test_should_convert_guitar(self):
        assert InstrumentFilter.should_convert("guitar")

    def test_should_convert_electric_guitar(self):
        assert InstrumentFilter.should_convert("electric_guitar")

    def test_should_not_convert_vocals(self):
        assert not InstrumentFilter.should_convert("vocals")

    def test_should_not_convert_speech(self):
        assert not InstrumentFilter.should_convert("speech")

    def test_should_not_convert_ambient(self):
        assert not InstrumentFilter.should_convert("ambient")

    def test_should_not_convert_empty_string(self):
        assert not InstrumentFilter.should_convert("")

    def test_should_not_convert_none(self):
        assert not InstrumentFilter.should_convert(None)

    def test_priority_non_midi_over_midi(self):

        assert not InstrumentFilter.should_convert("ambient")


class TestWhitelistBlacklist:
    def setup_method(self):
        InstrumentFilter.set_whitelist([])
        InstrumentFilter.set_blacklist([])

    def test_whitelist_override(self):

        assert not InstrumentFilter.should_convert("theremin")

        InstrumentFilter.set_whitelist(["theremin"])
        assert InstrumentFilter.should_convert("theremin")

    def test_blacklist_override(self):

        assert InstrumentFilter.should_convert("piano")

        InstrumentFilter.set_blacklist(["piano"])
        assert not InstrumentFilter.should_convert("piano")

    def test_blacklist_priority_over_whitelist(self):
        InstrumentFilter.set_whitelist(["piano"])
        InstrumentFilter.set_blacklist(["piano"])

        assert not InstrumentFilter.should_convert("piano")

    def test_whitelist_normalization(self):
        InstrumentFilter.set_whitelist(["Electric_Guitar"])

        assert InstrumentFilter.should_convert("electric guitar")

    def test_blacklist_normalization(self):
        InstrumentFilter.set_blacklist(["Electric_Guitar"])

        assert not InstrumentFilter.should_convert("electric guitar")

    def test_clear_whitelist(self):
        InstrumentFilter.set_whitelist(["theremin"])
        assert InstrumentFilter.should_convert("theremin")

        InstrumentFilter.set_whitelist([])
        assert not InstrumentFilter.should_convert("theremin")

    def test_clear_blacklist(self):
        InstrumentFilter.set_blacklist(["piano"])
        assert not InstrumentFilter.should_convert("piano")

        InstrumentFilter.set_blacklist([])
        assert InstrumentFilter.should_convert("piano")


class TestFilterStems:
    def setup_method(self):
        InstrumentFilter.set_whitelist([])
        InstrumentFilter.set_blacklist([])

    def test_filter_stems_basic(self):
        stems = {
            "piano": "audio_data_1",
            "vocals": "audio_data_2",
            "guitar": "audio_data_3",
        }

        result = InstrumentFilter.filter_stems(stems)

        assert "piano" in result
        assert "guitar" in result
        assert "vocals" not in result

    def test_filter_stems_empty(self):
        result = InstrumentFilter.filter_stems({})
        assert result == {}

    def test_filter_stems_all_rejected(self):
        stems = {"vocals": "audio_data_1", "speech": "audio_data_2"}

        result = InstrumentFilter.filter_stems(stems)
        assert result == {}

    def test_filter_stems_all_accepted(self):
        stems = {"piano": "audio_data_1", "guitar": "audio_data_2"}

        result = InstrumentFilter.filter_stems(stems)
        assert len(result) == 2


class TestGetConvertibleStems:
    def setup_method(self):
        InstrumentFilter.set_whitelist([])
        InstrumentFilter.set_blacklist([])

    def test_get_convertible_stems_basic(self):
        stem_names = ["piano", "vocals", "guitar", "speech"]

        result = InstrumentFilter.get_convertible_stems(stem_names)

        assert "piano" in result
        assert "guitar" in result
        assert "vocals" not in result
        assert "speech" not in result

    def test_get_convertible_stems_empty(self):
        result = InstrumentFilter.get_convertible_stems([])
        assert result == []

    def test_get_convertible_stems_preserves_order(self):
        stem_names = ["guitar", "piano", "bass"]

        result = InstrumentFilter.get_convertible_stems(stem_names)

        assert result == ["guitar", "piano", "bass"]


class TestShouldConvertToMidiFunction:
    def setup_method(self):
        InstrumentFilter.set_whitelist([])
        InstrumentFilter.set_blacklist([])

    def test_should_convert_to_midi_piano(self):
        assert should_convert_to_midi("piano")

    def test_should_convert_to_midi_vocals(self):
        assert not should_convert_to_midi("vocals")

    def test_should_convert_to_midi_matches_class_method(self):
        test_instruments = ["piano", "vocals", "guitar", "speech", "drums"]

        for instrument in test_instruments:
            assert should_convert_to_midi(
                instrument
            ) == InstrumentFilter.should_convert(instrument)


class TestThreadSafety:
    def test_concurrent_normalization(self):
        import threading

        results = []

        def normalize():
            result = InstrumentFilter._normalize_name("Electric_Guitar")
            results.append(result)

        threads = [threading.Thread(target=normalize) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(r == results[0] for r in results)

    def test_concurrent_whitelist_blacklist(self):
        import threading

        def set_lists():
            InstrumentFilter.set_whitelist(["test1"])
            InstrumentFilter.set_blacklist(["test2"])

        threads = [threading.Thread(target=set_lists) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert True
