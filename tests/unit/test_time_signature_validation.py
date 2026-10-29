import pytest

from korshmer.audio_processing.validation import validate_time_signature
from korshmer.midi_processing.midi_file_enhancement import MIDIFileEnhancer
from korshmer.midi_processing.pattern_generation import Arpeggiator
from korshmer.midi_processing.quantization import AdvancedQuantizer
from korshmer.midi_processing.score_notation import Measure


class TestValidateTimeSignature:
    @pytest.mark.parametrize("num", [0, -1, 65])
    def test_bad_numerator(self, num):
        with pytest.raises(ValueError):
            validate_time_signature((num, 4))

    @pytest.mark.parametrize("den", [0, 3, 5, 64])
    def test_bad_denominator(self, den):
        with pytest.raises(ValueError):
            validate_time_signature((4, den))

    @pytest.mark.parametrize("den", [1, 2, 4, 8, 16, 32])
    def test_good_denominators(self, den):
        for num in (1, 4, 7, 64):
            assert validate_time_signature((num, den)) == (num, den)

    @pytest.mark.parametrize(
        "ts",
        [("4", 4), (4, None), (True, 4), (4, False), (4.0, 4), (4, 4.0)],
    )
    def test_non_integer(self, ts):
        with pytest.raises(ValueError):
            validate_time_signature(ts)

    @pytest.mark.parametrize("ts", [(), (4,), (4, 4, 4), [4, 4, 4], 4, "4/4"])
    def test_wrong_shape(self, ts):
        with pytest.raises(ValueError):
            validate_time_signature(ts)

    def test_accepts_list(self):
        assert validate_time_signature([3, 8]) == (3, 8)


class TestEntryPoints:
    def test_measure_rejects_zero_denominator(self):
        with pytest.raises(ValueError):
            Measure(
                number=1,
                notes=[],
                time_signature=(0, 4),
                key_signature=0,
                dynamics=[],
            )

    def test_measure_valid(self):
        m = Measure(
            number=1,
            notes=[],
            time_signature=(4, 4),
            key_signature=0,
            dynamics=[],
        )
        assert m.get_duration() == 4.0

    def test_add_time_signature_valid(self):
        e = MIDIFileEnhancer()
        e.add_time_signature(0.0, 4, 4)
        assert len(e.time_signatures) == 1

    def test_add_time_signature_invalid(self):
        e = MIDIFileEnhancer()
        with pytest.raises(ValueError):
            e.add_time_signature(0.0, 4, 3)

    def test_arpeggiator_accepts_3_8(self):
        arp = Arpeggiator(time_signature=(3, 8))
        assert arp.time_signature == (3, 8)

    def test_arpeggiator_rejects_bad(self):
        with pytest.raises(ValueError):
            Arpeggiator(time_signature=(4, 0))

    def test_quantizer_rejects_bad(self):
        with pytest.raises(ValueError):
            AdvancedQuantizer(time_signature=(0, 4))
