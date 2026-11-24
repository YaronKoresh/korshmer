from korshmer.network.youtube_live import valid_stream_key


def test_a_stream_key_with_a_trailing_newline_is_refused():
    assert not valid_stream_key("abcd-1234-efgh\n")


def test_a_long_key_of_dashes_with_a_bad_last_character_is_refused():
    assert not valid_stream_key("a-" * 63 + "!")


def test_keys_made_of_letters_digits_dashes_and_underscores_are_accepted():
    assert valid_stream_key("abcd-1234_efgh")
    assert valid_stream_key("-leading-and-trailing-")
    assert valid_stream_key("x" * 128)
    assert not valid_stream_key("x" * 129)
