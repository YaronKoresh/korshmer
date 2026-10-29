import pytest

from korshmer.utils import credential_store as cs


class Keyring:
    def __init__(self):
        self.data = {}

    def set_password(self, service, name, value):
        self.data[(service, name)] = value


@pytest.mark.parametrize("value", [5, b"secret", ["a"], None, 1.5, True])
def test_value_that_is_not_text_is_a_value_error(value):
    keyring = Keyring()
    with pytest.raises(ValueError, match="credential value"):
        cs.set_secret("TWITCH_TOKEN", value, keyring)
    assert keyring.data == {}


def test_text_value_is_stored_trimmed():
    keyring = Keyring()
    cs.set_secret("TWITCH_TOKEN", " abc ", keyring)
    assert keyring.data[("korshmer", "TWITCH_TOKEN")] == "abc"
