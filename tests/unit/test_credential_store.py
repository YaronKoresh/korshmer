import pytest

from korshmer.network.twitch_client import settings_from_env
from korshmer.utils import credential_store as cs


class PasswordDeleteError(Exception):
    pass


class FakeKeyring:
    def __init__(self):
        self.data = {}
        self.fail = False

    def get_password(self, service, name):
        if self.fail:
            raise OSError("locked")
        return self.data.get((service, name))

    def set_password(self, service, name, value):
        if self.fail:
            raise OSError("locked")
        self.data[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.data:
            raise PasswordDeleteError(name)
        del self.data[(service, name)]


@pytest.fixture
def kr():
    return FakeKeyring()


def test_env_wins_over_keychain(kr):
    kr.data[("korshmer", "OBS_PASSWORD")] = "stored"
    env = {"KORSHMER_OBS_PASSWORD": "from-env"}
    assert cs.get_secret("OBS_PASSWORD", env, kr) == "from-env"


def test_keychain_used_when_env_missing(kr):
    kr.data[("korshmer", "OBS_PASSWORD")] = " stored "
    assert cs.get_secret("OBS_PASSWORD", {}, kr) == "stored"


def test_missing_everywhere_is_empty(kr):
    assert cs.get_secret("OBS_PASSWORD", {}, kr) == ""


def test_no_keychain_is_empty(monkeypatch):
    monkeypatch.setattr(cs, "_load_keyring", lambda: None)
    assert cs.get_secret("OBS_PASSWORD", {}) == ""
    assert cs.keychain_available() is False
    with pytest.raises(cs.CredentialStoreError):
        cs.set_secret("OBS_PASSWORD", "x")
    with pytest.raises(cs.CredentialStoreError):
        cs.delete_secret("OBS_PASSWORD")


def test_read_error_falls_back_to_empty(kr):
    kr.fail = True
    assert cs.get_secret("OBS_PASSWORD", {}, kr) == ""


def test_set_and_delete_round_trip(kr):
    cs.set_secret("TWITCH_TOKEN", " abc ", kr)
    assert kr.data[("korshmer", "TWITCH_TOKEN")] == "abc"
    assert cs.delete_secret("TWITCH_TOKEN", kr) is True
    assert cs.delete_secret("TWITCH_TOKEN", kr) is False


def test_set_failure_raises_store_error(kr):
    kr.fail = True
    with pytest.raises(cs.CredentialStoreError):
        cs.set_secret("TWITCH_TOKEN", "abc", kr)


@pytest.mark.parametrize("name", ["", "lower", "1ABC", "A-B", "A" * 65])
def test_bad_names_rejected(name, kr):
    with pytest.raises(ValueError):
        cs.get_secret(name, {}, kr)


def test_empty_value_rejected(kr):
    with pytest.raises(ValueError):
        cs.set_secret("TWITCH_TOKEN", "   ", kr)


def test_audit_never_contains_value(kr, caplog):
    caplog.set_level("INFO", logger="korshmer.audit")
    cs.set_secret("TWITCH_TOKEN", "super-secret-value", kr)
    assert "credentials.store" in caplog.text
    assert "super-secret-value" not in caplog.text


def test_twitch_settings_use_keychain_when_env_has_no_token(monkeypatch):
    monkeypatch.delenv("KORSHMER_TWITCH_TOKEN", raising=False)
    monkeypatch.setattr(
        "korshmer.network.twitch_client.get_secret",
        lambda name: "kc-token" if name == "TWITCH_TOKEN" else "",
    )
    assert settings_from_env()["token"] == "kc-token"
    assert settings_from_env({})["token"] == ""


def test_cli_set_reads_hidden_prompt(monkeypatch, kr):
    monkeypatch.setattr(cs, "_load_keyring", lambda: kr)
    monkeypatch.setattr(cs.getpass, "getpass", lambda prompt: "typed")
    assert cs.main(["set", "OBS_PASSWORD"]) == 0
    assert kr.data[("korshmer", "OBS_PASSWORD")] == "typed"
    assert cs.main(["delete", "OBS_PASSWORD"]) == 0
    assert cs.main(["delete", "OBS_PASSWORD"]) == 1
    assert cs.main(["bogus"]) == 2
