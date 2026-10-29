from __future__ import annotations

import getpass
import logging
import os
import re
import sys
from collections.abc import Mapping

from korshmer.utils.audit_log import OUTCOME_FAILED, audit

SERVICE_NAME = "korshmer"
ENV_PREFIX = "KORSHMER_"

_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

_logger = logging.getLogger(__name__)


class CredentialStoreError(RuntimeError):
    pass


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            "credential name must be upper case letters, digits and "
            "underscores, starting with a letter"
        )
    return name


def _load_keyring():
    try:
        import keyring
        import keyring.backends.fail
    except ImportError:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(backend, keyring.backends.fail.Keyring):
        return None
    return keyring


def keychain_available(keyring_module=None) -> bool:
    module = keyring_module or _load_keyring()
    return module is not None


def get_secret(
    name: str,
    environ: Mapping[str, str] | None = None,
    keyring_module=None,
) -> str:
    _check_name(name)
    env = os.environ if environ is None else environ
    value = (env.get(ENV_PREFIX + name) or "").strip()
    if value:
        return value
    module = keyring_module or _load_keyring()
    if module is None:
        return ""
    try:
        stored = module.get_password(SERVICE_NAME, name)
    except Exception as e:
        _logger.warning(
            "Could not read %s from the system keychain: %s", name, e
        )
        audit(
            "credentials.read",
            OUTCOME_FAILED,
            name=name,
            reason=type(e).__name__,
        )
        return ""
    return (stored or "").strip()


def set_secret(name: str, value: str, keyring_module=None) -> None:
    _check_name(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("credential value must be non-empty text")
    module = keyring_module or _load_keyring()
    if module is None:
        audit(
            "credentials.store", OUTCOME_FAILED, name=name, reason="no_keychain"
        )
        raise CredentialStoreError("no system keychain is available")
    try:
        module.set_password(SERVICE_NAME, name, value.strip())
    except Exception as e:
        audit(
            "credentials.store",
            OUTCOME_FAILED,
            name=name,
            reason=type(e).__name__,
        )
        raise CredentialStoreError(f"could not store {name}: {e}") from e
    audit("credentials.store", name=name)


def delete_secret(name: str, keyring_module=None) -> bool:
    _check_name(name)
    module = keyring_module or _load_keyring()
    if module is None:
        audit(
            "credentials.delete",
            OUTCOME_FAILED,
            name=name,
            reason="no_keychain",
        )
        raise CredentialStoreError("no system keychain is available")
    try:
        module.delete_password(SERVICE_NAME, name)
    except Exception as e:
        if type(e).__name__ == "PasswordDeleteError":
            return False
        audit(
            "credentials.delete",
            OUTCOME_FAILED,
            name=name,
            reason=type(e).__name__,
        )
        raise CredentialStoreError(f"could not delete {name}: {e}") from e
    audit("credentials.delete", name=name)
    return True


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    usage = "usage: python -m korshmer.utils.credential_store set|delete NAME"
    if len(args) != 2 or args[0] not in ("set", "delete"):
        print(usage, file=sys.stderr)
        return 2
    action, name = args
    try:
        if action == "set":
            set_secret(name, getpass.getpass(f"{name}: "))
        elif not delete_secret(name):
            print(f"{name} was not stored", file=sys.stderr)
            return 1
    except (ValueError, CredentialStoreError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
