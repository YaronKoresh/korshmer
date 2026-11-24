import ssl

from korshmer.network.twitch_client import chat_tls_context


def test_twitch_chat_refuses_tls_older_than_1_2():
    assert chat_tls_context().minimum_version >= ssl.TLSVersion.TLSv1_2


def test_twitch_chat_still_checks_the_server_certificate_and_name():
    context = chat_tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
