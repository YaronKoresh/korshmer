import json

import pytest

from korshmer.ai.copilot import MusicalContext, MusicCopilot


class Reply:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


@pytest.fixture
def service(monkeypatch):
    exchange = {
        "requests": [],
        "reply": {
            "response": "ok",
            "choices": [{"message": {"content": "ok"}}],
            "content": [{"type": "text", "text": "ok"}],
        },
    }

    def fake_urlopen(request, timeout=None):
        exchange["requests"].append(request)
        return Reply(exchange["reply"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return exchange


def ask(provider, **options):
    copilot = MusicCopilot(provider=provider, api_key="k", **options)
    return copilot.suggest(MusicalContext(), "next chord?")


def sent_body(service):
    return json.loads(service["requests"][0].data.decode("utf-8"))


def test_the_anthropic_request_states_its_api_version(service):
    ask("anthropic")
    request = service["requests"][0]
    assert request.get_header("Anthropic-version") == "2026-10-21"
    assert request.get_header("X-api-key") == "k"


def test_anthropic_uses_a_claude_model_by_default(service):
    ask("anthropic")
    assert sent_body(service)["model"] == "claude-opus-5-5"


def test_the_anthropic_request_sets_no_sampling_parameters_as_a_safeguard(
    service,
):
    ask("anthropic")
    assert not {"temperature", "top_p", "top_k"} & set(sent_body(service))


def test_openai_is_not_asked_for_an_ollama_model(service):
    ask("openai")
    assert sent_body(service)["model"] not in ("llama3", "")


def test_ollama_keeps_its_local_model_as_a_safeguard(service):
    ask("ollama")
    assert sent_body(service)["model"] == "llama3"


@pytest.mark.parametrize("provider", ["ollama", "openai", "anthropic"])
def test_a_model_the_user_chose_is_kept_as_a_safeguard(service, provider):
    assert ask(provider, model="my-model") == "ok"
    assert sent_body(service)["model"] == "my-model"


def test_the_answer_is_found_after_a_thinking_part(service):
    service["reply"] = {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": "Use a "},
            {"type": "text", "text": "ii-V-I."},
        ],
    }
    assert ask("anthropic") == "Use a ii-V-I."


def test_a_declined_request_gives_an_empty_answer_as_a_safeguard(service):
    service["reply"] = {"stop_reason": "refusal", "content": []}
    assert ask("anthropic") == ""


def test_the_anthropic_request_leaves_room_for_thinking_and_an_answer(service):
    ask("anthropic")
    assert sent_body(service)["max_tokens"] >= 16000
