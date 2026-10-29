from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from korshmer.midi_processing.note_validation import (
    MAX_CHORD_PROGRESSION_LENGTH,
)
from korshmer.network.circuit_breaker import CircuitBreaker, CircuitOpenError
from korshmer.utils.credential_store import get_secret
from korshmer.utils.tempo import validate_bpm


@dataclass
class MusicalContext:
    key: str = "C"
    scale: str = "major"
    bpm: float = 120.0
    time_signature: str = "4/4"
    instrument: str = "piano"
    midi_notes: list[dict] = field(default_factory=list)
    chord_progression: list[str] = field(default_factory=list)
    genre: str = ""
    mood: str = ""


class MusicCopilot:
    def __init__(
        self, provider: str = "ollama", model: str = "llama3", api_key: str = ""
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self._breakers = {
            name: CircuitBreaker(name)
            for name in ("ollama", "openai", "anthropic")
        }
        self.SYSTEM_PROMPT = (
            "You are an expert music theory assistant helping with a digital audio workstation. "
            "Provide concise, actionable suggestions for musical composition, arrangement, and harmony."
        )

    def _build_prompt(self, context: MusicalContext, request: str) -> str:
        if not isinstance(request, str):
            raise ValueError(f"request must be text, got {request!r}")
        chords = context.chord_progression
        if not isinstance(chords, list) or not all(
            isinstance(c, str) for c in chords
        ):
            raise ValueError("chord progression must be a list of chord names")
        if len(context.chord_progression) > MAX_CHORD_PROGRESSION_LENGTH:
            raise ValueError(
                f"chord progression must contain at most "
                f"{MAX_CHORD_PROGRESSION_LENGTH} chords, got "
                f"{len(context.chord_progression)}"
            )
        context_dict = {
            "key": context.key,
            "scale": context.scale,
            "bpm": validate_bpm(context.bpm),
            "time_signature": context.time_signature,
            "instrument": context.instrument,
            "chord_progression": context.chord_progression,
            "genre": context.genre,
            "mood": context.mood,
        }
        prompt = {
            "system": self.SYSTEM_PROMPT,
            "context": context_dict,
            "request": request,
        }
        return json.dumps(prompt)

    def _guarded(self, provider: str, func: Callable[[], str]) -> str:
        try:
            return self._breakers[provider].call(func)
        except CircuitOpenError:
            self.logger.warning(
                f"{provider} calls paused after repeated failures"
            )
            return ""
        except Exception as e:
            self.logger.error(f"{provider} call failed: {e}")
            return ""

    def _call_ollama(self, prompt: str) -> str:
        return self._guarded("ollama", lambda: self._request_ollama(prompt))

    def _request_ollama(self, prompt: str) -> str:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(
                {"model": self.model, "prompt": prompt, "stream": False}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "")

    def _call_openai(self, prompt: str) -> str:
        return self._guarded("openai", lambda: self._request_openai(prompt))

    def _request_openai(self, prompt: str) -> str:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: str) -> str:
        return self._guarded(
            "anthropic", lambda: self._request_anthropic(prompt)
        )

    def _request_anthropic(self, prompt: str) -> str:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": self.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data["content"][0]["text"]

    def suggest(
        self,
        context: MusicalContext,
        request: str,
        callback: Callable[[str], None] | None = None,
    ) -> str:
        prompt = self._build_prompt(context, request)

        if self.provider == "ollama":
            result = self._call_ollama(prompt)
        elif self.provider == "openai":
            result = self._call_openai(prompt)
        elif self.provider == "anthropic":
            result = self._call_anthropic(prompt)
        else:
            result = ""
            self.logger.error(f"Unknown provider: {self.provider}")

        if callback:
            callback(result)

        return result

    def suggest_async(
        self,
        context: MusicalContext,
        request: str,
        callback: Callable[[str], None],
    ):
        thread = threading.Thread(
            target=lambda: self.suggest(context, request, callback), daemon=True
        )
        thread.start()

    def is_available(self) -> bool:
        if self.provider == "ollama":
            if self._breakers["ollama"].state == "open":
                return False
            try:
                urllib.request.urlopen(
                    urllib.request.Request("http://localhost:11434"), timeout=2
                )
                return True
            except Exception:
                return False
        elif self.provider in ["openai", "anthropic"]:
            return bool(self.api_key)
        return False

    def suggest_chords(self, context: MusicalContext) -> str:
        return self.suggest(
            context,
            "Suggest 4 chord progressions appropriate for this musical context",
        )

    def suggest_melody(self, context: MusicalContext) -> str:
        return self.suggest(
            context, "Suggest a melody that fits this musical context"
        )

    def suggest_arrangement(self, context: MusicalContext) -> str:
        return self.suggest(
            context, "Suggest an arrangement with multiple instruments"
        )

    def detect_tension_notes(self, context: MusicalContext) -> str:
        return self.suggest(
            context,
            "Identify tension notes and dissonances in the current progression",
        )

    def suggest_instruments(self, context: MusicalContext) -> str:
        return self.suggest(
            context, "Suggest complementary instruments for this arrangement"
        )


_copilot: MusicCopilot | None = None


def get_copilot(provider: str = "ollama") -> MusicCopilot:
    global _copilot
    if _copilot is None:
        _copilot = MusicCopilot(
            provider=provider, api_key=get_secret("COPILOT_API_KEY")
        )
    return _copilot
