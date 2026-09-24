"""Thin Gemini client shared by the media pass and the routing pass.

Kept deliberately small: model selection, retry with backoff for free tier rate limits,
and tolerant JSON extraction. Nothing here knows anything about routing.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

# Free tier quota is granted per model per day, so a single model is a single point of
# failure: gemini-2.5-flash allows 20 requests a day and this run needs about 143. The chain
# is ordered by capability, and the client walks down it as each model's daily quota is
# exhausted. Every model listed accepts image and audio input, which the media pass needs.
MODEL_PREFERENCE = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest",
)

MAX_RETRIES = 4
BASE_BACKOFF = 2.0

# Daily quota is not worth retrying: the reset is hours away. Per-minute quota is.
DAILY_QUOTA_MARKERS = ("perday", "requestsperday", "perdayperproject")
RETRYABLE_MARKERS = ("429", "resource_exhausted", "503", "500", "unavailable", "deadline")


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from google import genai  # imported lazily so --no-model needs no SDK

        key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        self._genai = genai
        self.client = genai.Client(api_key=key)

        forced = model or os.environ.get("GEMINI_MODEL", "").strip()
        self.chain: list[str] = [forced] if forced else self._available_chain()
        self.index = 0
        self.exhausted: list[str] = []

    def _available_chain(self) -> list[str]:
        """Keep only the preferred models this key can actually reach."""
        try:
            available = {m.name.split("/")[-1] for m in self.client.models.list()}
        except Exception:  # noqa: BLE001 - listing is a convenience, not a requirement
            return list(MODEL_PREFERENCE)
        chain = [m for m in MODEL_PREFERENCE if m in available]
        return chain or list(MODEL_PREFERENCE)

    @property
    def model(self) -> str:
        return self.chain[min(self.index, len(self.chain) - 1)]

    def _rotate(self) -> bool:
        """Move to the next model after a daily quota exhaustion. False when none are left."""
        self.exhausted.append(self.model)
        self.index += 1
        if self.index < len(self.chain):
            print(f"    daily quota exhausted, switching to {self.model}")
            return True
        return False

    def generate(self, parts: list[Any], system_instruction: str | None = None, temperature: float = 0.0) -> str:
        """Call the model with retry and model rotation.

        Temperature 0: the challenge asks for deterministic behaviour where possible, and
        routing is a classification task, not a generative one.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )

        attempt = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=parts, config=config
                )
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower().replace(" ", "").replace("_", "")

                # Daily quota gone: this model is done for the day, move to the next one.
                if "429" in str(exc) and any(m in message for m in DAILY_QUOTA_MARKERS):
                    if self._rotate():
                        attempt = 0
                        continue
                    raise RuntimeError(
                        f"daily free tier quota exhausted on all models: {', '.join(self.exhausted)}"
                    ) from exc

                attempt += 1
                if attempt >= MAX_RETRIES or not any(t in message for t in RETRYABLE_MARKERS):
                    raise
                sleep_for = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"    rate limited, retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)

    @staticmethod
    def part_from_file(path: str, mime_type: str) -> Any:
        from google.genai import types

        with open(path, "rb") as fh:
            return types.Part.from_bytes(data=fh.read(), mime_type=mime_type)


def parse_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON extraction. Models occasionally wrap output in prose or fences."""
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class JsonCache:
    """Small on-disk cache. Keeps iteration free and makes re-runs reproducible."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.data: dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False, sort_keys=True)
