"""Gemini API client used for forcing interpretation and planning narratives."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


class GeminiError(RuntimeError):
    """Raised when Gemini returns an unusable response."""


@dataclass
class GeminiClient:
    """Thin REST client for the Gemini API.

    The implementation intentionally uses raw HTTP so the project can run in
    lightweight environments without an additional SDK dependency.
    """

    api_key: str
    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: float = 30.0

    @classmethod
    def from_env(
        cls,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> "GeminiClient":
        """Construct a client from environment variables."""
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise GeminiError(f"Missing Gemini API key in environment variable {api_key_env}.")
        return cls(api_key=api_key, model=model, base_url=base_url)

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        """Generate a text response from Gemini."""
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if response_mime_type:
            payload["generationConfig"] = {"responseMimeType": response_mime_type}

        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError) as exc:
            raise GeminiError(f"Unexpected Gemini response payload: {data}") from exc

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object response from Gemini."""
        text = self.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type="application/json",
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiError(f"Gemini did not return valid JSON: {text}") from exc
