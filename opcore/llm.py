"""Thin LLM boundary. Everything model-specific lives here; swap the class to change providers."""
from __future__ import annotations

import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class AzureOpenAIClient:
    """Uses an Azure OpenAI *deployment* (not a model name). Credentials come from the environment."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 deployment: str | None = None, api_version: str | None = None):
        from openai import AzureOpenAI
        self.deployment = deployment or os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self._c = AzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self._send_temperature = True

    def complete_json(self, system: str, user: str) -> dict:
        kw = dict(model=self.deployment, response_format={"type": "json_object"},
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self._send_temperature:
            kw["temperature"] = 0
        try:
            r = self._c.chat.completions.create(**kw)
        except Exception as e:  # some reasoning deployments reject `temperature`
            if self._send_temperature and "temperature" in str(e).lower():
                self._send_temperature = False
                kw.pop("temperature")
                r = self._c.chat.completions.create(**kw)
            else:
                raise
        self.calls += 1
        if r.usage:
            self.prompt_tokens += r.usage.prompt_tokens; self.completion_tokens += r.usage.completion_tokens
        return json.loads(r.choices[0].message.content)
