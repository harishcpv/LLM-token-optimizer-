"""Token counting with an optional local tiktoken backend and deterministic fallback."""

from __future__ import annotations

import json
import math
from typing import Any

from .models import Message, PromptSpec, ToolDefinition


def _fallback_count(text: str) -> int:
    """Estimate tokens without network access or a tokenizer dependency.

    Four UTF-8 bytes per token is intentionally conservative and stable across
    machines. It is an estimate, not a claim of model-specific BPE fidelity.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


class TokenCounter:
    """Count text and normalized prompts using tiktoken when available.

    The optional import is performed during construction, so importing this
    package never requires network access or an installed tokenizer package.
    """

    def __init__(self, encoding_name: str = "cl100k_base", *, message_overhead: int = 4) -> None:
        if message_overhead < 0:
            raise ValueError("message_overhead cannot be negative")
        self.encoding_name = encoding_name
        self.message_overhead = message_overhead
        self._encoder: Any = None
        self._backend = "fallback"
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoder = tiktoken.get_encoding(encoding_name)
            self._backend = "tiktoken"
        except (ImportError, LookupError, RuntimeError, ValueError):
            # tiktoken is optional; a missing encoding should use the same safe
            # fallback as a missing installation.
            self._encoder = None

    @property
    def backend(self) -> str:
        """Return ``tiktoken`` or ``fallback`` for observability."""

        return self._backend

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return _fallback_count(text)

    def count_message(self, message: Message) -> int:
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        total = self.count_text(message.role) + self.count_text(message.content) + self.message_overhead
        if message.name:
            total += self.count_text(message.name)
        return total

    def count_tool(self, tool: ToolDefinition) -> int:
        if not isinstance(tool, ToolDefinition):
            raise TypeError("tool must be a ToolDefinition")
        payload = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self.count_text(serialized)

    def count_prompt(self, prompt: PromptSpec) -> int:
        if not isinstance(prompt, PromptSpec):
            raise TypeError("prompt must be a PromptSpec")
        message_tokens = sum(self.count_message(message) for message in prompt.messages)
        tool_tokens = sum(self.count_tool(tool) for tool in prompt.tools)
        return message_tokens + tool_tokens
