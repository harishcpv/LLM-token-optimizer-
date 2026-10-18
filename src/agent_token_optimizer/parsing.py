"""Input normalization for source code, prompts, and tool schemas."""

from __future__ import annotations

import json
from pathlib import PurePath
import re
from typing import Any, Mapping, Sequence

from .models import CodeArtifact, Message, PromptSpec, ToolDefinition


_FENCED_CODE = re.compile(r"^\s*```([^\n`]*)\n(?P<body>.*)```\s*$", re.DOTALL)
_LANGUAGE_BY_SUFFIX = {
    ".c": "c",
    ".cpp": "cpp",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def parse_code(source: str, *, language: str | None = None, path: str | None = None) -> CodeArtifact:
    """Normalize plain source or a single Markdown fenced code block.

    An explicit language wins, followed by the fence marker, the file suffix, and
    finally ``text``. The source body is otherwise preserved, including indentation.
    """

    if not isinstance(source, str):
        raise TypeError("code source must be a string")

    body = source
    fence_language = ""
    match = _FENCED_CODE.match(source)
    if match:
        fence_language = match.group(1).strip().split(maxsplit=1)[0] if match.group(1).strip() else ""
        body = match.group("body")
        if body.endswith("\n"):
            body = body[:-1]

    suffix_language = _LANGUAGE_BY_SUFFIX.get(PurePath(path).suffix.lower(), "") if path else ""
    chosen_language = (language or fence_language or suffix_language or "text").strip().lower()
    if not chosen_language:
        raise ValueError("language cannot be blank")
    return CodeArtifact(source=body, language=chosen_language, path=path)


def _decode_json_if_needed(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input: {exc.msg}") from exc


def _parse_message(value: Message | Mapping[str, Any]) -> Message:
    if isinstance(value, Message):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("each prompt message must be a Message or mapping")
    if "role" not in value or "content" not in value:
        raise ValueError("each prompt message requires role and content")
    content = value["content"]
    if isinstance(content, list):
        # Accept the common text-part form without introducing a provider SDK.
        text_parts = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type", "text") == "text" and "text" in part:
                text_parts.append(str(part["text"]))
            else:
                raise ValueError("message content lists may only contain text parts")
        content = "".join(text_parts)
    return Message(role=str(value["role"]).lower(), content=content, name=value.get("name"))


def parse_prompt(value: PromptSpec | str | Mapping[str, Any] | Sequence[Message | Mapping[str, Any]]) -> PromptSpec:
    """Parse a prompt string, message list, or provider-shaped prompt mapping."""

    if isinstance(value, PromptSpec):
        return value
    if isinstance(value, str):
        value = _decode_json_if_needed(value)
        if isinstance(value, str):
            return PromptSpec(messages=(Message(role="user", content=value),))

    raw_tools: Any = []
    raw_messages: Any = value
    if isinstance(value, Mapping):
        if "messages" in value:
            raw_messages = value["messages"]
        elif "role" in value and "content" in value:
            raw_messages = [value]
        else:
            raise ValueError("prompt mapping requires a messages field")
        raw_tools = value.get("tools", [])

    if isinstance(raw_messages, (str, bytes)) or not isinstance(raw_messages, Sequence):
        raise TypeError("prompt messages must be a sequence")
    if isinstance(raw_tools, (str, bytes)) or not isinstance(raw_tools, Sequence):
        raise TypeError("prompt tools must be a sequence")

    messages = tuple(_parse_message(item) for item in raw_messages)
    tools = tuple(parse_tool_schema(item) for item in raw_tools)
    return PromptSpec(messages=messages, tools=tools)


def parse_tool_schema(
    value: ToolDefinition | str | Mapping[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolDefinition:
    """Normalize direct or OpenAI-style function tool schemas.

    ``value`` may be a JSON string, ``{"name": ..., "parameters": ...}``,
    or ``{"type": "function", "function": {...}}``. A raw parameter schema
    is also accepted when ``name`` is supplied.
    """

    if isinstance(value, ToolDefinition):
        return value
    if isinstance(value, str):
        decoded = _decode_json_if_needed(value)
        if isinstance(decoded, str):
            raise ValueError("tool schema string must contain a JSON object")
        value = decoded
    if not isinstance(value, Mapping):
        raise TypeError("tool schema must be a mapping or JSON object string")

    data: Mapping[str, Any] = value
    if isinstance(data.get("function"), Mapping):
        data = data["function"]

    tool_name = name or data.get("name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool schema requires a name")
    tool_description = description if description is not None else data.get("description", "")
    if not isinstance(tool_description, str):
        raise TypeError("tool description must be a string")

    if "parameters" in data:
        parameters = data["parameters"]
    elif any(key in data for key in ("type", "properties", "required", "additionalProperties")):
        parameters = data
    else:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, Mapping):
        raise TypeError("tool parameters must be a JSON object")
    return ToolDefinition(name=tool_name.strip(), description=tool_description, parameters=parameters)
