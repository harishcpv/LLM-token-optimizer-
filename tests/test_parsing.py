import pytest

from agent_token_optimizer import Message, parse_code, parse_prompt, parse_tool_schema


def test_parse_code_unfences_and_infers_language() -> None:
    artifact = parse_code("```python\ndef greet():\n    return 'hi'\n```", path="example.py")

    assert artifact.language == "python"
    assert artifact.path == "example.py"
    assert artifact.source == "def greet():\n    return 'hi'"
    assert artifact.line_count == 2


def test_parse_prompt_accepts_json_messages_text_parts_and_function_tools() -> None:
    prompt = parse_prompt(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search records.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                }
            ],
        }
    )

    assert prompt.messages == (
        Message(role="system", content="Be concise."),
        Message(role="user", content="Hello"),
    )
    assert prompt.tools[0].name == "search"
    assert prompt.tools[0].parameters["properties"]["query"]["type"] == "string"


def test_parse_tool_schema_accepts_json_and_rejects_missing_name() -> None:
    tool = parse_tool_schema('{"name":"weather","parameters":{"type":"object"}}')
    assert tool.name == "weather"
    assert tool.parameters["type"] == "object"

    with pytest.raises(ValueError, match="requires a name"):
        parse_tool_schema({"type": "object", "properties": {}})
