from agent_token_optimizer import Message, PromptSpec, TokenCounter, ToolDefinition


def test_counter_is_stable_and_counts_prompt_components() -> None:
    counter = TokenCounter(message_overhead=4)
    assert counter.count_text("") == 0
    assert counter.count_text("same input") == counter.count_text("same input")

    prompt = PromptSpec(
        messages=(Message(role="user", content="Summarize this."),),
        tools=(
            ToolDefinition(
                name="lookup",
                description="Look up a record.",
                parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ),
    )
    assert counter.count_prompt(prompt) >= counter.count_message(prompt.messages[0])
    assert counter.count_tool(prompt.tools[0]) > 0
    assert counter.backend in {"fallback", "tiktoken"}


def test_unicode_text_remains_countable_without_provider_specific_assumptions() -> None:
    counter = TokenCounter()
    assert counter.count_text("東京 café") > 0
    assert counter.count_text("a" * 100) >= counter.count_text("a")


def test_missing_tiktoken_uses_deterministic_fallback(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def block_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("optional dependency intentionally hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_tiktoken)
    counter = TokenCounter()

    assert counter.backend == "fallback"
    assert counter.count_text("12345") == 2
