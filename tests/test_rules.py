from agent_token_optimizer import RuleConfig, TokenCounter, analyze_prompt, parse_code, analyze_code, parse_prompt


def test_rules_find_duplicates_verbose_messages_and_tool_descriptions() -> None:
    prompt = parse_prompt(
        {
            "messages": [
                {"role": "user", "content": "repeat this instruction"},
                {"role": "user", "content": "repeat this instruction"},
                {"role": "system", "content": "very " * 30},
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "A tool description that is intentionally long enough to optimize. " * 3,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A query description that is also intentionally long. " * 3,
                            }
                        },
                    },
                }
            ],
        }
    )

    findings = analyze_prompt(
        prompt,
        counter=TokenCounter(),
        config=RuleConfig(verbose_message_tokens=5, verbose_description_tokens=5),
    )
    rule_ids = {finding.rule_id for finding in findings}

    assert {"duplicate-message", "verbose-message", "verbose-tool-description"} <= rule_ids
    assert all(finding.estimated_token_savings > 0 for finding in findings)
    assert all(finding.location for finding in findings)


def test_rules_find_bloated_tool_schema() -> None:
    prompt = parse_prompt(
        {
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [
                {
                    "name": "catalog",
                    "description": "Catalog lookup. " * 20,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query. " * 20}
                        },
                    },
                }
            ],
        }
    )

    findings = analyze_prompt(
        prompt,
        counter=TokenCounter(),
        config=RuleConfig(bloated_tool_tokens=10),
    )

    assert any(finding.rule_id == "bloated-tool-schema" for finding in findings)


def test_code_rules_flag_dynamic_python_prompt_prefix() -> None:
    artifact = parse_code(
        "from datetime import datetime\n"
        "system_prompt = f'Run at {datetime.now()}'\n",
        language="python",
    )

    findings = analyze_code(artifact, counter=TokenCounter())

    assert len(findings) == 1
    assert findings[0].rule_id == "dynamic-header-prefix"
    assert "cached prefix" in findings[0].recommendation
