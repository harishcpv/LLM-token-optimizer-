import json

from agent_token_optimizer import RuleConfig
from audit_cli import run_audit


def test_audit_pipeline_combines_prompt_and_source_findings() -> None:
    result = run_audit(
        code_source=(
            "from datetime import datetime\n"
            "system_prompt = f'Generated at {datetime.now()}'\n"
        ),
        language="python",
        prompt_input="Use the catalog tool.",
        tools_input=json.dumps(
            {
                "name": "catalog",
                "description": "long " * 50,
                "parameters": {"type": "object"},
            }
        ),
        rule_config=RuleConfig(bloated_tool_tokens=10),
    )

    rule_ids = {finding.rule_id for finding in result.findings}
    assert "dynamic-header-prefix" in rule_ids
    assert "bloated-tool-schema" in rule_ids
    assert result.to_dict()["tokens"]["analyzed_total"] >= result.prompt_tokens
