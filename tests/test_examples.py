"""Keep the buildathon demo fixtures executable and meaningful."""

from __future__ import annotations

import json
from pathlib import Path

from agent_token_optimizer import RuleConfig
from audit_cli import run_audit


ROOT = Path(__file__).resolve().parents[1]


def _fixture_audit(prefix: str, *, rule_config: RuleConfig | None = None):
    return run_audit(
        code_source=(ROOT / "examples" / f"{prefix}agent.py").read_text(encoding="utf-8"),
        code_path=f"{prefix}agent.py",
        prompt_input=(ROOT / "examples" / f"{prefix}prompt.json").read_text(encoding="utf-8"),
        tools_input=(ROOT / "examples" / f"{prefix}tools.json").read_text(encoding="utf-8"),
        rule_config=rule_config,
    )


def test_checked_in_demo_shows_a_measurable_optimization() -> None:
    baseline = _fixture_audit("", rule_config=RuleConfig(bloated_tool_tokens=256))
    optimized = _fixture_audit("optimized_", rule_config=RuleConfig(bloated_tool_tokens=256))

    baseline_rule_ids = {finding.rule_id for finding in baseline.findings}
    assert {
        "dynamic-header-prefix",
        "duplicate-message",
        "verbose-message",
        "bloated-tool-schema",
        "verbose-tool-description",
    } <= baseline_rule_ids
    assert not optimized.findings
    assert baseline.prompt_tokens > optimized.prompt_tokens
    assert baseline.estimated_savings_tokens > 0


def test_checked_in_tool_fixture_is_valid_json() -> None:
    for filename in (
        "prompt.json",
        "tools.json",
        "optimized_prompt.json",
        "optimized_tools.json",
    ):
        json.loads((ROOT / "examples" / filename).read_text(encoding="utf-8"))
