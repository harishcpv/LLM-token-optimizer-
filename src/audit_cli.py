"""Shared audit orchestration and a small JSON CLI for Agent Token Optimizer.

The Streamlit app deliberately calls the same ``run_audit`` function as this CLI.
That keeps ingestion, counting, simulation, and recommendation numbers identical
across the interactive and CI-friendly entry points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from agent_token_optimizer import (
        CodeArtifact,
        OptimizationFinding,
        Pricing,
        PromptSpec,
        RuleConfig,
        SimulationConfig,
        SimulationResult,
        TokenCounter,
        analyze_code,
        analyze_prompt,
        estimate_cost,
        parse_code,
        parse_prompt,
        parse_tool_schema,
        simulate_conversation,
    )
except ModuleNotFoundError:
    # Running ``python audit_cli.py`` from a source checkout should work before
    # the package has been installed. Installed environments take the normal
    # import path above.
    source_root = Path(__file__).resolve().parent
    if not source_root.is_dir():
        raise
    sys.path.insert(0, str(source_root))
    from agent_token_optimizer import (
        CodeArtifact,
        OptimizationFinding,
        Pricing,
        PromptSpec,
        RuleConfig,
        SimulationConfig,
        SimulationResult,
        TokenCounter,
        analyze_code,
        analyze_prompt,
        estimate_cost,
        parse_code,
        parse_prompt,
        parse_tool_schema,
        simulate_conversation,
    )


def parse_tools_input(value: str) -> tuple[Any, ...]:
    """Parse a JSON array/object containing tool schemas.

    A wrapper object such as ``{"tools": [...]}`` is accepted because it is a
    common export format. Empty input means that embedded prompt tools, if any,
    should be used by the caller.
    """

    if not isinstance(value, str):
        raise TypeError("tool input must be a string")
    if not value.strip():
        return ()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tool JSON: {exc.msg}") from exc

    if isinstance(decoded, Mapping) and "tools" in decoded:
        decoded = decoded["tools"]
    if isinstance(decoded, Mapping):
        decoded = [decoded]
    if not isinstance(decoded, list):
        raise TypeError('tool input must be a JSON object, array, or {"tools": [...]}')
    return tuple(parse_tool_schema(item) for item in decoded)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """All deterministic values needed to render or export a token audit."""

    code: CodeArtifact
    prompt: PromptSpec
    counter_backend: str
    code_tokens: int
    message_tokens: int
    tool_tokens: int
    prompt_tokens: int
    findings: tuple[OptimizationFinding, ...]
    simulation: SimulationResult
    pricing: Pricing
    estimated_savings_tokens: int
    optimized_prompt_tokens: int
    one_turn_cost: float
    optimized_one_turn_cost: float

    @property
    def estimated_savings_ratio(self) -> float:
        """Return the finding-based prompt reduction as a bounded ratio."""

        if self.prompt_tokens <= 0:
            return 0.0
        return min(self.estimated_savings_tokens / self.prompt_tokens, 1.0)

    @property
    def one_turn_cost_savings(self) -> float:
        return max(self.one_turn_cost - self.optimized_one_turn_cost, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report without requiring a dataclass/JSON encoder."""

        return {
            "code": {
                "language": self.code.language,
                "path": self.code.path,
                "line_count": self.code.line_count,
                "tokens": self.code_tokens,
            },
            "prompt": {
                "message_count": len(self.prompt.messages),
                "tool_count": len(self.prompt.tools),
                "tool_names": [tool.name for tool in self.prompt.tools],
            },
            "tokens": {
                "backend": self.counter_backend,
                "messages": self.message_tokens,
                "tools": self.tool_tokens,
                "prompt_total": self.prompt_tokens,
                "analyzed_total": self.code_tokens + self.prompt_tokens,
            },
            "pricing": {
                "input_per_1k": self.pricing.input_per_1k,
                "output_per_1k": self.pricing.output_per_1k,
            },
            "simulation": {
                "turns": [
                    {
                        "turn": turn.turn_number,
                        "input_tokens": turn.input_tokens,
                        "output_tokens": turn.output_tokens,
                        "total_tokens": turn.total_tokens,
                        "cost": turn.cost,
                    }
                    for turn in self.simulation.turns
                ],
                "total_input_tokens": self.simulation.total_input_tokens,
                "total_output_tokens": self.simulation.total_output_tokens,
                "total_tokens": self.simulation.total_tokens,
                "total_cost": self.simulation.total_cost,
            },
            "findings": [
                {
                    "rule_id": finding.rule_id,
                    "severity": finding.severity.value,
                    "title": finding.title,
                    "description": finding.description,
                    "recommendation": finding.recommendation,
                    "estimated_token_savings": finding.estimated_token_savings,
                    "location": finding.location,
                }
                for finding in self.findings
            ],
            "savings": {
                "estimated_tokens": self.estimated_savings_tokens,
                "optimized_prompt_tokens": self.optimized_prompt_tokens,
                "ratio": self.estimated_savings_ratio,
                "one_turn_cost": self.one_turn_cost,
                "optimized_one_turn_cost": self.optimized_one_turn_cost,
                "one_turn_cost_savings": self.one_turn_cost_savings,
            },
        }


def run_audit(
    *,
    code_source: str = "",
    code_path: str | None = None,
    language: str | None = None,
    prompt_input: str = "",
    tools_input: str = "",
    pricing: Pricing | None = None,
    simulation_config: SimulationConfig | None = None,
    rule_config: RuleConfig | None = None,
) -> AuditResult:
    """Normalize inputs and calculate a complete deterministic audit report."""

    if not any((code_source.strip(), prompt_input.strip(), tools_input.strip())):
        raise ValueError("provide agent code, a prompt, or at least one tool schema")

    code = parse_code(code_source, language=language, path=code_path)
    parsed_prompt = parse_prompt(prompt_input) if prompt_input.strip() else PromptSpec()
    explicit_tools = parse_tools_input(tools_input)
    prompt = PromptSpec(
        messages=parsed_prompt.messages,
        tools=explicit_tools if tools_input.strip() else parsed_prompt.tools,
    )

    counter = TokenCounter()
    pricing = pricing or Pricing()
    simulation_config = simulation_config or SimulationConfig()
    findings = analyze_prompt(prompt, counter=counter, config=rule_config)
    code_findings = analyze_code(code, counter=counter, config=rule_config)
    findings = tuple(
        sorted(
            (*findings, *code_findings),
            key=lambda finding: (finding.location or "", finding.rule_id),
        )
    )
    message_tokens = sum(counter.count_message(message) for message in prompt.messages)
    tool_tokens = sum(counter.count_tool(tool) for tool in prompt.tools)
    prompt_tokens = message_tokens + tool_tokens
    code_tokens = counter.count_text(code.source)
    simulation = simulate_conversation(
        prompt,
        simulation_config,
        counter=counter,
        pricing=pricing,
    )
    estimated_savings = sum(finding.estimated_token_savings for finding in findings)
    optimized_prompt_tokens = max(prompt_tokens - estimated_savings, 0)
    one_turn_cost = estimate_cost(prompt_tokens, simulation_config.response_tokens, pricing)
    optimized_one_turn_cost = estimate_cost(
        optimized_prompt_tokens,
        simulation_config.response_tokens,
        pricing,
    )

    return AuditResult(
        code=code,
        prompt=prompt,
        counter_backend=counter.backend,
        code_tokens=code_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        prompt_tokens=prompt_tokens,
        findings=findings,
        simulation=simulation,
        pricing=pricing,
        estimated_savings_tokens=estimated_savings,
        optimized_prompt_tokens=optimized_prompt_tokens,
        one_turn_cost=one_turn_cost,
        optimized_one_turn_cost=optimized_one_turn_cost,
    )


def _read_argument(raw: str | None, path: Path | None, label: str) -> str:
    if raw is not None and path is not None:
        raise ValueError(f"choose either --{label} or --{label}-file")
    if path is not None:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read {label} file: {exc}") from exc
    return raw or ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic Agent Token Optimizer audit.")
    parser.add_argument("--code", help="agent source code")
    parser.add_argument("--code-file", type=Path, help="UTF-8 agent source file")
    parser.add_argument("--language", help="explicit code language, otherwise infer from the file")
    parser.add_argument("--prompt", help="plain text or JSON prompt")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument('--tools', help='JSON tool object, array, or {"tools": [...]} wrapper')
    parser.add_argument("--tools-file", type=Path, help="UTF-8 JSON tool schema file")
    parser.add_argument("--input-per-1k", type=float, default=0.15)
    parser.add_argument("--output-per-1k", type=float, default=0.60)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--response-tokens", type=int, default=256)
    parser.add_argument("--follow-up-tokens", type=int, default=32)
    parser.add_argument("--verbose-message-tokens", type=int, default=200)
    parser.add_argument("--verbose-description-tokens", type=int, default=80)
    parser.add_argument("--bloated-tool-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_audit(
            code_source=_read_argument(args.code, args.code_file, "code"),
            code_path=str(args.code_file) if args.code_file else None,
            language=args.language,
            prompt_input=_read_argument(args.prompt, args.prompt_file, "prompt"),
            tools_input=_read_argument(args.tools, args.tools_file, "tools"),
            pricing=Pricing(input_per_1k=args.input_per_1k, output_per_1k=args.output_per_1k),
            simulation_config=SimulationConfig(
                turns=args.turns,
                response_tokens=args.response_tokens,
                follow_up_tokens=args.follow_up_tokens,
            ),
            rule_config=RuleConfig(
                verbose_message_tokens=args.verbose_message_tokens,
                verbose_description_tokens=args.verbose_description_tokens,
                bloated_tool_tokens=args.bloated_tool_tokens,
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
