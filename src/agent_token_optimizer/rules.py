"""Small, deterministic optimization rules for normalized prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

from .models import OptimizationFinding, PromptSpec, Severity
from .token_counting import TokenCounter


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Thresholds controlling when verbosity rules produce findings."""

    verbose_message_tokens: int = 200
    verbose_description_tokens: int = 80
    bloated_tool_tokens: int = 256
    detect_dynamic_headers: bool = True

    def __post_init__(self) -> None:
        if (
            self.verbose_message_tokens < 1
            or self.verbose_description_tokens < 1
            or self.bloated_tool_tokens < 1
        ):
            raise ValueError("rule thresholds must be positive")
        if not isinstance(self.detect_dynamic_headers, bool):
            raise TypeError("detect_dynamic_headers must be a boolean")


def find_duplicate_messages(prompt: PromptSpec, counter: TokenCounter) -> list[OptimizationFinding]:
    seen: dict[tuple[str, str, str | None], int] = {}
    findings: list[OptimizationFinding] = []
    for index, message in enumerate(prompt.messages):
        key = (message.role, message.content, message.name)
        if key not in seen:
            seen[key] = index
            continue
        savings = counter.count_message(message)
        findings.append(
            OptimizationFinding(
                rule_id="duplicate-message",
                severity=Severity.WARNING,
                title="Duplicate message",
                description=f"Message {index} repeats message {seen[key]} with the same role and content.",
                recommendation="Keep one copy of the instruction or example.",
                estimated_token_savings=savings,
                location=f"messages[{index}]",
            )
        )
    return findings


def find_verbose_messages(
    prompt: PromptSpec,
    counter: TokenCounter,
    threshold: int,
) -> list[OptimizationFinding]:
    findings: list[OptimizationFinding] = []
    for index, message in enumerate(prompt.messages):
        tokens = counter.count_text(message.content)
        if tokens <= threshold:
            continue
        findings.append(
            OptimizationFinding(
                rule_id="verbose-message",
                severity=Severity.INFO,
                title="Long message",
                description=f"Message {index} contains approximately {tokens} content tokens.",
                recommendation="Remove repeated context and move stable policy into a shorter instruction.",
                estimated_token_savings=tokens - threshold,
                location=f"messages[{index}]",
            )
        )
    return findings


def _description_entries(prompt: PromptSpec) -> Iterator[tuple[str, str]]:
    for tool in prompt.tools:
        if tool.description:
            yield f"tools[{tool.name}].description", tool.description
        properties = tool.parameters.get("properties")
        if not isinstance(properties, Mapping):
            continue
        for property_name, schema in properties.items():
            if isinstance(schema, Mapping) and isinstance(schema.get("description"), str):
                yield f"tools[{tool.name}].parameters.properties[{property_name!r}].description", schema["description"]


def find_verbose_descriptions(
    prompt: PromptSpec,
    counter: TokenCounter,
    threshold: int,
) -> list[OptimizationFinding]:
    findings: list[OptimizationFinding] = []
    for location, description in _description_entries(prompt):
        tokens = counter.count_text(description)
        if tokens <= threshold:
            continue
        findings.append(
            OptimizationFinding(
                rule_id="verbose-tool-description",
                severity=Severity.INFO,
                title="Verbose tool description",
                description=f"{location} contains approximately {tokens} tokens.",
                recommendation="Describe the tool or parameter in one concise sentence.",
                estimated_token_savings=tokens - threshold,
                location=location,
            )
        )
    return findings


def find_bloated_tools(
    prompt: PromptSpec,
    counter: TokenCounter,
    threshold: int,
) -> list[OptimizationFinding]:
    """Report tool schemas whose serialized definition dominates the prompt."""

    findings: list[OptimizationFinding] = []
    for tool in prompt.tools:
        tokens = counter.count_tool(tool)
        if tokens <= threshold:
            continue
        findings.append(
            OptimizationFinding(
                rule_id="bloated-tool-schema",
                severity=Severity.WARNING,
                title="Bloated tool schema",
                description=(
                    f"Tool {tool.name!r} contributes approximately {tokens} tokens, "
                    f"above the configured {threshold}-token threshold."
                ),
                recommendation=(
                    "Remove redundant properties and examples, shorten descriptions, "
                    "and split rarely used operations into a separate tool or route."
                ),
                estimated_token_savings=tokens - threshold,
                location=f"tools[{tool.name}]",
            )
        )
    return findings


def analyze_prompt(
    prompt: PromptSpec,
    *,
    counter: TokenCounter | None = None,
    config: RuleConfig | None = None,
) -> tuple[OptimizationFinding, ...]:
    """Run all built-in rules and return findings in stable source order."""

    if not isinstance(prompt, PromptSpec):
        raise TypeError("prompt must be a PromptSpec")
    counter = counter or TokenCounter()
    config = config or RuleConfig()
    findings = [
        *find_duplicate_messages(prompt, counter),
        *find_verbose_messages(prompt, counter, config.verbose_message_tokens),
        *find_verbose_descriptions(prompt, counter, config.verbose_description_tokens),
        *find_bloated_tools(prompt, counter, config.bloated_tool_tokens),
    ]
    return tuple(sorted(findings, key=lambda finding: (finding.location or "", finding.rule_id)))
