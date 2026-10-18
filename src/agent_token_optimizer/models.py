"""Validated, dependency-free data types used by the optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence


VALID_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message in the normalized prompt representation."""

    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_MESSAGE_ROLES:
            valid = ", ".join(sorted(VALID_MESSAGE_ROLES))
            raise ValueError(f"unsupported message role {self.role!r}; expected one of {valid}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.name is not None and not self.name.strip():
            raise ValueError("message name cannot be blank")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A callable tool and its JSON-Schema-like parameter definition."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(
                "tool name must start with a letter and contain only letters, numbers, "
                "underscores, or hyphens (maximum 64 characters)"
            )
        if not isinstance(self.description, str):
            raise TypeError("tool description must be a string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a mapping")
        # Copy the outer mapping so a caller cannot alter the definition by mutating
        # the dictionary that was passed to the constructor.
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Normalized prompt messages and optional tools."""

    messages: Sequence[Message] = field(default_factory=tuple)
    tools: Sequence[ToolDefinition] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        if not all(isinstance(message, Message) for message in messages):
            raise TypeError("prompt messages must contain Message instances")
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("prompt tools must contain ToolDefinition instances")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("prompt tools must have unique names")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)


@dataclass(frozen=True, slots=True)
class CodeArtifact:
    """Source code normalized from plain text or a fenced code block."""

    source: str
    language: str = "text"
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("code source must be a string")
        if not self.language.strip():
            raise ValueError("code language cannot be blank")

    @property
    def line_count(self) -> int:
        """Return the number of source lines, with empty source counting as zero."""

        return 0 if not self.source else self.source.count("\n") + 1


class Severity(str, Enum):
    """Finding severity used by optimization rules."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class OptimizationFinding:
    """A deterministic, actionable optimization rule result."""

    rule_id: str
    severity: Severity
    title: str
    description: str
    recommendation: str
    estimated_token_savings: int = 0
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("finding rule_id cannot be blank")
        if self.estimated_token_savings < 0:
            raise ValueError("estimated token savings cannot be negative")


def _finite_nonnegative(value: float, field_name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite, non-negative number")
    return float(value)


@dataclass(frozen=True, slots=True)
class Pricing:
    """Model pricing in currency units per 1,000 input/output tokens."""

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_per_1k", _finite_nonnegative(self.input_per_1k, "input_per_1k"))
        object.__setattr__(self, "output_per_1k", _finite_nonnegative(self.output_per_1k, "output_per_1k"))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Deterministic assumptions for a multi-turn cost simulation."""

    turns: int = 1
    response_tokens: int = 256
    follow_up_tokens: int = 32
    tool_calls_per_turn: int = 0
    tool_result_tokens: int = 0
    tool_call_overhead_tokens: int = 8

    def __post_init__(self) -> None:
        for field_name in (
            "turns",
            "response_tokens",
            "follow_up_tokens",
            "tool_calls_per_turn",
            "tool_result_tokens",
            "tool_call_overhead_tokens",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.turns == 0:
            raise ValueError("turns must be at least 1")


@dataclass(frozen=True, slots=True)
class SimulationTurn:
    """Token and cost totals for one simulated request."""

    turn_number: int
    input_tokens: int
    output_tokens: int
    cost: float
    tool_overhead_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete simulation result, including per-turn audit data."""

    turns: tuple[SimulationTurn, ...]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_tool_overhead_tokens(self) -> int:
        """Return tool-message tokens attributed to all simulated turns."""

        return sum(turn.tool_overhead_tokens for turn in self.turns)
