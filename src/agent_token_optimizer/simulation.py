"""Deterministic conversation simulation and cost prediction."""

from __future__ import annotations

from .models import Pricing, PromptSpec, SimulationConfig, SimulationResult, SimulationTurn
from .token_counting import TokenCounter


def estimate_cost(input_tokens: int, output_tokens: int, pricing: Pricing) -> float:
    """Estimate one request cost from token counts and per-1K rates."""

    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
        raise ValueError("input_tokens must be a non-negative integer")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
        raise ValueError("output_tokens must be a non-negative integer")
    if not isinstance(pricing, Pricing):
        raise TypeError("pricing must be a Pricing instance")
    return (input_tokens * pricing.input_per_1k + output_tokens * pricing.output_per_1k) / 1000


def simulate_conversation(
    prompt: PromptSpec,
    config: SimulationConfig | None = None,
    *,
    counter: TokenCounter | None = None,
    pricing: Pricing | None = None,
) -> SimulationResult:
    """Simulate repeated requests as history grows with responses and follow-ups.

    The initial normalized prompt is sent on every turn. After each non-final
    turn, the next request also includes the assumed assistant response and a
    fixed-size follow-up message. This makes the growth and cost assumptions
    explicit and reproducible without calling a model.
    """

    if not isinstance(prompt, PromptSpec):
        raise TypeError("prompt must be a PromptSpec")
    config = config or SimulationConfig()
    counter = counter or TokenCounter()
    pricing = pricing or Pricing()

    initial_tokens = counter.count_prompt(prompt)
    current_input_tokens = initial_tokens
    turns: list[SimulationTurn] = []
    for turn_number in range(1, config.turns + 1):
        cost = estimate_cost(current_input_tokens, config.response_tokens, pricing)
        turns.append(
            SimulationTurn(
                turn_number=turn_number,
                input_tokens=current_input_tokens,
                output_tokens=config.response_tokens,
                cost=cost,
            )
        )
        if turn_number < config.turns:
            current_input_tokens += config.response_tokens + config.follow_up_tokens

    return SimulationResult(
        turns=tuple(turns),
        total_input_tokens=sum(turn.input_tokens for turn in turns),
        total_output_tokens=sum(turn.output_tokens for turn in turns),
        total_cost=sum(turn.cost for turn in turns),
    )


def predict_cost(
    prompt: PromptSpec,
    expected_output_tokens: int,
    pricing: Pricing,
    *,
    counter: TokenCounter | None = None,
) -> float:
    """Convenience helper for the cost of one prompt/response exchange."""

    result = simulate_conversation(
        prompt,
        SimulationConfig(turns=1, response_tokens=expected_output_tokens, follow_up_tokens=0),
        counter=counter,
        pricing=pricing,
    )
    return result.total_cost
