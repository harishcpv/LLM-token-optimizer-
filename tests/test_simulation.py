import pytest

from agent_token_optimizer import (
    Pricing,
    SimulationConfig,
    TokenCounter,
    estimate_cost,
    parse_prompt,
    predict_cost,
    simulate_conversation,
)


def test_simulation_accounts_for_history_growth_and_cost() -> None:
    prompt = parse_prompt("Explain the result.")
    counter = TokenCounter(message_overhead=0)
    pricing = Pricing(input_per_1k=2.0, output_per_1k=4.0)
    base_tokens = counter.count_prompt(prompt)

    result = simulate_conversation(
        prompt,
        SimulationConfig(turns=3, response_tokens=10, follow_up_tokens=3),
        counter=counter,
        pricing=pricing,
    )

    assert [turn.input_tokens for turn in result.turns] == [base_tokens, base_tokens + 13, base_tokens + 26]
    assert result.total_output_tokens == 30
    assert result.total_input_tokens == base_tokens * 3 + 39
    assert result.total_cost == pytest.approx(
        sum(estimate_cost(turn.input_tokens, 10, pricing) for turn in result.turns)
    )


def test_predict_cost_matches_single_turn_simulation() -> None:
    prompt = parse_prompt("Classify this text.")
    counter = TokenCounter()
    pricing = Pricing(input_per_1k=1.5, output_per_1k=3.0)

    result = simulate_conversation(
        prompt,
        SimulationConfig(turns=1, response_tokens=20, follow_up_tokens=0),
        counter=counter,
        pricing=pricing,
    )
    assert predict_cost(prompt, 20, pricing, counter=counter) == pytest.approx(result.total_cost)


def test_simulation_config_rejects_zero_turns() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SimulationConfig(turns=0)
