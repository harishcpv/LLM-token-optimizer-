"""A deliberately inefficient, provider-neutral agent request builder.

This fixture does not call a model or a network. It gives the audit CLI a small
source example with a dynamic prompt prefix and duplicated conversation history.
"""

from datetime import datetime


def build_request(user_text: str, history: list[str]) -> dict[str, object]:
    """Build a request with a few intentionally expensive token choices."""

    system_prompt = f"""
You are a helpful catalog assistant. Always be accurate, polite, thorough, and
explicit. Explain your reasoning, restate the customer's request, describe all
available alternatives, mention every relevant policy, and include a complete
summary even when the answer is simple. The same instructions must be followed
for every request. This request was generated at {datetime.now().isoformat()}.
""".strip()

    repeated_history = "\n".join(history + history)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{repeated_history}\n{user_text}"},
        ]
    }
