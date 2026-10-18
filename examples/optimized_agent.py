"""A compact comparison request builder for the inefficient fixture."""


SYSTEM_PROMPT = "You are a catalog assistant. Return concise, factual matches."


def build_request(user_text: str, request_id: str | None = None) -> dict[str, object]:
    """Keep the stable prefix cacheable and pass only current request data."""

    context = user_text if request_id is None else f"request_id={request_id}; {user_text}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
    }
