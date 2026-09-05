from app.core.router import estimate_tokens, route_prompt


def test_default_route():
    prompt = "Hello, how are you?"

    result = route_prompt(prompt)

    assert result == "groq/llama3-8b-8192"


def test_complex_keyword_routes_to_high_tier():
    prompt = "Analyze this system architecture."

    result = route_prompt(prompt)

    assert result == "gpt-4o"


def test_critique_keyword_routes_to_high_tier():
    prompt = "Critique this solution."

    result = route_prompt(prompt)

    assert result == "gpt-4o"


def test_synthesize_keyword_routes_to_high_tier():
    prompt = "Synthesize these research findings."

    result = route_prompt(prompt)

    assert result == "gpt-4o"


def test_keyword_matching_is_case_insensitive():
    prompt = "ANALYZE this problem."

    result = route_prompt(prompt)

    assert result == "gpt-4o"


def test_token_estimation_returns_positive_value():
    prompt = "Explain how an API gateway works."

    result = estimate_tokens(prompt)

    assert result > 0