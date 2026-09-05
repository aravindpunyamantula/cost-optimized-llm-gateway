from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm import APIConnectionError, RateLimitError

from app.core.orchestrator import call_llm, get_fallback_model


def make_mock_response(content: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content
                )
            )
        ]
    )


def test_get_fallback_model():
    assert (
        get_fallback_model("gpt-4o")
        == "claude-3-5-sonnet-20240620"
    )

    assert (
        get_fallback_model("claude-3-5-sonnet-20240620")
        == "groq/llama3-8b-8192"
    )

    assert (
        get_fallback_model("groq/llama3-8b-8192")
        == "gpt-4o"
    )


@pytest.mark.asyncio
async def test_successful_llm_call():
    mock_response = make_mock_response(
        "Hello from the LLM"
    )

    with patch(
        "app.core.orchestrator.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_completion, patch(
        "app.core.orchestrator.litellm.completion_cost",
        return_value=0.001,
    ):
        result = await call_llm(
            prompt="Hello",
            model="gpt-4o",
        )

    mock_completion.assert_awaited_once()

    assert result["response"] == "Hello from the LLM"
    assert result["model_used"] == "gpt-4o"
    assert result["cost_usd"] == 0.001


@pytest.mark.asyncio
async def test_rate_limit_uses_fallback():
    fallback_response = make_mock_response(
        "Fallback response"
    )

    rate_limit_error = RateLimitError(
        "Rate limit",
        "openai",
        "gpt-4o",
    )

    with patch(
        "app.core.orchestrator.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=[
            rate_limit_error,
            fallback_response,
        ],
    ) as mock_completion, patch(
        "app.core.orchestrator.litellm.completion_cost",
        return_value=0.002,
    ):
        result = await call_llm(
            prompt="Hello",
            model="gpt-4o",
        )

    assert mock_completion.await_count == 2

    assert result["response"] == "Fallback response"
    assert result["model_used"] == "claude-3-5-sonnet-20240620"
    assert result["cost_usd"] == 0.002


@pytest.mark.asyncio
async def test_api_connection_error_uses_fallback():
    fallback_response = make_mock_response(
        "Fallback connection response"
    )

    connection_error = APIConnectionError(
        "Connection failed",
        "openai",
        "gpt-4o",
    )

    with patch(
        "app.core.orchestrator.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=[
            connection_error,
            fallback_response,
        ],
    ) as mock_completion, patch(
        "app.core.orchestrator.litellm.completion_cost",
        return_value=0.003,
    ):
        result = await call_llm(
            prompt="Hello",
            model="gpt-4o",
        )

    assert mock_completion.await_count == 2

    assert result["response"] == "Fallback connection response"
    assert result["model_used"] == "claude-3-5-sonnet-20240620"
    assert result["cost_usd"] == 0.003