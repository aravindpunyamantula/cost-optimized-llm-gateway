import logging
from typing import Any

import litellm
from litellm import APIConnectionError, RateLimitError

from app.config import config

logger = logging.getLogger(__name__)


def get_fallback_model(primary_model: str) -> str | None:
    for fallback in config["fallbacks"]:
        if fallback["primary"] == primary_model:
            return fallback["backup"]
    return None


def _extract_usage(response: Any) -> dict[str, int]:
    """
    Extract token usage from a LiteLLM response.
    """

    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = int(
        getattr(usage, "prompt_tokens", 0) or 0
    )

    completion_tokens = int(
        getattr(usage, "completion_tokens", 0) or 0
    )

    total_tokens = int(
        getattr(usage, "total_tokens", 0)
        or prompt_tokens + completion_tokens
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def _complete(
    prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Execute one LiteLLM completion and extract cost and usage.
    """

    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )

    cost = litellm.completion_cost(
        completion_response=response
    )

    usage = _extract_usage(response)

    return {
        "response": response.choices[0].message.content,
        "model_used": model,
        "cost_usd": float(cost or 0.0),
        **usage,
    }


async def call_llm(
    prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:

    try:
        return await _complete(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
        )

    except (RateLimitError, APIConnectionError) as exc:

        fallback_model = get_fallback_model(model)

        if not fallback_model:
            logger.error(
                "LLM request failed for %s and no fallback "
                "is configured: %s",
                model,
                exc,
            )
            raise

        logger.warning(
            "LLM request failed for %s. Retrying with "
            "fallback %s. Error: %s",
            model,
            fallback_model,
            exc,
        )

        return await _complete(
            prompt=prompt,
            model=fallback_model,
            max_tokens=max_tokens,
        )