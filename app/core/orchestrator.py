import logging
from typing import Any

import litellm
from litellm import APIConnectionError, RateLimitError

from app.config import config


logger = logging.getLogger(__name__)


def get_fallback_model(primary_model: str) -> str | None:
    """
    Return the configured fallback model for a primary model.
    """

    for fallback in config["fallbacks"]:
        if fallback["primary"] == primary_model:
            return fallback["backup"]

    return None


async def call_llm(
    prompt: str,
    model: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Call an LLM through LiteLLM.

    If the primary model encounters a rate limit or connection error,
    retry once using its configured fallback model.
    """

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
        )

        cost = litellm.completion_cost(completion_response=response)

        return {
            "response": response.choices[0].message.content,
            "model_used": model,
            "cost_usd": float(cost or 0.0),
        }

    except (RateLimitError, APIConnectionError) as exc:
        fallback_model = get_fallback_model(model)

        if not fallback_model:
            logger.error(
                "LLM request failed for %s and no fallback is configured: %s",
                model,
                exc,
            )
            raise

        logger.warning(
            "LLM request failed for %s. Retrying with fallback %s. Error: %s",
            model,
            fallback_model,
            exc,
        )

        response = await litellm.acompletion(
            model=fallback_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
        )

        cost = litellm.completion_cost(completion_response=response)

        return {
            "response": response.choices[0].message.content,
            "model_used": fallback_model,
            "cost_usd": float(cost or 0.0),
        }