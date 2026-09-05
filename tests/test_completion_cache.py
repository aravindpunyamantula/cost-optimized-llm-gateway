from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_completion_cache_miss_then_hit(monkeypatch):
    import app.api.endpoints as endpoints

    class FakeCache:
        stored = None

        async def generate_embedding(self, text):
            return [1.0] + [0.0] * 1535

        async def search_cache(self, embedding):
            if FakeCache.stored is None:
                return None

            return FakeCache.stored

        async def store_cache(
            self,
            prompt,
            embedding,
            response,
            model_used,
            cost_usd,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ):
            FakeCache.stored = {
                "response": response,
                "model_used": model_used,
                "cost_usd": cost_usd,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "similarity": 1.0,
            }

        async def close(self):
            pass

    FakeCache.stored = None

    async def mock_call_llm(prompt, model, max_tokens=None):
        return {
            "response": "Mock cached response",
            "model_used": model,
            "cost_usd": 0.002,
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
        }

    async def mock_create_request_log(
        db,
        *,
        user_id,
        model_used,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost_usd,
        cache_hit,
    ):
        return (
            "11111111-1111-1111-1111-111111111111"
        )

    monkeypatch.setattr(
        endpoints,
        "SemanticCache",
        FakeCache,
    )
    monkeypatch.setattr(
        endpoints,
        "call_llm",
        mock_call_llm,
    )
    monkeypatch.setattr(
        endpoints,
        "create_request_log",
        mock_create_request_log,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:

        # First request → cache MISS
        first_response = await client.post(
            "/v1/completions",
            json={
                "prompt": "Explain machine learning",
                "user_id": "cache-test-user",
            },
        )

        assert first_response.status_code == 200

        first_data = first_response.json()

        assert first_data["response"] == "Mock cached response"
        assert first_data["cache_hit"] is False
        assert first_data["cost_usd"] == 0.002
        assert first_response.headers.get("X-Request-ID")

        # Second request → cache HIT
        second_response = await client.post(
            "/v1/completions",
            json={
                "prompt": "Explain machine learning",
                "user_id": "cache-test-user",
            },
        )

        assert second_response.status_code == 200

        second_data = second_response.json()

        assert second_data["response"] == "Mock cached response"
        assert second_data["cache_hit"] is True
        assert second_data["cost_usd"] == 0.0
        assert second_response.headers.get("X-Request-ID")