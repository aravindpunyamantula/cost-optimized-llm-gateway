import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_completion_creates_telemetry(monkeypatch):
    import app.api.endpoints as endpoints

    class FakeCache:
        async def generate_embedding(self, text):
            return [1.0] + [0.0] * 1535

        async def search_cache(self, embedding):
            return None

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
            pass

        async def close(self):
            pass

    async def mock_call_llm(
        prompt: str,
        model: str,
        max_tokens: int | None = None,
    ):
        return {
            "response": "Mock LLM response",
            "model_used": model,
            "cost_usd": 0.001,
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
        return "11111111-1111-1111-1111-111111111111"

    monkeypatch.setattr(endpoints, "SemanticCache", FakeCache)
    monkeypatch.setattr(endpoints, "call_llm", mock_call_llm)
    monkeypatch.setattr(
        endpoints,
        "create_request_log",
        mock_create_request_log,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/completions",
            json={
                "prompt": "Explain machine learning",
                "user_id": "test-user",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["response"] == "Mock LLM response"
    assert data["cache_hit"] is False
    assert data["cost_usd"] == 0.001
    assert data["model_used"]
    assert response.headers.get("X-Request-ID")