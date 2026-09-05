from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_completion_creates_telemetry(monkeypatch):
    import app.api.endpoints as endpoints

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

    monkeypatch.setattr(
        endpoints,
        "call_llm",
        mock_call_llm,
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

    request_id = response.headers["X-Request-ID"]

    assert len(request_id) == 36