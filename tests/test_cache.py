import pytest

from app.core.cache import SemanticCache


@pytest.mark.asyncio
async def test_store_and_search_cache():
    cache = SemanticCache()

    await cache.create_index()

    embedding = [1.0] + [0.0] * 1535

    await cache.store_cache(
        prompt="What is machine learning?",
        embedding=embedding,
        response="Machine learning is a method of learning from data.",
        model_used="test-model",
        cost_usd=0.001,
    )

    result = await cache.search_cache(embedding)

    assert result is not None
    assert result["response"] == (
        "Machine learning is a method of learning from data."
    )
    assert result["model_used"] == "test-model"
    assert result["cost_usd"] == pytest.approx(0.001)
    assert result["similarity"] > 0.95

    await cache.close()


@pytest.mark.asyncio
async def test_search_cache_miss():
    cache = SemanticCache()

    await cache.create_index()

    stored_embedding = [1.0] + [0.0] * 1535
    different_embedding = [0.0, 1.0] + [0.0] * 1534

    await cache.store_cache(
        prompt="Cached prompt",
        embedding=stored_embedding,
        response="Cached response",
        model_used="test-model",
        cost_usd=0.001,
    )

    result = await cache.search_cache(different_embedding)

    assert result is None

    await cache.close()