import pytest

from app.core.cache import SemanticCache


@pytest.mark.asyncio
async def test_store_and_search_cache():
    cache = SemanticCache()

    await cache.redis.flushdb()
    await cache.create_index()

    embedding = [1.0] + [0.0] * 1535

    await cache.store_cache(
        prompt="What is machine learning?",
        embedding=embedding,
        response="Machine learning is a method of learning from data.",
        model_used="test-model",
        cost_usd=0.001,
        prompt_tokens=5,
        completion_tokens=8,
        total_tokens=13,
    )

    result = await cache.search_cache(embedding)

    assert result is not None
    assert result["response"] == (
        "Machine learning is a method of learning from data."
    )
    assert result["model_used"] == "test-model"
    assert result["cost_usd"] == 0.001
    assert result["prompt_tokens"] == 5
    assert result["completion_tokens"] == 8
    assert result["total_tokens"] == 13
    assert result["similarity"] > 0.95

    await cache.close()


@pytest.mark.asyncio
async def test_cache_miss_below_similarity_threshold():
    cache = SemanticCache()

    await cache.redis.flushdb()
    await cache.create_index()

    stored_embedding = [1.0] + [0.0] * 1535

    query_embedding = [0.0, 1.0] + [0.0] * 1534

    await cache.store_cache(
        prompt="What is machine learning?",
        embedding=stored_embedding,
        response="Cached response",
        model_used="test-model",
        cost_usd=0.001,
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
    )

    result = await cache.search_cache(query_embedding)

    assert result is None

    await cache.close()