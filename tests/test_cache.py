import pytest
from unittest.mock import AsyncMock, patch

from app.core.cache import (
    CACHE_INDEX_NAME,
    SemanticCache,
)


@pytest.mark.asyncio
async def test_redis_connection():
    cache = SemanticCache()

    try:
        result = await cache.ping()
        assert result is True
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_create_vector_index():
    cache = SemanticCache()

    try:
        await cache.create_index()

        indexes = await cache.redis.execute_command(
            "FT._LIST"
        )

        assert CACHE_INDEX_NAME.encode() in indexes

    finally:
        await cache.close()
@pytest.mark.asyncio
async def test_generate_embedding():
    fake_embedding = [0.1] * 1536

    mock_response = type(
        "MockResponse",
        (),
        {
            "data": [
                {
                    "embedding": fake_embedding,
                }
            ]
        },
    )()

    cache = SemanticCache()

    try:
        with patch(
            "litellm.aembedding",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_embedding:

            result = await cache.generate_embedding(
                "Hello world"
            )

        mock_embedding.assert_awaited_once()

        assert len(result) == 1536
        assert result == fake_embedding

    finally:
        await cache.close()