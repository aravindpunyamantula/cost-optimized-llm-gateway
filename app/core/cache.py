import redis.asyncio as redis
from redis.exceptions import ResponseError

from app.config import (
    REDIS_HOST,
    REDIS_PORT,
    config,
)


CACHE_INDEX_NAME = "llm_cache_idx"
CACHE_PREFIX = "llm_cache:"
VECTOR_DIMENSION = 1536


class SemanticCache:
    def __init__(self) -> None:
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            decode_responses=False,
        )

    async def ping(self) -> bool:
        return await self.redis.ping()

    async def close(self) -> None:
        await self.redis.aclose()

    async def create_index(self) -> None:
        """
        Create the Redis vector search index for semantic caching.
        """

        try:
            await self.redis.execute_command(
                "FT.CREATE",
                CACHE_INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                CACHE_PREFIX,
                "SCHEMA",
                "prompt",
                "TEXT",
                "response",
                "TEXT",
                "model_used",
                "TEXT",
                "cost_usd",
                "NUMERIC",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(VECTOR_DIMENSION),
                "DISTANCE_METRIC",
                "COSINE",
            )

        except ResponseError as exc:
            if "Index already exists" not in str(exc):
                raise
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text using LiteLLM.
        """

        import litellm

        response = await litellm.aembedding(
            model=config["models"]["embedding"],
            input=[text],
        )

        return response.data[0]["embedding"]