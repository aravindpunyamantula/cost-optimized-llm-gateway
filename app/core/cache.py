import hashlib
import struct

import redis.asyncio as redis
from redis.exceptions import ResponseError

from app.config import REDIS_HOST, REDIS_PORT, config


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
        Generate an embedding using the configured LiteLLM embedding model.
        """
        import litellm

        response = await litellm.aembedding(
            model=config["models"]["embedding"],
            input=[text],
        )

        return response.data[0]["embedding"]

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        """
        Convert a float embedding into Redis FLOAT32 binary format.
        """
        if len(embedding) != VECTOR_DIMENSION:
            raise ValueError(
                f"Expected embedding dimension {VECTOR_DIMENSION}, "
                f"got {len(embedding)}"
            )

        return struct.pack(
            f"{VECTOR_DIMENSION}f",
            *embedding,
        )

    @staticmethod
    def _cache_key(prompt: str) -> str:
        """
        Generate a stable Redis key from the prompt.
        """
        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        return f"{CACHE_PREFIX}{prompt_hash}"

    async def store_cache(
        self,
        prompt: str,
        embedding: list[float],
        response: str,
        model_used: str,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """
        Store an LLM response, token usage, cost, and embedding in Redis.
        """
        cache_key = self._cache_key(prompt)

        embedding_bytes = self._serialize_embedding(embedding)

        await self.redis.hset(
            cache_key,
            mapping={
                "prompt": prompt,
                "response": response,
                "model_used": model_used,
                "cost_usd": str(cost_usd),
                "prompt_tokens": str(prompt_tokens),
                "completion_tokens": str(completion_tokens),
                "total_tokens": str(total_tokens),
                "embedding": embedding_bytes,
            },
        )

        ttl = int(config["caching"]["ttl"])

        await self.redis.expire(
            cache_key,
            ttl,
        )

    async def search_cache(
        self,
        embedding: list[float],
    ) -> dict | None:
        """
        Search Redis for the most semantically similar cached response.

        Redis COSINE distance:
            0.0 = identical
            1.0 = increasingly different

        Similarity:
            similarity = 1 - cosine_distance

        A cache hit requires similarity to be greater than
        the configured threshold.
        """
        embedding_bytes = self._serialize_embedding(embedding)

        query = (
            "*=>[KNN 1 @embedding $query_vector "
            "AS vector_score]"
        )

        try:
            result = await self.redis.execute_command(
                "FT.SEARCH",
                CACHE_INDEX_NAME,
                query,
                "PARAMS",
                "2",
                "query_vector",
                embedding_bytes,
                "SORTBY",
                "vector_score",
                "ASC",
                "RETURN",
                "8",
                "prompt",
                "response",
                "model_used",
                "cost_usd",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "vector_score",
                "DIALECT",
                "2",
            )
        except ResponseError:
            return None

        if not result:
            return None

        # redis-py returns FT.SEARCH results as a dictionary.
        results = result.get(b"results", [])

        if not results:
            return None

        document = results[0]

        extra_attributes = document.get(
            b"extra_attributes",
            {},
        )

        if not extra_attributes:
            return None

        distance = extra_attributes.get(b"vector_score")

        if distance is None:
            return None

        if isinstance(distance, bytes):
            distance = distance.decode("utf-8")

        distance = float(distance)

        similarity = 1.0 - distance

        threshold = float(
            config["caching"]["similarity_threshold"]
        )

        if similarity <= threshold:
            return None

        response = extra_attributes.get(b"response")
        model_used = extra_attributes.get(b"model_used")
        cost_usd = extra_attributes.get(b"cost_usd")
        prompt_tokens = extra_attributes.get(b"prompt_tokens")
        completion_tokens = extra_attributes.get(b"completion_tokens")
        total_tokens = extra_attributes.get(b"total_tokens")

        if (
            response is None
            or model_used is None
            or cost_usd is None
            or prompt_tokens is None
            or completion_tokens is None
            or total_tokens is None
        ):
            return None

        if isinstance(response, bytes):
            response = response.decode("utf-8")

        if isinstance(model_used, bytes):
            model_used = model_used.decode("utf-8")

        if isinstance(cost_usd, bytes):
            cost_usd = cost_usd.decode("utf-8")

        if isinstance(prompt_tokens, bytes):
            prompt_tokens = prompt_tokens.decode("utf-8")

        if isinstance(completion_tokens, bytes):
            completion_tokens = completion_tokens.decode("utf-8")

        if isinstance(total_tokens, bytes):
            total_tokens = total_tokens.decode("utf-8")

        return {
            "response": response,
            "model_used": model_used,
            "cost_usd": float(cost_usd),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "similarity": similarity,
        }