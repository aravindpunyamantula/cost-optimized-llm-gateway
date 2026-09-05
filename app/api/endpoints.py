import time

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import CompletionRequest, CompletionResponse
from app.core.cache import SemanticCache
from app.core.orchestrator import call_llm
from app.core.preprocessing import preprocess_prompt
from app.core.router import route_prompt
from app.db.database import get_db
from app.db.respository import create_request_log


router = APIRouter()


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> CompletionResponse:

    start_time = time.perf_counter()

    # 1. Preprocess the prompt
    prompt = preprocess_prompt(request.prompt)

    # 2. Generate embedding and search semantic cache
    cache = SemanticCache()

    try:
        embedding = await cache.generate_embedding(prompt)
        cached_result = await cache.search_cache(embedding)

        # 3. Cache HIT
        if cached_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000

            request_id = await create_request_log(
                db,
                user_id=request.user_id,
                model_used=cached_result["model_used"],
                prompt_tokens=cached_result["prompt_tokens"],
                completion_tokens=cached_result["completion_tokens"],
                total_tokens=cached_result["total_tokens"],
                latency_ms=latency_ms,
                cost_usd=0.0,
                cache_hit=True,
            )

            response.headers["X-Request-ID"] = str(request_id)

            return CompletionResponse(
                response=cached_result["response"],
                model_used=cached_result["model_used"],
                cache_hit=True,
                cost_usd=0.0,
            )

        # 4. Cache MISS → route to the appropriate model
        model = route_prompt(prompt)

        result = await call_llm(
            prompt=prompt,
            model=model,
            max_tokens=request.max_tokens,
        )

        # 5. Store successful LLM response in semantic cache
        await cache.store_cache(
            prompt=prompt,
            embedding=embedding,
            response=result["response"],
            model_used=result["model_used"],
            cost_usd=result["cost_usd"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 6. Store telemetry in PostgreSQL
        request_id = await create_request_log(
            db,
            user_id=request.user_id,
            model_used=result["model_used"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
            latency_ms=latency_ms,
            cost_usd=result["cost_usd"],
            cache_hit=False,
        )

        response.headers["X-Request-ID"] = str(request_id)

        return CompletionResponse(
            response=result["response"],
            model_used=result["model_used"],
            cache_hit=False,
            cost_usd=result["cost_usd"],
        )

    finally:
        await cache.close()