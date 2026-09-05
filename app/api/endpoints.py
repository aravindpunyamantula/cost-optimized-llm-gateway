from fastapi import APIRouter

from app.api.schemas import CompletionRequest, CompletionResponse

router = APIRouter()

@router.post("/v1/completions", response_model=CompletionResponse)

async def create_completion(request: CompletionRequest):
    response_text = f"Echo: {request.prompt}"
    model_used = "mock-model"
    cache_hit = False
    cost_usd = 0.0 

    return CompletionResponse(
        response=response_text,
        model_used=model_used,
        cache_hit=cache_hit,
        cost_usd=cost_usd
    )