from typing import Any
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length = 1)
    user_id: str = Field(..., min_length=1)
    max_tokens: int  | None = Field(default=None, gt=0)
    metadata: dict[str, Any] | None = None

class CompletionResponse(BaseModel):
    response: str
    model_used: str
    cache_hit: bool
    cost_usd: float