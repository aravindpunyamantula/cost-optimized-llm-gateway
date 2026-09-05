import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RequestLog


async def create_request_log(
    session: AsyncSession,
    *,
    user_id: str,
    model_used: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    cost_usd: float,
    cache_hit: bool,
) -> uuid.UUID:
    """
    Store telemetry for one successful completion request.
    """

    request_log = RequestLog(
        user_id=user_id,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        timestamp=datetime.now(timezone.utc),
    )

    session.add(request_log)

    await session.commit()
    await session.refresh(request_log)

    return request_log.id