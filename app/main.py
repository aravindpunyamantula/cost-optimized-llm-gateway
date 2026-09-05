from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.endpoints import router
from app.db.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    yield

    await close_db()


app = FastAPI(
    title="Cost-Optimized LLM Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "llm-gateway",
    }