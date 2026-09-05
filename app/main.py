from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(
    title="LLM-Gateway",
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {
        "Status": "OK",
        "Service": "LLM-Gateway",
    }

