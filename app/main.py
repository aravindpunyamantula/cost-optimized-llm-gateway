from fastapi import FastAPI

app = FastAPI(
    title="LLM-Gateway",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {
        "Status": "OK",
        "Service": "LLM-Gateway",
    }