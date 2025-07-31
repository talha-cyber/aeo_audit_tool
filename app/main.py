from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import audits

app = FastAPI(
    title="AEO Competitive Intelligence Tool",
    description=(
        "Multi-platform AEO audit tool that simulates user questions "
        "across AI platforms"
    ),
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audits.router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    return {"status": "ok"}
