"""Gushwork Inbound AEO Voice Agent — FastAPI Application.

This is the central nervous system.  Every webhook, tool endpoint, and
health check routes through here.

Startup sequence:
1. Load .env configuration
2. Mount route modules
3. Configure CORS for the web widget
4. Initialize logging
5. Start Uvicorn on SERVER_HOST:SERVER_PORT
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.retell import router as retell_router
from app.routes.tools import router as tools_router
from app.routes.widget import router as widget_router

# ── Logging ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gushwork")

# ── App ────────────────────────────────────────────────────

app = FastAPI(
    title="Gushwork SDR Voice Agent",
    description=(
        "Inbound AEO Voice Agent — sub-500ms latency pipeline from "
        "WebRTC ↔ Deepgram STT ↔ LangGraph ↔ Gemini 3 Flash ↔ ElevenLabs TTS"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (for web widget) ─────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Timing Middleware ──────────────────────────────


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log request latency for every endpoint — critical for sub-500ms SLA."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({elapsed_ms:.1f}ms)"
    )
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ── Routes ─────────────────────────────────────────────────

app.include_router(retell_router)
app.include_router(tools_router)
app.include_router(widget_router)


# ── Health Check ───────────────────────────────────────────


@app.get("/health")
async def health():
    """Liveness probe for load balancers and monitoring."""
    return {
        "status": "healthy",
        "service": "gushwork-sdr",
        "version": "1.0.0",
        "uptime_check": True,
    }


@app.get("/")
async def root():
    """Root redirect to docs."""
    return {
        "service": "Gushwork Inbound AEO Voice Agent",
        "docs": "/docs",
        "health": "/health",
    }


# ── Startup / Shutdown ─────────────────────────────────────


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("  Gushwork SDR Voice Agent — ONLINE")
    logger.info(f"  Model: {settings.gemini_model}")
    logger.info(f"  Temperature: {settings.llm_temperature}")
    logger.info(f"  Max Tokens: {settings.llm_max_tokens}")
    logger.info(f"  VAD Silence: {settings.vad_silence_ms}ms")
    logger.info(f"  CRM: {settings.crm_provider}")
    logger.info(f"  Webhook URL: {settings.webhook_base_url}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Gushwork SDR Voice Agent — SHUTTING DOWN")


# ── Entrypoint ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info",
    )
