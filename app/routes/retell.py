"""Retell AI webhook routes.

Supports two modes:
- POST /retell/webhook — Standard JSON response (lifecycle events)
- POST /retell/webhook/stream — SSE streaming response (sub-100ms TTFB)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.agent.retell_handler import (
    handle_retell_event,
    handle_retell_event_stream,
    verify_retell_signature,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retell", tags=["retell"])


@router.post("/webhook")
async def retell_webhook(
    request: Request,
    x_retell_signature: str = Header(default="", alias="x-retell-signature"),
):
    """Standard Retell AI webhook — complete JSON response per turn."""
    body = await request.body()

    if not verify_retell_signature(body, x_retell_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    logger.debug(f"Retell event: {payload.get('event', 'unknown')}")

    response = await handle_retell_event(payload)
    return response


@router.post("/webhook/stream")
async def retell_webhook_stream(
    request: Request,
    x_retell_signature: str = Header(default="", alias="x-retell-signature"),
):
    """SSE Streaming Retell webhook — sub-100ms TTFB (Improvement 1).

    Pipes Gemini tokens directly to Retell's TTS engine as they're generated.
    Configure your Retell agent to point at this endpoint for streaming.
    """
    body = await request.body()

    if not verify_retell_signature(body, x_retell_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    logger.debug(f"Retell stream event: {payload.get('event', 'unknown')}")

    return EventSourceResponse(
        handle_retell_event_stream(payload),
        media_type="text/event-stream",
    )
