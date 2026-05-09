"""Retell AI routes — WebSocket + Web Call API.

Two endpoints:
1. WebSocket /llm-websocket/{call_id} — Retell connects here for Custom LLM
2. POST /create-web-call — Frontend calls this to get an access_token
3. POST /webhook — Lifecycle events (call_started, call_ended, call_analyzed)
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.agent.retell_handler import handle_websocket
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["retell"])


# ═══════════════════════════════════════════════════════════
# 1. Custom LLM WebSocket — Retell connects here
# ═══════════════════════════════════════════════════════════


@router.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    """Retell Custom LLM WebSocket endpoint.

    Retell opens a bidirectional WebSocket here when a call starts.
    We receive transcripts, run Gemini + tools, and stream responses back.
    """
    await handle_websocket(websocket, call_id)


# ═══════════════════════════════════════════════════════════
# 2. Create Web Call — Frontend requests an access_token
# ═══════════════════════════════════════════════════════════


@router.post("/create-web-call")
async def create_web_call():
    """Create a Retell web call and return the access_token.

    The frontend calls this endpoint, gets a short-lived token,
    and uses it with retell-client-js-sdk to start the call.
    """
    if not settings.retell_api_key:
        return JSONResponse(
            status_code=500,
            content={"error": "RETELL_API_KEY not configured"},
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.retellai.com/v2/create-web-call",
                headers={
                    "Authorization": f"Bearer {settings.retell_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "agent_id": settings.retell_agent_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"🌐 Web call created: call_id={data.get('call_id', 'unknown')}")

            return {
                "access_token": data.get("access_token"),
                "call_id": data.get("call_id"),
            }

    except httpx.HTTPError as e:
        logger.error(f"Failed to create web call: {e}", exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": f"Retell API error: {str(e)}"},
        )


# ═══════════════════════════════════════════════════════════
# 3. Lifecycle Webhook — call_started, call_ended, call_analyzed
# ═══════════════════════════════════════════════════════════


@router.post("/webhook")
async def handle_webhook(request: Request):
    """Retell lifecycle webhook — call_started, call_ended, call_analyzed.

    This is separate from the WebSocket. Retell POSTs here for
    lifecycle events like recordings, transcripts, and analytics.
    """
    try:
        post_data = await request.json()
        event = post_data.get("event", "unknown")
        call_id = post_data.get("data", {}).get("call_id", "unknown")

        if event == "call_started":
            logger.info(f"📞 Webhook: call_started {call_id}")
        elif event == "call_ended":
            logger.info(f"📴 Webhook: call_ended {call_id}")
        elif event == "call_analyzed":
            logger.info(f"📊 Webhook: call_analyzed {call_id}")
        else:
            logger.info(f"❓ Webhook: {event} {call_id}")

        return JSONResponse(status_code=200, content={"received": True})

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error"},
        )
