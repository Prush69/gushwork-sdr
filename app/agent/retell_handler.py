"""Retell AI Custom LLM — WebSocket Handler.

Retell connects to our server via WebSocket (not HTTP).
Protocol: wss://our-server/llm-websocket/{call_id}

Message types FROM Retell:
  - call_details: initial call metadata
  - ping_pong: keep-alive
  - update_only: real-time transcript (no response needed)
  - response_required: user finished speaking → we must respond
  - reminder_required: user went silent → nudge them

Message types TO Retell:
  - config: optional initial configuration
  - response: text to speak (supports streaming via content_complete=False)
  - ping_pong: echo back timestamps
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from langsmith import traceable

from app.config import settings
from app.schemas import CallState, ConversationNode

logger = logging.getLogger(__name__)

# ── In-memory call state ──────────────────────────────────
_active_calls: dict[str, CallState] = {}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_INDUSTRY_ALIASES: tuple[tuple[str, str], ...] = (
    ("b2b saas", "B2B SaaS"),
    ("saas", "SaaS"),
    ("software", "Software"),
    ("ecommerce", "eCommerce"),
    ("e-commerce", "eCommerce"),
    ("retail", "Retail"),
    ("fintech", "FinTech"),
    ("healthcare", "Healthcare"),
    ("agency", "Agency"),
    ("marketing", "Marketing"),
)


# ═══════════════════════════════════════════════════════════
# WebSocket Handler — The Core Loop
# ═══════════════════════════════════════════════════════════


async def handle_websocket(websocket: WebSocket, call_id: str) -> None:
    """Main WebSocket handler for Retell Custom LLM.

    This is the bidirectional connection Retell opens when a call starts.
    We receive transcripts and send back text for TTS.
    """
    await websocket.accept()
    logger.info(f"🔌 WebSocket connected: {call_id}")

    # Create call state
    state = CallState(
        call_id=call_id,
        current_node=ConversationNode.GREETING,
        call_start_epoch=time.time(),
    )
    _active_calls[call_id] = state

    # Track the latest response_id to handle barge-in
    current_response_id = 0

    try:
        # Step 1: Send config
        await websocket.send_json({
            "response_type": "config",
            "config": {
                "auto_reconnect": True,
                "call_details": True,
            },
        })

        # Step 2: Send greeting (first message)
        await websocket.send_json({
            "response_type": "response",
            "response_id": 0,
            "content": (
                "Thanks for calling Gushwork! Are you calling for support, "
                "or are you looking to generate more inbound leads?"
            ),
            "content_complete": True,
            "end_call": False,
        })

        # Step 3: Listen for messages
        async for data in websocket.iter_json():
            interaction_type = data.get("interaction_type", "")

            if interaction_type == "call_details":
                logger.info(f"📞 Call details received for {call_id}")
                continue

            if interaction_type == "ping_pong":
                await websocket.send_json({
                    "response_type": "ping_pong",
                    "timestamp": data.get("timestamp", 0),
                })
                continue

            if interaction_type == "update_only":
                # Real-time transcript update — extract fields passively
                transcript = data.get("transcript", [])
                if transcript:
                    last_utterance = transcript[-1].get("content", "")
                    _extract_known_fields(last_utterance, state)
                continue

            if interaction_type in ("response_required", "reminder_required"):
                response_id = data.get("response_id", 0)
                current_response_id = response_id
                transcript = data.get("transcript", [])

                logger.info(
                    f"💬 {interaction_type} | response_id={response_id} | "
                    f"last='{transcript[-1]['content'][:80]}...'" if transcript else f"💬 {interaction_type} (empty)"
                )

                # Process and stream response
                await _handle_response(
                    websocket=websocket,
                    state=state,
                    transcript=transcript,
                    response_id=response_id,
                    is_reminder=(interaction_type == "reminder_required"),
                    current_response_id_ref=lambda: current_response_id,
                )

    except WebSocketDisconnect:
        logger.info(f"📴 WebSocket disconnected: {call_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for {call_id}: {e}", exc_info=True)
        try:
            await websocket.close(1011, "Server error")
        except Exception:
            pass
    finally:
        # Call ended — trigger CRM sync
        state.call_end_epoch = time.time()
        _active_calls.pop(call_id, None)
        duration = state.call_end_epoch - state.call_start_epoch
        logger.info(
            f"📊 Call ended: {call_id} | "
            f"duration={duration:.1f}s | node={state.current_node.value}"
        )
        asyncio.create_task(_safe_fire_crm_sync(state))


# ═══════════════════════════════════════════════════════════
# Response Generation
# ═══════════════════════════════════════════════════════════


@traceable(name="handle_response", run_type="chain")
async def _handle_response(
    websocket: WebSocket,
    state: CallState,
    transcript: list[dict],
    response_id: int,
    is_reminder: bool,
    current_response_id_ref: Any,
) -> None:
    """Generate a response and stream it back to Retell.

    Handles:
    - Regular conversation (streamed token by token)
    - Tool calls (audit, booking) with filler words
    - Barge-in (abandon response if response_id changes)
    - Reminders (nudge silent users)
    """
    from app.agent.graph import get_llm_with_tools, llm_inference_stream
    from app.agent.prompts import SYSTEM_PROMPT, NODE_PROMPTS

    # Sync Retell transcript into our state
    state.messages = _retell_transcript_to_messages(transcript)

    # Extract known fields from latest user utterance
    if transcript:
        last_user = transcript[-1].get("content", "")
        _extract_known_fields(last_user, state)

    # ⚡ TRIGGER BACKGROUND AEO AUDIT IF NOT STARTED
    if state.company_name and state.industry and not state.audit_started:
        state.audit_started = True
        from app.tools.audit import run_background_aeo_audit
        asyncio.create_task(run_background_aeo_audit(state))

    # Determine conversation node
    user_text = transcript[-1]["content"] if transcript else ""
    state.current_node = _detect_routing_intent(user_text, state)

    if state.current_node == ConversationNode.BOOKING:
        # Let Gemini handle booking naturally via tool calls
        pass

    # Loop to allow continuous generation after a tool call
    while True:
        try:
            full_response = ""
            async for token in llm_inference_stream(state):
                # Check for barge-in
                if current_response_id_ref() != response_id:
                    logger.info(f"🛑 Barge-in detected, abandoning response_id={response_id}")
                    return

                if token.startswith("__TOOL_CALL__:"):
                    # Tool call detected during streaming
                    tool_name = token.split(":", 1)[1]
                    await _handle_tool_call_during_stream(
                        websocket, state, response_id, tool_name, current_response_id_ref
                    )
                    break # Break inner loop, restart inference

                full_response += token
                # Stream each token to Retell
                await websocket.send_json({
                    "response_type": "response",
                    "response_id": response_id,
                    "content": token,
                    "content_complete": False,
                    "end_call": False,
                })
            else:
                # Loop naturally finished without a break
                state.last_bot_utterance = full_response
                await websocket.send_json({
                    "response_type": "response",
                    "response_id": response_id,
                    "content": "",
                    "content_complete": True,
                    "end_call": state.current_node == ConversationNode.TERMINAL,
                })
                return

        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            await websocket.send_json({
                "response_type": "response",
                "response_id": response_id,
                "content": "I apologize, could you repeat that?",
                "content_complete": True,
                "end_call": False,
            })
            return


# ═══════════════════════════════════════════════════════════
# Tool Execution with Filler Words
# ═══════════════════════════════════════════════════════════


async def _handle_audit_with_filler(
    websocket: WebSocket,
    state: CallState,
    response_id: int,
    current_response_id_ref: Any,
) -> None:
    """Run the AEO audit while speaking a filler phrase.

    This masks the API latency — the user hears "Let me check your
    AI visibility real quick..." while we run 5 parallel Gemini queries.
    """
    from app.tools.audit import run_aeo_audit
    from app.schemas import AuditRequest

    # Send filler word immediately
    filler = (
        f"Great, let me run a quick AI visibility check on {state.company_name} "
        f"in the {state.industry} space. Just one moment..."
    )
    await websocket.send_json({
        "response_type": "response",
        "response_id": response_id,
        "content": filler,
        "content_complete": False,
        "end_call": False,
    })

    # Run audit in background
    try:
        request = AuditRequest(
            company_name=state.company_name,
            industry=state.industry,
        )
        result = await run_aeo_audit(request)
        state.audit_result = result
        state.current_node = ConversationNode.AUDIT_RESULTS

        # Wait for next response_required with the audit results ready
        # The next turn, Gemini will have the audit results in context
        # and will present them naturally
        logger.info(
            f"✅ Audit complete: SoV={result.share_of_voice_pct}%, "
            f"Score={result.aeo_score}"
        )

    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        state.current_node = ConversationNode.ICP_QUALIFICATION


async def _handle_tool_call_during_stream(
    websocket: WebSocket,
    state: CallState,
    response_id: int,
    tool_name: str,
    current_response_id_ref: Any,
) -> None:
    """Handle a tool call that was detected mid-stream."""
    from app.agent.graph import agent_graph

    logger.info(f"🔧 Tool call detected: {tool_name}")

    # Send filler
    if tool_name == "audit_ai_search":
        filler = "Let me check your AI visibility real quick..."
    elif tool_name == "book_calendar_slot":
        filler = "Let me check the calendar for you..."
    else:
        filler = "One moment please..."

    await websocket.send_json({
        "response_type": "response",
        "response_id": response_id,
        "content": filler,
        "content_complete": False,
        "end_call": False,
    })

    # Run the full graph (which will execute the tool)
    try:
        result = await agent_graph.ainvoke(state.model_dump())
        _apply_graph_result(state, result)
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════
# Routing Logic
# ═══════════════════════════════════════════════════════════


def _detect_routing_intent(transcript: str, state: CallState) -> ConversationNode:
    """Deterministic node routing based on transcript + current state."""
    text = transcript.lower() if transcript else ""
    current = state.current_node

    # Objection detection overrides everything
    if any(w in text for w in ["cost", "price", "expensive", "how much"]):
        return ConversationNode.OBJECTION_HANDLING
    if any(w in text for w in ["not ready", "later", "next quarter"]):
        return ConversationNode.OBJECTION_HANDLING
    if any(w in text for w in ["my boss", "team", "check with"]):
        return ConversationNode.OBJECTION_HANDLING

    # Booking intent
    if any(w in text for w in ["book", "schedule", "meeting", "set up"]):
        return ConversationNode.BOOKING

    # Progressive BANT flow
    if current == ConversationNode.GREETING:
        return ConversationNode.ROUTING
    if current == ConversationNode.ROUTING:
        if any(w in text for w in ["support", "help", "issue"]):
            return ConversationNode.TERMINAL
        return ConversationNode.ICP_QUALIFICATION
    if current == ConversationNode.ICP_QUALIFICATION:
        if state.company_name and state.industry:
            return ConversationNode.BANT_NEED
        return ConversationNode.ICP_QUALIFICATION
    if current == ConversationNode.BANT_NEED:
        return ConversationNode.BANT_BUDGET
    if current == ConversationNode.BANT_BUDGET:
        return ConversationNode.BANT_AUTHORITY
    if current == ConversationNode.BANT_AUTHORITY:
        return ConversationNode.BANT_TIMELINE
    if current == ConversationNode.BANT_TIMELINE:
        return ConversationNode.BOOKING
    if current == ConversationNode.BOOKING:
        if state.booking_result and state.booking_result.success:
            return ConversationNode.CLOSING
        return ConversationNode.BOOKING
    if current == ConversationNode.CLOSING:
        return ConversationNode.TERMINAL

    return current


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _retell_transcript_to_messages(transcript: list[dict]) -> list[dict]:
    """Convert Retell's transcript format to our internal message format."""
    messages = []
    for entry in transcript:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "agent":
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": content})
    return messages


def _extract_known_fields(transcript: str, state: CallState) -> None:
    """Populate cheap structured fields before the LLM turn."""
    if not transcript:
        return

    if not state.email:
        email_match = _EMAIL_RE.search(transcript)
        if email_match:
            state.email = email_match.group(0)

    text = transcript.lower()
    if not state.industry:
        for needle, canonical in _INDUSTRY_ALIASES:
            if needle in text:
                state.industry = canonical
                break

    if state.company_name:
        return

    # Simple company name extraction patterns
    patterns = [
        r"(?i)(?:company|we(?:\s+are|'re)|i(?:\s+am|'m)\s+(?:with|from|at)|at)\s+([A-Z][A-Za-z0-9&.\- ]+)",
        r"called\s+([A-Z][A-Za-z0-9&.\- ]+)",
        r"(?:^|,\s*)\s*([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+)*)\s*,\s*(?:in |an? )",
    ]
    for pattern in patterns:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            company = match.group(1).strip(" ,.;:")
            # Truncate at common boundary words to prevent over-capture
            boundary = re.search(
                r"\b(?:and|but|or|in|we|who|that|which|for|with|from|to|is|are|do|does)\b",
                company,
                re.IGNORECASE,
            )
            if boundary:
                company = company[: boundary.start()].strip(" ,.;:")
            if company and company.lower() not in {"a", "an", "the", "we", "i", "my", "our"}:
                state.company_name = company
                return


def _apply_graph_result(state: CallState, result: dict) -> None:
    """Merge LangGraph output back into call state."""
    state.messages = result.get("messages", state.messages)
    state.last_bot_utterance = result.get("last_bot_utterance", state.last_bot_utterance)
    state.audit_result = result.get("audit_result", state.audit_result)
    state.booking_result = result.get("booking_result", state.booking_result)

    current_node = result.get("current_node")
    if current_node:
        state.current_node = ConversationNode(current_node)

    for field in ("company_name", "industry", "prospect_name", "email", "phone"):
        value = result.get(field)
        if value:
            setattr(state, field, value)


async def _safe_fire_crm_sync(state: CallState) -> None:
    """Run CRM sync outside the call path and contain failures."""
    try:
        from app.tools.crm import fire_crm_sync
        await fire_crm_sync(state)
    except Exception:
        logger.exception("CRM sync failed after call end: %s", state.call_id)
