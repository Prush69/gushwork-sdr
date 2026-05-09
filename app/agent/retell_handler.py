"""Retell AI webhook handler — manages the full call lifecycle.

This module bridges Retell's WebRTC voice infrastructure with the LangGraph
state machine.  Supports both:
- Standard JSON responses (for lifecycle events)
- SSE streaming responses (Improvement 1: sub-100ms TTFB)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, AsyncIterator

from langsmith import traceable

from app.agent.graph import agent_graph, llm_inference_stream, _build_messages, get_llm
from app.config import settings
from app.schemas import CallState, ConversationNode

logger = logging.getLogger(__name__)

# In-memory call state store (swap for Redis in production)
_active_calls: dict[str, CallState] = {}


def verify_retell_signature(payload: bytes, signature: str) -> bool:
    """Verify the HMAC-SHA256 signature from Retell's webhook."""
    if not settings.retell_webhook_secret:
        logger.warning("Retell webhook secret not configured — skipping verification")
        return True

    expected = hmac.new(
        settings.retell_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def get_or_create_call_state(call_id: str) -> CallState:
    """Get existing call state or create a fresh one."""
    if call_id not in _active_calls:
        _active_calls[call_id] = CallState(
            call_id=call_id,
            current_node=ConversationNode.GREETING,
            call_start_epoch=time.time(),
        )
    return _active_calls[call_id]


def destroy_call_state(call_id: str) -> CallState | None:
    """Remove and return the call state on disconnect."""
    state = _active_calls.pop(call_id, None)
    if state:
        state.call_end_epoch = time.time()
    return state


@traceable(name="handle_retell_event", run_type="chain")
async def handle_retell_event(event: dict[str, Any]) -> dict[str, Any]:
    """Main dispatcher for Retell webhook events.

    Events:
    - call_started: Initialize state, return greeting config
    - call_ended: Trigger CRM sync, cleanup
    - agent_response_required: User spoke → LLM inference → response
    - ping: Health check
    """
    event_type = event.get("event", "")
    call_id = event.get("call_id", event.get("call", {}).get("call_id", ""))

    if event_type == "call_started":
        return await _handle_call_started(call_id, event)
    elif event_type == "call_ended":
        return await _handle_call_ended(call_id, event)
    elif event_type == "agent_response_required":
        return await _handle_response_required(call_id, event)
    elif event_type == "ping":
        return {"response_type": "pong"}
    else:
        logger.warning(f"Unknown Retell event type: {event_type}")
        return {"response_type": "ack"}


async def _handle_call_started(call_id: str, event: dict) -> dict:
    """Phase 1 — Connection & Zero-Latency Cold Start."""
    state = get_or_create_call_state(call_id)
    logger.info(f"Call started: {call_id}")

    return {
        "response_type": "agent_response",
        "response": {
            "content": (
                "Thanks for calling Gushwork. Are you calling for support, "
                "or looking to generate more inbound leads?"
            ),
        },
    }


async def _handle_call_ended(call_id: str, event: dict) -> dict:
    """Phase 7 — Terminal State & CRM Handoff."""
    state = destroy_call_state(call_id)

    if state:
        logger.info(
            f"Call ended: {call_id} | "
            f"duration={state.call_end_epoch - state.call_start_epoch:.1f}s | "
            f"node={state.current_node.value}"
        )
        from app.tools.crm import fire_crm_sync
        await fire_crm_sync(state)

    return {"response_type": "ack"}


@traceable(name="response_required", run_type="chain")
async def _handle_response_required(call_id: str, event: dict) -> dict:
    """Phase 3 → 5 — STT transcript → LLM inference → TTS response.

    The hot path.  Traced end-to-end by LangSmith.
    """
    state = get_or_create_call_state(call_id)
    transcript = event.get("transcript", "")

    # Handle barge-in context (Phase 6)
    if event.get("interrupted", False):
        state.interrupted_at_char = event.get("interrupted_at_char")
        logger.info(f"Barge-in detected on call {call_id}")

    # Inject user transcript into state
    if transcript:
        state.messages.append({"role": "user", "content": transcript})
        state.transcript_segments.append(transcript)

    # Determine current node from intent
    intent = _detect_routing_intent(transcript, state)
    state.current_node = intent

    # Run LangGraph
    t0 = time.perf_counter()
    try:
        result = await agent_graph.ainvoke(state.model_dump())

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Full pipeline: {elapsed_ms:.0f}ms | node={state.current_node.value}")

        # Extract response
        response_text = ""
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if msg.get("role") == "assistant" and msg.get("content"):
                    response_text = msg["content"]
                    break

        # Update state
        state.messages = result.get("messages", state.messages)
        state.last_bot_utterance = result.get("last_bot_utterance", "")
        state.audit_result = result.get("audit_result", state.audit_result)
        state.booking_result = result.get("booking_result", state.booking_result)
        state.current_node = ConversationNode(
            result.get("current_node", state.current_node)
        )

        if result.get("company_name"):
            state.company_name = result["company_name"]
        if result.get("industry"):
            state.industry = result["industry"]
        if result.get("prospect_name"):
            state.prospect_name = result["prospect_name"]

        return {
            "response_type": "agent_response",
            "response": {"content": response_text},
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Pipeline error on call {call_id} ({elapsed_ms:.0f}ms): {e}",
            exc_info=True,
        )
        return {
            "response_type": "agent_response",
            "response": {
                "content": (
                    "I apologize, I'm having a brief technical issue.  "
                    "Could you repeat that?"
                ),
            },
        }


# ═══════════════════════════════════════════════════════════
# SSE Streaming Handler (Improvement 1)
# ═══════════════════════════════════════════════════════════


async def handle_retell_event_stream(event: dict[str, Any]) -> AsyncIterator[str]:
    """Streaming variant — yields SSE events with sub-100ms TTFB.

    Instead of waiting for Gemini to finish the full response, we pipe
    tokens directly to Retell's TTS engine as they're generated.
    Each SSE event is a JSON fragment that Retell can ingest immediately.
    """
    event_type = event.get("event", "")
    call_id = event.get("call_id", event.get("call", {}).get("call_id", ""))

    # Only agent_response_required benefits from streaming
    if event_type != "agent_response_required":
        result = await handle_retell_event(event)
        yield json.dumps(result)
        return

    state = get_or_create_call_state(call_id)
    transcript = event.get("transcript", "")

    if event.get("interrupted", False):
        state.interrupted_at_char = event.get("interrupted_at_char")

    if transcript:
        state.messages.append({"role": "user", "content": transcript})
        state.transcript_segments.append(transcript)

    intent = _detect_routing_intent(transcript, state)
    state.current_node = intent

    # Stream tokens from Gemini
    full_response = ""
    try:
        async for token in llm_inference_stream(state):
            if token.startswith("__TOOL_CALL__:"):
                # Tool call detected — fall back to non-streaming path
                result = await agent_graph.ainvoke(state.model_dump())
                response_text = ""
                if result.get("messages"):
                    for msg in reversed(result["messages"]):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            response_text = msg["content"]
                            break
                state.messages = result.get("messages", state.messages)
                state.last_bot_utterance = result.get("last_bot_utterance", "")
                state.audit_result = result.get("audit_result", state.audit_result)
                state.booking_result = result.get("booking_result", state.booking_result)
                yield json.dumps({
                    "response_type": "agent_response",
                    "response": {"content": response_text},
                })
                return

            full_response += token
            # Yield each token as an SSE data chunk
            yield json.dumps({
                "response_type": "agent_response_stream",
                "response": {"content_delta": token},
            })

        # Final message — signal stream complete
        state.messages.append({"role": "assistant", "content": full_response})
        state.last_bot_utterance = full_response
        yield json.dumps({
            "response_type": "agent_response_stream_end",
            "response": {"content": full_response},
        })

    except Exception as e:
        logger.error(f"Stream error on call {call_id}: {e}", exc_info=True)
        yield json.dumps({
            "response_type": "agent_response",
            "response": {
                "content": "I apologize, could you repeat that?"
            },
        })


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
            return ConversationNode.AEO_AUDIT
        return ConversationNode.ICP_QUALIFICATION
    if current == ConversationNode.AEO_AUDIT:
        return ConversationNode.AUDIT_RESULTS
    if current == ConversationNode.AUDIT_RESULTS:
        return ConversationNode.BANT_NEED
    if current == ConversationNode.BANT_NEED:
        return ConversationNode.BANT_BUDGET
    if current == ConversationNode.BANT_BUDGET:
        return ConversationNode.BANT_TIMELINE
    if current == ConversationNode.BANT_TIMELINE:
        return ConversationNode.BOOKING
    if current == ConversationNode.BOOKING:
        return ConversationNode.CLOSING
    if current == ConversationNode.CLOSING:
        return ConversationNode.TERMINAL

    return current
