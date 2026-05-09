"""LangGraph state machine for the Gushwork SDR conversation flow.

Implements the full BANT qualification graph with:
- Deterministic routing (no LLM-decided edges)
- Tool-level retry loops with graceful failure (Improvement 3)
- LangSmith tracing on every node (Improvement 2)
- Streaming-capable inference (Improvement 1)

LLM: Google Gemini 3 Flash Preview via langchain-google-genai.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.agent.prompts import NODE_PROMPTS, SYSTEM_PROMPT, TOOL_DEFINITIONS
from app.config import settings
from app.schemas import (
    AuditRequest,
    BookingRequest,
    CallState,
    ConversationNode,
)

logger = logging.getLogger(__name__)

# Maximum retries for tool execution before falling back
TOOL_MAX_RETRIES = 2

# ═══════════════════════════════════════════════════════════
# LLM Singleton (supports Gemini & Groq)
# ═══════════════════════════════════════════════════════════

_llm: Any = None
_llm_with_tools: Any | None = None


def get_llm() -> Any:
    """Lazy-init the LLM based on LLM_PROVIDER config (gemini or groq)."""
    global _llm
    if _llm is None:
        provider = settings.llm_provider.lower()
        logger.info(f"Initializing LLM: provider={provider}")

        if provider == "groq":
            from langchain_groq import ChatGroq

            logger.info(f"  model={settings.groq_model}")
            _llm = ChatGroq(
                model=settings.groq_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.groq_api_key,
                streaming=True,
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info(f"  model={settings.gemini_model}")
            kwargs: dict[str, Any] = {
                "model": settings.gemini_model,
                "temperature": settings.llm_temperature,
                "max_output_tokens": settings.llm_max_tokens,
                "google_api_key": settings.gemini_api_key,
                "streaming": True,
            }
            if "preview" in settings.gemini_model or settings.gemini_model.startswith("gemini-3"):
                if not settings.gemini_model.endswith("-lite"):
                    kwargs["thinking_budget"] = 0
                    kwargs["include_thoughts"] = False
            _llm = ChatGoogleGenerativeAI(**kwargs)
    return _llm


def get_llm_with_tools(state: CallState | None = None) -> Any:
    """Return a tool-bound LLM wrapper, dynamically filtering tools based on state."""
    llm = get_llm()
    if not state:
        return llm.bind_tools(TOOL_DEFINITIONS)
        
    tools_to_bind = []
    for tool in TOOL_DEFINITIONS:
        name = tool["function"]["name"]
        if name == "audit_ai_search":
            if state.company_name and state.industry:
                tools_to_bind.append(tool)
        elif name == "book_calendar_slot":
            if state.current_node == ConversationNode.BOOKING:
                tools_to_bind.append(tool)
        else:
            tools_to_bind.append(tool)
            
    if not tools_to_bind:
        return llm
    return llm.bind_tools(tools_to_bind)


def _coerce_state(state: CallState | dict[str, Any]) -> CallState:
    """LangGraph can pass either the Pydantic model or a dumped dict."""
    if isinstance(state, CallState):
        return state
    return CallState(**state)


def _content_to_text(content: Any) -> str:
    """Normalize provider-specific message content blocks to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


# ═══════════════════════════════════════════════════════════
# Node Executors
# ═══════════════════════════════════════════════════════════


def _build_state_summary(state: CallState) -> str:
    """Compact structured state so capped history does not lose key facts."""
    facts: list[str] = [f"Current node: {state.current_node.value}"]

    if state.prospect_name:
        facts.append(f"Prospect: {state.prospect_name}")
    if state.company_name:
        facts.append(f"Company: {state.company_name}")
    if state.industry:
        facts.append(f"Industry: {state.industry}")
    if state.email:
        facts.append(f"Email: {state.email}")
    if state.phone:
        facts.append(f"Phone: {state.phone}")

    bant_facts = [
        ("Budget", state.bant.budget),
        ("Authority", state.bant.authority),
        ("Need", state.bant.need),
        ("Timeline", state.bant.timeline),
    ]
    for label, value in bant_facts:
        if value:
            facts.append(f"{label}: {value}")

    if state.audit_result:
        facts.append(
            "Audit: "
            f"{state.audit_result.share_of_voice_pct:.1f}% Share of Voice, "
            f"{state.audit_result.aeo_score:.1f}/100 AEO score"
        )
    if state.booking_result and state.booking_result.success:
        facts.append(f"Booking confirmed: {state.booking_result.booked_at}")

    return "[KNOWN CALL STATE]\n" + "\n".join(facts)


def _recent_messages(state: CallState) -> list[dict[str, Any]]:
    """Keep LLM context bounded while preserving a valid tool-message sequence."""
    max_messages = max(settings.llm_context_messages, 0)
    history = state.messages[-max_messages:] if max_messages else state.messages

    # Tool messages without their immediately preceding assistant tool-call
    # can be rejected by chat providers, so drop any orphaned leading tool rows.
    while history and history[0].get("role") == "tool":
        history = history[1:]
    return history


def _build_messages(state: CallState | dict[str, Any]) -> list:
    """Construct the message array for Gemini from current state."""
    state = _coerce_state(state)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.append(SystemMessage(content=_build_state_summary(state)))

    # Inject node-specific context
    node_prompt = NODE_PROMPTS.get(state.current_node.value, "")
    if node_prompt:
        messages.append(
            SystemMessage(content=f"[CURRENT PHASE: {state.current_node.value}]\n{node_prompt}")
        )

    # Barge-in context
    if state.interrupted_at_char is not None and state.last_bot_utterance:
        truncated = state.last_bot_utterance[: state.interrupted_at_char]
        messages.append(
            SystemMessage(
                content=(
                    f"[BARGE-IN] You were interrupted mid-sentence.  "
                    f"You had said: \"{truncated}...\" before the user cut in."
                )
            )
        )

    # Conversation history
    for msg in _recent_messages(state):
        role = msg.get("role", "user")
        content = _content_to_text(msg.get("content", ""))
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )

    return messages


@traceable(name="llm_inference", run_type="llm")
async def llm_inference(state: CallState | dict[str, Any]) -> dict[str, Any]:
    """Core LLM inference node — fires Gemini 3 Flash with tools bound.

    Traced by LangSmith for full observability.
    """
    state = _coerce_state(state)
    llm_with_tools = get_llm_with_tools(state)
    messages = _build_messages(state)

    t0 = time.perf_counter()
    response = await llm_with_tools.ainvoke(messages)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Gemini inference: {latency_ms:.0f}ms")

    # Check for tool calls
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": state.messages
            + [{"role": "assistant", "content": "", "tool_calls": tool_calls}],
            "last_bot_utterance": "",
        }

    # Regular text response
    text = _content_to_text(response.content)
    return {
        "messages": state.messages + [{"role": "assistant", "content": text}],
        "last_bot_utterance": text,
        "interrupted_at_char": None,
    }


@traceable(name="llm_stream", run_type="llm")
async def llm_inference_stream(state: CallState | dict[str, Any]) -> AsyncIterator[str]:
    """Streaming variant — yields tokens as Gemini produces them.

    Used by the SSE endpoint for sub-100ms TTFB (Improvement 1).
    """
    state = _coerce_state(state)
    llm_with_tools = get_llm_with_tools(state)
    messages = _build_messages(state)

    t0 = time.perf_counter()
    first_token = True

    async for chunk in llm_with_tools.astream(messages):
        if first_token:
            ttfb = (time.perf_counter() - t0) * 1000
            logger.info(f"Gemini TTFB: {ttfb:.0f}ms")
            first_token = False

        # Check for tool calls in the stream
        tool_calls = getattr(chunk, "tool_calls", None) or []
        if tool_calls:
            # Tool calls can't be streamed — yield a marker
            yield f"__TOOL_CALL__:{tool_calls[0].get('name', '')}"
            return

        content = _content_to_text(chunk.content)
        if content:
            yield content

    total = (time.perf_counter() - t0) * 1000
    logger.info(f"Gemini total stream: {total:.0f}ms")


# ═══════════════════════════════════════════════════════════
# Routing Logic
# ═══════════════════════════════════════════════════════════


def _detect_intent(user_text: str) -> str:
    """Keyword-based intent detection for deterministic routing."""
    text = user_text.lower()

    if any(w in text for w in ["cost", "price", "expensive", "how much", "budget", "afford"]):
        return "objection_pricing"
    if any(w in text for w in ["not ready", "later", "next quarter", "not now"]):
        return "objection_timing"
    if any(w in text for w in ["my boss", "team", "check with", "not my call"]):
        return "objection_authority"
    if any(w in text for w in ["book", "schedule", "meeting", "let's set up", "calendar"]):
        return "booking"
    if any(w in text for w in ["support", "help", "issue", "bug", "problem"]):
        return "support"
    if any(w in text for w in ["leads", "seo", "marketing", "growth", "inbound"]):
        return "leads"
    return "continue"


def route_after_greeting(state: CallState | dict[str, Any]) -> str:
    """Router after greeting — determines if leads or support."""
    state = _coerce_state(state)
    if not state.messages:
        return "llm_inference"
    return "llm_inference"


def route_after_inference(state: CallState | dict[str, Any]) -> str:
    """Router after LLM inference — check for tool calls or next node."""
    state = _coerce_state(state)
    if not state.messages:
        return END

    last_msg = state.messages[-1]
    tool_calls = last_msg.get("tool_calls", [])

    if tool_calls:
        tool_name = tool_calls[0].get("name", "")
        if tool_name == "audit_ai_search":
            return "execute_audit"
        elif tool_name == "book_calendar_slot":
            return "execute_booking"

    node = state.current_node
    if node == ConversationNode.CLOSING:
        return END

    return END  # Single turn — return to Retell


# ═══════════════════════════════════════════════════════════
# Tool Execution Nodes — With Retry Loops (Improvement 3)
# ═══════════════════════════════════════════════════════════


class ToolExecutionError(Exception):
    """Raised when a tool fails and needs LLM-mediated recovery."""

    def __init__(self, tool_name: str, error: str, original_args: dict):
        self.tool_name = tool_name
        self.error = error
        self.original_args = original_args
        super().__init__(f"Tool '{tool_name}' failed: {error}")


@traceable(name="execute_audit", run_type="tool")
async def execute_audit(state: CallState | dict[str, Any]) -> dict[str, Any]:
    """Execute the AEO audit tool call with retry logic.

    If the audit fails (API timeout, bad data), the error is passed back
    to Gemini with instructions to ask the user for clarification.
    """
    from app.tools.audit import run_aeo_audit

    state = _coerce_state(state)
    last_msg = state.messages[-1]
    tool_calls = last_msg.get("tool_calls", [])
    if not tool_calls:
        return {}

    tc = tool_calls[0]
    args = tc.get("args", {})

    for attempt in range(1, TOOL_MAX_RETRIES + 1):
        try:
            request = AuditRequest(**args)
            result = await run_aeo_audit(request)

            return {
                "messages": state.messages
                + [
                    {
                        "role": "tool",
                        "content": result.model_dump_json(),
                        "tool_call_id": tc.get("id", ""),
                    }
                ],
                "audit_result": result,
                "company_name": request.company_name,
                "industry": request.industry,
                "current_node": ConversationNode.AUDIT_RESULTS,
            }

        except Exception as e:
            logger.warning(
                f"Audit tool attempt {attempt}/{TOOL_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            if attempt == TOOL_MAX_RETRIES:
                # Final failure — inject error into conversation for LLM recovery
                error_msg = (
                    f"The audit_ai_search tool failed with error: {str(e)}. "
                    f"Original arguments: {args}. "
                    f"Ask the user to clarify their company name and industry, "
                    f"then try the tool call again."
                )
                return {
                    "messages": state.messages
                    + [
                        {
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": tc.get("id", ""),
                        }
                    ],
                    "current_node": ConversationNode.ICP_QUALIFICATION,
                }

    return {}


@traceable(name="execute_booking", run_type="tool")
async def execute_booking(state: CallState | dict[str, Any]) -> dict[str, Any]:
    """Execute the calendar booking tool call with retry logic.

    If Cal.com rejects the time or dateutil can't parse it, Gemini
    asks the user for a clearer time.
    """
    from app.tools.calendar import book_slot

    state = _coerce_state(state)
    last_msg = state.messages[-1]
    tool_calls = last_msg.get("tool_calls", [])
    if not tool_calls:
        return {}

    tc = tool_calls[0]
    args = tc.get("args", {})

    for attempt in range(1, TOOL_MAX_RETRIES + 1):
        try:
            request = BookingRequest(**args)
            result = await book_slot(request)

            if not result.success:
                raise Exception(result.error or "Booking failed with no error message")

            return {
                "messages": state.messages
                + [
                    {
                        "role": "tool",
                        "content": result.model_dump_json(),
                        "tool_call_id": tc.get("id", ""),
                    }
                ],
                "booking_result": result,
                "prospect_name": request.prospect_name,
                "current_node": ConversationNode.CLOSING,
            }

        except Exception as e:
            logger.warning(
                f"Booking tool attempt {attempt}/{TOOL_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            if attempt == TOOL_MAX_RETRIES:
                error_msg = (
                    f"The book_calendar_slot tool failed with error: {str(e)}. "
                    f"The proposed time was: '{args.get('proposed_time', 'unknown')}'. "
                    f"Ask the user for a more specific date and time, "
                    f"then try the tool call again."
                )
                return {
                    "messages": state.messages
                    + [
                        {
                            "role": "tool",
                            "content": error_msg,
                            "tool_call_id": tc.get("id", ""),
                        }
                    ],
                    "current_node": ConversationNode.BOOKING,
                }

    return {}


@traceable(name="secondary_inference", run_type="llm")
async def secondary_inference(state: CallState | dict[str, Any]) -> dict[str, Any]:
    """Secondary LLM call after tool execution — presents results to user."""
    state = _coerce_state(state)
    try:
        return await llm_inference(state)
    except Exception:
        logger.exception("Secondary inference failed; using deterministic tool summary")

        if state.audit_result:
            audit = state.audit_result
            text = (
                f"I ran the audit: {audit.company_name} is at "
                f"{audit.share_of_voice_pct:.1f}% Share of Voice with an "
                f"AEO score of {audit.aeo_score:.1f}/100. {audit.diagnosis}"
            )
            return {
                "messages": state.messages + [{"role": "assistant", "content": text}],
                "last_bot_utterance": text,
                "current_node": ConversationNode.AUDIT_RESULTS,
            }

        if state.booking_result:
            booking = state.booking_result
            if booking.success:
                text = f"You're booked for {booking.booked_at}. I'll sync these notes now."
                return {
                    "messages": state.messages + [{"role": "assistant", "content": text}],
                    "last_bot_utterance": text,
                    "current_node": ConversationNode.CLOSING,
                }

        raise


# ═══════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine.

    All nodes are traced via @traceable for LangSmith observability.
    Tool nodes include retry logic — on failure, errors are passed
    back to Gemini for self-correction (Improvement 3).
    """
    workflow = StateGraph(CallState)

    workflow.add_node("llm_inference", llm_inference)
    workflow.add_node("execute_audit", execute_audit)
    workflow.add_node("execute_booking", execute_booking)
    workflow.add_node("secondary_inference", secondary_inference)

    workflow.set_entry_point("llm_inference")

    workflow.add_conditional_edges(
        "llm_inference",
        route_after_inference,
        {
            "execute_audit": "execute_audit",
            "execute_booking": "execute_booking",
            END: END,
        },
    )
    workflow.add_edge("execute_audit", "secondary_inference")
    workflow.add_edge("execute_booking", "secondary_inference")
    workflow.add_edge("secondary_inference", END)

    return workflow.compile()


# Compiled graph singleton
agent_graph = build_graph()
