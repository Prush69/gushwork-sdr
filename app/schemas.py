"""Pydantic schemas for every tool I/O in the pipeline.

Gemini tool_call outputs are validated against these schemas BEFORE
the request ever hits FastAPI.  This is the hallucination firewall —
if the LLM fabricates a field or drops a required one, the call is
rejected at the validation layer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# Phase 4 — AEO Audit Tool
# ═══════════════════════════════════════════════════════════


class AuditRequest(BaseModel):
    """Input schema for the `/audit_ai_search` tool call.

    Gemini must produce exactly these fields when it recognizes
    that the caller has stated their company name and industry.
    """

    company_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The prospect's company name as stated on the call.",
    )
    industry: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The prospect's industry vertical (e.g. 'SaaS', 'eCommerce').",
    )
    website_url: Optional[str] = Field(
        None,
        description="Company website URL if mentioned.",
    )


class AuditResult(BaseModel):
    """Output schema returned by the AEO audit engine."""

    company_name: str
    industry: str
    share_of_voice_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current Share of Voice in AI answer engines (0-100%).",
    )
    aeo_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Answer Engine Optimization readiness score.",
    )
    diagnosis: str = Field(
        ...,
        description="Plain-English diagnosis of the company's AI search visibility.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable next steps to improve AEO.",
    )


# ═══════════════════════════════════════════════════════════
# Phase 7 — Calendar Booking Tool
# ═══════════════════════════════════════════════════════════


class BookingRequest(BaseModel):
    """Input schema for the `/book_calendar_slot` tool call.

    Gemini triggers this when the prospect agrees to a meeting.
    """

    prospect_name: str = Field(..., description="Full name of the prospect.")
    prospect_email: Optional[str] = Field(None, description="Email if provided.")
    proposed_time: str = Field(
        ...,
        description=(
            "Conversational time string as spoken by the user, "
            "e.g. 'tomorrow at 3pm', 'next Tuesday morning'."
        ),
    )
    timezone: str = Field(
        default="America/New_York",
        description="IANA timezone of the prospect.",
    )


class BookingResult(BaseModel):
    """Output schema from the calendar booking endpoint."""

    success: bool
    booked_at: Optional[datetime] = None
    calendar_link: Optional[str] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Phase 7 — CRM Sync
# ═══════════════════════════════════════════════════════════


class BANTQualification(BaseModel):
    """BANT qualification data extracted during the call."""

    budget: Optional[str] = Field(None, description="Budget range or signals.")
    authority: Optional[str] = Field(None, description="Decision-maker status.")
    need: Optional[str] = Field(None, description="Core pain point identified.")
    timeline: Optional[str] = Field(None, description="Purchase timeline.")


class LeadSyncRequest(BaseModel):
    """Payload pushed to CRM after call termination."""

    prospect_name: str
    company_name: str
    industry: str
    email: Optional[str] = None
    phone: Optional[str] = None
    bant: BANTQualification
    call_transcript: str = Field(..., description="Full conversation transcript.")
    call_duration_seconds: float
    call_outcome: str = Field(
        ...,
        description="Terminal state: 'meeting_booked', 'callback_requested', 'not_qualified', 'dropped'.",
    )
    audit_result: Optional[AuditResult] = None


class LeadSyncResult(BaseModel):
    """CRM write confirmation."""

    success: bool
    crm_provider: str
    lead_id: Optional[str] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════


class ConversationNode(str, Enum):
    """Every node the LangGraph state machine can occupy."""

    GREETING = "greeting"
    ROUTING = "routing"                 # "support or leads?"
    ICP_QUALIFICATION = "icp_qualification"
    BANT_BUDGET = "bant_budget"
    BANT_AUTHORITY = "bant_authority"
    BANT_NEED = "bant_need"
    BANT_TIMELINE = "bant_timeline"
    AEO_AUDIT = "aeo_audit"            # tool call phase
    AUDIT_RESULTS = "audit_results"     # presenting findings
    OBJECTION_HANDLING = "objection_handling"
    BOOKING = "booking"
    CLOSING = "closing"
    TERMINAL = "terminal"


class CallState(BaseModel):
    """Mutable state carried through LangGraph across every turn."""

    call_id: str = ""
    current_node: ConversationNode = ConversationNode.GREETING
    messages: list[dict] = Field(default_factory=list)
    transcript_segments: list[str] = Field(default_factory=list)

    # Extracted data
    prospect_name: Optional[str] = None
    company_name: Optional[str] = None
    industry: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    # BANT
    bant: BANTQualification = Field(default_factory=BANTQualification)

    # Audit
    audit_result: Optional[AuditResult] = None

    # Booking
    booking_result: Optional[BookingResult] = None

    # Barge-in context
    last_bot_utterance: str = ""
    interrupted_at_char: Optional[int] = None

    # Metadata
    call_start_epoch: float = 0.0
    call_end_epoch: float = 0.0
