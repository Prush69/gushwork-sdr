"""Tool execution routes — the FastAPI endpoints Gemini calls via tool_call.

These are the three endpoints from the pipeline spec:
- POST /audit_ai_search  (Phase 4, Step 12)
- POST /book_calendar_slot  (Phase 7, Step 23)
- POST /sync_crm  (Phase 7, Step 26)

Each endpoint validates input via Pydantic schemas (the PydanticAI firewall),
executes the tool logic, and returns structured JSON.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas import (
    AuditRequest,
    AuditResult,
    BookingRequest,
    BookingResult,
    LeadSyncRequest,
    LeadSyncResult,
)
from app.tools.audit import run_aeo_audit
from app.tools.calendar import book_slot
from app.tools.crm import _sync_hubspot, _sync_salesforce
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tools"])


@router.post("/audit_ai_search", response_model=AuditResult)
async def audit_ai_search(request: AuditRequest):
    """Phase 4, Step 12 — AEO Audit Execution.

    Gemini generates a tool_call → Pydantic validates → this endpoint
    runs the audit → returns diagnosis JSON → Gemini presents results.
    """
    logger.info(f"Audit requested: {request.company_name} ({request.industry})")
    result = await run_aeo_audit(request)
    return result


@router.post("/book_calendar_slot", response_model=BookingResult)
async def book_calendar_slot(request: BookingRequest):
    """Phase 7, Step 23 — Calendar Booking.

    Parses conversational time string → normalizes to ISO-8601 →
    books via Cal.com → returns confirmation.
    """
    logger.info(f"Booking requested: {request.prospect_name} @ {request.proposed_time}")
    result = await book_slot(request)
    return result


@router.post("/sync_crm", response_model=LeadSyncResult)
async def sync_crm(request: LeadSyncRequest):
    """Phase 7, Step 26-27 — Asynchronous CRM Sync.

    Called after call termination.  Pushes full transcript + BANT
    qualification data to HubSpot or Salesforce.
    """
    logger.info(f"CRM sync: {request.prospect_name} @ {request.company_name}")

    if settings.crm_provider == "hubspot":
        result = await _sync_hubspot(request)
    elif settings.crm_provider == "salesforce":
        result = await _sync_salesforce(request)
    else:
        result = LeadSyncResult(
            success=False,
            crm_provider=settings.crm_provider,
            error=f"Unknown CRM provider: {settings.crm_provider}",
        )

    return result
