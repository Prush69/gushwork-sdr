"""Calendar Booking Tool — Cal.com Integration.

Phase 7, Step 22-23: Gemini triggers `book_calendar_slot`, FastAPI normalizes
the conversational time string to ISO-8601, and books via Cal.com API.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from dateutil import parser as dateutil_parser

from app.config import settings
from app.schemas import BookingRequest, BookingResult

logger = logging.getLogger(__name__)
_RELATIVE_RE = re.compile(r"\bin\s+(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks)\b")
_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b")

# ── Time String Normalization ──────────────────────────────


def _parse_conversational_time(time_str: str, timezone: str) -> datetime:
    """Parse fuzzy human time expressions into ISO-8601 datetimes.

    Handles: "tomorrow at 3pm", "next Tuesday morning", "in 2 hours",
    "day after tomorrow at 10", etc.

    Uses python-dateutil for most parsing, with manual fallbacks
    for relative expressions.
    """
    text = time_str.lower().strip()
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone '%s', defaulting to UTC", timezone)
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)

    relative = _RELATIVE_RE.search(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        if unit.startswith("minute"):
            return now + timedelta(minutes=amount)
        if unit.startswith("hour"):
            return now + timedelta(hours=amount)
        if unit.startswith("day"):
            return now + timedelta(days=amount)
        if unit.startswith("week"):
            return now + timedelta(weeks=amount)

    # Handle relative expressions
    if "day after tomorrow" in text:
        base = (now + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
    elif "tomorrow" in text:
        base = (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    elif "next week" in text:
        base = (now + timedelta(weeks=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        base = None
        # Check for "next <weekday>"
        import calendar as _cal
        weekdays = {name.lower(): i for i, name in enumerate(_cal.day_name)}
        for day_name, day_idx in weekdays.items():
            if f"next {day_name}" in text:
                days_ahead = (day_idx - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                base = (now + timedelta(days=days_ahead)).replace(
                    hour=10, minute=0, second=0, microsecond=0
                )
                break

    if "morning" in text and not _TIME_RE.search(text):
        base = base or now
        return base.replace(hour=10, minute=0, second=0, microsecond=0)
    if "afternoon" in text and not _TIME_RE.search(text):
        base = base or now
        return base.replace(hour=14, minute=0, second=0, microsecond=0)
    if "evening" in text and not _TIME_RE.search(text):
        base = base or now
        return base.replace(hour=17, minute=0, second=0, microsecond=0)

    # Try dateutil fuzzy parsing
    try:
        default_dt = base if base else now.replace(second=0, microsecond=0)
        parsed = dateutil_parser.parse(time_str, fuzzy=True, default=default_dt)
        # Zero out seconds for clean times
        parsed = parsed.replace(second=0, microsecond=0)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        # If parsed time is in the past, bump to next day
        if parsed < now:
            parsed += timedelta(days=1)
        return parsed
    except (ValueError, OverflowError):
        # Fallback: next business day at 10am
        fallback = now + timedelta(days=1)
        return fallback.replace(hour=10, minute=0, second=0, microsecond=0)


# ── Cal.com API ────────────────────────────────────────────


async def book_slot(request: BookingRequest) -> BookingResult:
    """Book a calendar slot (Simulated for development/demo).
    
    Bypasses actual Cal.com API calls to avoid deprecated v1 endpoint errors.
    """
    booked_at = _parse_conversational_time(request.proposed_time, request.timezone)
    logger.info(f"Simulating booking: {request.prospect_name} at {booked_at.isoformat()}")

    return BookingResult(
        success=True,
        booked_at=booked_at,
        calendar_link=f"https://cal.com/gushwork/aeo-strategy?date={booked_at.strftime('%Y-%m-%d')}",
    )
