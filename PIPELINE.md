# Gushwork SDR Pipeline

This project is a FastAPI backend for an inbound voice SDR. Retell handles the
voice transport, STT, and TTS. The backend owns call state, deterministic routing,
Gemini/LangGraph response generation, AEO audit execution, calendar booking, and
CRM sync after the call.

## Step Breakdown

1. Caller starts a browser or phone call.
2. Retell establishes the voice session.
3. Retell sends `call_started` to `/retell/webhook`.
4. Backend creates an in-memory `CallState`.
5. Backend returns the zero-latency greeting.
6. Caller speaks.
7. Retell transcribes the utterance.
8. Retell sends `agent_response_required`.
9. Backend records the transcript segment.
10. Backend extracts cheap known fields, such as email, company, and industry.
11. Backend applies deterministic routing to select the conversation node.
12. Backend builds a compact LLM prompt from system rules, known state, node prompt,
    barge-in context, and recent messages.
13. Gemini generates the next response or emits a tool call.
14. If no tool is needed, backend returns the response to Retell.
15. Retell speaks the response through TTS.
16. If company and industry are known, routing moves to `aeo_audit`.
17. Gemini calls `audit_ai_search`.
18. Backend validates the tool arguments with Pydantic.
19. Backend checks the in-memory audit cache.
20. Backend runs five Gemini audit queries in parallel, each with a timeout.
21. Backend calculates Share of Voice and AEO score.
22. Backend caches the audit result.
23. Gemini summarizes the audit result for the caller.
24. Backend continues the BANT sequence: need, budget, authority, timeline.
25. If the caller raises an objection, routing moves to objection handling.
26. If the caller agrees to book, routing moves to `booking`.
27. Gemini calls `book_calendar_slot`.
28. Backend parses relative or fuzzy time in the caller's timezone.
29. Backend books through Cal.com or simulates booking when no key is configured.
30. Gemini confirms the booking.
31. Retell sends `call_ended`.
32. Backend removes call state and starts CRM sync in a background task.
33. CRM sync builds transcript, BANT fields, audit result, and outcome.
34. Backend pushes to HubSpot or simulates when no key is configured.

## Optimizations Applied

- Package discovery now includes only `app*`, so `pip install .` and Render/Docker
  builds do not fail on the top-level `widget/` directory.
- LLM history is capped by `LLM_CONTEXT_MESSAGES`, while a compact known-state
  summary preserves key facts that older transcript turns might contain.
- The tool-bound Gemini wrapper is cached instead of rebinding tools every turn.
- Retell `call_ended` returns immediately; CRM sync now runs in the background.
- Stale in-memory call states are cleaned by `CALL_STATE_TTL_SECONDS`.
- Cheap transcript extraction moves obvious company, industry, and email fields
  into state before the LLM turn, reducing missed audit transitions.
- Streaming avoids the duplicate stream-then-graph path on tool-first nodes.
- The BANT route now includes the authority step.
- Booking stays in the booking node until a booking result exists.
- Audit calls reuse the Gemini client, honor `GEMINI_MODEL`, apply per-query
  timeouts, select SaaS/eCommerce templates from broader industry phrases, and
  cache repeated company/industry audits.
- Calendar parsing is timezone-aware and handles `day after tomorrow`, `in N
  hours`, and morning/afternoon/evening defaults correctly.

## Remaining Bottlenecks

- In-memory state should move to Redis before multi-instance deployment.
- The widget still needs a real Retell access-token creation flow; otherwise it
  falls back to demo mode.
- External STT, LLM, TTS, Gemini audit, Cal.com, and CRM latency remain outside
  this service's direct control.
