# Gushwork SDR — Inbound AEO Voice Agent

Sub-500ms latency voice SDR that qualifies inbound leads using BANT methodology and runs **real-time AI visibility audits** on prospects — powered by Gemini 3 Flash and Retell AI.

## Architecture

```
Caller (browser/phone)
    ↕  WebRTC
Retell AI (managed)
    ├── Deepgram STT (managed by Retell)
    ├── ElevenLabs TTS (managed by Retell)
    └── Webhook POST →  Render backend
                           ├── /retell/webhook      → LangGraph → Gemini 3 Flash
                           ├── /retell/webhook/stream → SSE streaming (sub-100ms TTFB)
                           ├── /audit_ai_search      → Real AEO audit via Gemini
                           ├── /book_calendar_slot   → Cal.com API
                           └── /sync_crm             → HubSpot (optional)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Voice Infrastructure | Retell AI (WebRTC + STT + TTS) |
| LLM | Gemini 3 Flash Preview via `langchain-google-genai` |
| State Machine | LangGraph |
| API Framework | FastAPI + SSE Streaming |
| Calendar | Cal.com API |
| CRM | HubSpot API (optional) |
| Observability | LangSmith tracing |
| Deployment | Render |

## Key Features

- **Real AEO Audit** — Queries Gemini with 5 industry-relevant prompts to measure actual Share of Voice
- **SSE Streaming** — Sub-100ms TTFB via token-level streaming to Retell TTS
- **BANT State Machine** — Deterministic conversation routing, no LLM-decided edges
- **Tool Retry Loops** — Self-correcting tool execution with LLM-mediated error recovery
- **LangSmith Tracing** — Full observability across every node and tool call
- **Rich CRM Visualization** — Color-coded terminal output for demo recordings

## Setup

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Fill in your API keys

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RETELL_API_KEY` | ✅ | Retell AI API key |
| `RETELL_AGENT_ID` | ✅ | Retell agent ID (Custom LLM mode) |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `CALCOM_API_KEY` | ✅ | Cal.com API key |
| `CALCOM_EVENT_TYPE_ID` | ✅ | Cal.com event type ID |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing key |
| `HUBSPOT_API_KEY` | Optional | HubSpot CRM key |

## Deploy to Render

1. Push to GitHub
2. Connect repo in Render dashboard
3. Render auto-detects `render.yaml`
4. Set environment variables in Render dashboard
5. Deploy

## Project Structure

```
app/
├── main.py                  # FastAPI entrypoint
├── config.py                # Settings from .env
├── schemas.py               # Pydantic models
├── agent/
│   ├── retell_handler.py    # Webhook handler + SSE streaming
│   ├── graph.py             # LangGraph state machine
│   └── prompts.py           # System prompt + tool definitions
├── routes/
│   ├── retell.py            # POST /retell/webhook[/stream]
│   ├── tools.py             # POST /audit, /book, /sync
│   └── widget.py            # GET /widget
└── tools/
    ├── audit.py             # Real AEO audit via Gemini
    ├── calendar.py          # Cal.com booking
    └── crm.py               # HubSpot sync + Rich terminal viz
```
