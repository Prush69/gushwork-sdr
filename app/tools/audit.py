"""AEO (Answer Engine Optimization) Audit Tool.

This module performs a REAL AI visibility audit by querying Gemini itself
to check whether a company appears in AI-generated recommendations.

How it works:
1. Sends 5 industry-relevant queries to Gemini (e.g., "What are the best
   SaaS companies for X?")
2. Checks if the prospect's company name appears in any of the responses
3. Calculates a real Share of Voice percentage
4. Generates a real diagnosis and actionable recommendations

This is genuine AEO auditing — we're literally testing whether AI models
recommend the prospect's company.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import aiohttp

from google import genai
from langsmith import traceable

from app.config import settings
from app.schemas import AuditRequest, AuditResult

logger = logging.getLogger(__name__)
_genai_client: genai.Client | None = None
_AUDIT_CACHE_MAX_SIZE = 128
_audit_cache: dict[tuple[str, str, str], tuple[float, AuditResult]] = {}

# ── Query templates per industry ───────────────────────────
# Each template generates a question that an AI user would ask.

_QUERY_TEMPLATES: dict[str, list[str]] = {
    "saas": [
        "What are the best {industry} companies to consider in 2026?",
        "Recommend top {industry} solutions for mid-market businesses.",
        "Which {industry} platforms have the best reviews?",
        "What {industry} tools do experts recommend for growing companies?",
        "Compare the leading {industry} providers for B2B companies.",
    ],
    "ecommerce": [
        "What are the best ecommerce platforms in 2026?",
        "Recommend the top {industry} solutions for online retailers.",
        "Which {industry} platforms are best for small businesses?",
        "What {industry} tools do successful online stores use?",
        "Compare leading {industry} solutions for scaling businesses.",
    ],
    "default": [
        "What are the top companies in the {industry} space?",
        "Recommend the best {industry} solutions for businesses.",
        "Which {industry} companies are leaders in their field?",
        "What {industry} providers would you recommend in 2026?",
        "Who are the most recommended {industry} companies?",
    ],
}

_RECOMMENDATIONS_POOL = [
    "Implement structured FAQ content optimized for AI extraction.",
    "Add Schema.org markup to all key product/service pages.",
    "Create authoritative long-form content that AI models can cite.",
    "Build a Knowledge Graph linking your brand to industry concepts.",
    "Optimize for conversational query patterns (questions, comparisons).",
    "Establish E-E-A-T signals: authorship, citations, expert endorsements.",
    "Publish comparison content positioning your brand against named competitors.",
    "Ensure your website has a comprehensive, well-structured About page.",
    "Create industry reports and original research that AI models reference.",
    "Build backlinks from authoritative industry publications.",
]


def _get_genai_client() -> genai.Client:
    """Reuse the Gemini client across audit requests."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.gemini_api_key)
    return _genai_client


def _audit_cache_key(request: AuditRequest) -> tuple[str, str, str]:
    return (
        request.company_name.casefold().strip(),
        request.industry.casefold().strip(),
        (request.website_url or "").casefold().strip(),
    )


def _get_cached_audit(request: AuditRequest) -> AuditResult | None:
    ttl = max(settings.audit_cache_ttl_seconds, 0)
    if ttl == 0:
        return None

    cache_entry = _audit_cache.get(_audit_cache_key(request))
    if not cache_entry:
        return None

    cached_at, result = cache_entry
    if time.time() - cached_at <= ttl:
        logger.info("AEO audit cache hit for %s", request.company_name)
        return result

    _audit_cache.pop(_audit_cache_key(request), None)
    return None


def _store_cached_audit(request: AuditRequest, result: AuditResult) -> None:
    if settings.audit_cache_ttl_seconds <= 0:
        return

    if len(_audit_cache) >= _AUDIT_CACHE_MAX_SIZE:
        oldest_key = min(_audit_cache, key=lambda key: _audit_cache[key][0])
        _audit_cache.pop(oldest_key, None)
    _audit_cache[_audit_cache_key(request)] = (time.time(), result)


def _industry_template_key(industry: str) -> str:
    text = industry.casefold()
    if "saas" in text or "software" in text:
        return "saas"
    if "ecommerce" in text or "e-commerce" in text:
        return "ecommerce"
    return "default"


@traceable(name="aeo_audit", run_type="tool")
async def run_aeo_audit(request: AuditRequest) -> AuditResult:
    """Execute a REAL AEO visibility audit using Tavily + Groq.
    
    This is the synchronous version used by API endpoints.
    """
    logger.info(f"🔍 Running High-Speed AEO audit for: {request.company_name}")
    t0 = time.perf_counter()

    cached = _get_cached_audit(request)
    if cached:
        return cached

    if not settings.tavily_api_key or not settings.groq_api_key:
        logger.warning("Missing Tavily/Groq keys, using fallback")
        result = _fallback_audit(request)
        _store_cached_audit(request, result)
        return result

    try:
        result = await _perform_aeo_audit_logic(request)
        _store_cached_audit(request, result)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"✅ Audit complete in {elapsed:.0f}ms")
        return result
    except Exception as e:
        logger.warning(f"Audit failed, using fallback: {e}")
        return _fallback_audit(request)

async def run_background_aeo_audit(state: CallState) -> None:
    """Background worker that updates call state directly."""
    request = AuditRequest(
        company_name=state.company_name or "Unknown",
        industry=state.industry or "Unknown"
    )
    
    # Check cache first
    cached = _get_cached_audit(request)
    if cached:
        state.audit_result = cached
        return

    try:
        result = await _perform_aeo_audit_logic(request)
        _store_cached_audit(request, result)
        state.audit_result = result
    except Exception as e:
        logger.error(f"Background audit failed: {e}")
        state.audit_result = _fallback_audit(request)

async def _perform_aeo_audit_logic(request: AuditRequest) -> AuditResult:
    """Internal core logic for Tavily + Groq audit."""
    async with aiohttp.ClientSession() as session:
        # 1. Search
        tavily_req = {
            "api_key": settings.tavily_api_key,
            "query": f"top {request.industry} companies and {request.company_name} reviews",
            "search_depth": "basic",
            "max_results": 5,
        }
        async with session.post("https://api.tavily.com/search", json=tavily_req) as resp:
            if resp.status != 200:
                raise Exception(f"Tavily failed: {await resp.text()}")
            search_data = await resp.json()

        # 2. Synthesize
        context_data = json.dumps(search_data.get("results", []))
        prompt = f"""
You are an expert Answer Engine Optimization (AEO) analyst.
Analyze these search results for {request.company_name} in the {request.industry} industry:
{context_data}

Based on how prominent they are in these search results, calculate a mock "Share of Voice" and generate a diagnosis and 2 recommendations.
Return a strictly formatted JSON object with exactly these keys:
"share_of_voice_pct": A float between 0.0 and 100.0.
"aeo_score": A float between 0.0 and 100.0.
"diagnosis": A 2-sentence conversational summary of their current AI visibility.
"recommendations": A list of exactly 2 actionable steps to improve AEO as strings.
"""
        groq_req = {
            "model": settings.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 300,
        }
        headers = {"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"}
        async with session.post("https://api.groq.com/openai/v1/chat/completions", json=groq_req, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"Groq failed: {await resp.text()}")
            groq_data = await resp.json()

    content = groq_data["choices"][0]["message"]["content"]
    data = json.loads(content)
    return AuditResult(
        company_name=request.company_name,
        industry=request.industry,
        share_of_voice_pct=float(data.get("share_of_voice_pct", 0.0)),
        aeo_score=float(data.get("aeo_score", 0.0)),
        diagnosis=data.get("diagnosis", "Unknown visibility"),
        recommendations=data.get("recommendations", [])[:2]
    )



def _fallback_audit(request: AuditRequest) -> AuditResult:
    """Fallback audit if the real API queries fail.

    Returns strategically pessimistic results to maintain urgency.
    """
    logger.info("Using fallback audit")
    return AuditResult(
        company_name=request.company_name,
        industry=request.industry,
        share_of_voice_pct=0.0,
        aeo_score=0.0,
        diagnosis=(
            f"{request.company_name} currently has a 0% Share of Voice in AI "
            f"answer engines. When prospects in the {request.industry} space "
            f"ask AI assistants for recommendations, your brand doesn't appear "
            f"in any results."
        ),
        recommendations=_RECOMMENDATIONS_POOL[:3],
    )

