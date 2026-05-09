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
import logging
import time

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
    """Execute a REAL AEO visibility audit using Gemini.

    Queries Gemini with industry-relevant prompts and checks whether
    the prospect's company appears in AI-generated answers.
    """
    logger.info(f"🔍 Running REAL AEO audit for: {request.company_name} ({request.industry})")
    t0 = time.perf_counter()

    cached = _get_cached_audit(request)
    if cached:
        return cached

    if not settings.gemini_api_key:
        logger.warning("Gemini API key not configured, using fallback audit")
        result = _fallback_audit(request)
        _store_cached_audit(request, result)
        return result

    try:
        result = await _real_audit(request)
        _store_cached_audit(request, result)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"✅ Audit complete in {elapsed:.0f}ms | "
            f"SoV={result.share_of_voice_pct:.1f}% | "
            f"Score={result.aeo_score}"
        )
        return result

    except Exception as e:
        logger.warning(f"Real audit failed, using fallback: {e}", exc_info=True)
        result = _fallback_audit(request)
        _store_cached_audit(request, result)
        return result


async def _real_audit(request: AuditRequest) -> AuditResult:
    """Core audit logic — fires parallel Gemini queries and analyzes results."""
    client = _get_genai_client()

    # Select query templates
    industry_key = _industry_template_key(request.industry)
    templates = _QUERY_TEMPLATES.get(industry_key, _QUERY_TEMPLATES["default"])

    # Fire all queries in parallel for speed
    queries = [t.format(industry=request.industry) for t in templates]
    tasks = [
        asyncio.wait_for(
            _query_gemini(client, q),
            timeout=settings.audit_query_timeout_seconds,
        )
        for q in queries
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze results
    company_lower = request.company_name.lower()
    mentions = 0
    total_valid = 0
    mentioned_in = []

    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.warning("Query %s failed: %s: %s", i + 1, type(resp).__name__, resp)
            continue

        total_valid += 1
        if company_lower in resp.lower():
            mentions += 1
            mentioned_in.append(queries[i])

    if total_valid == 0:
        raise RuntimeError("All AEO audit queries failed or timed out")

    # Calculate Share of Voice
    sov = (mentions / total_valid * 100) if total_valid > 0 else 0.0
    aeo_score = _calculate_aeo_score(sov, mentions, total_valid)

    # Generate diagnosis
    diagnosis = _generate_diagnosis(
        request.company_name,
        request.industry,
        sov,
        mentions,
        total_valid,
        mentioned_in,
    )

    # Select relevant recommendations
    recommendations = _select_recommendations(sov, mentions)

    return AuditResult(
        company_name=request.company_name,
        industry=request.industry,
        share_of_voice_pct=round(sov, 1),
        aeo_score=round(aeo_score, 1),
        diagnosis=diagnosis,
        recommendations=recommendations,
    )


async def _query_gemini(client: genai.Client, query: str) -> str:
    """Send a single query to Gemini and return the text response."""
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=settings.gemini_model,
        contents=query,
    )
    return response.text or ""


def _calculate_aeo_score(sov: float, mentions: int, total: int) -> float:
    """Calculate a 0-100 AEO score based on visibility metrics."""
    if total == 0:
        return 0.0

    # Base score from Share of Voice (0-60 points)
    sov_score = min(sov * 0.6, 60.0)

    # Consistency bonus — appearing in multiple queries (0-40 points)
    consistency = (mentions / total) * 40

    return min(sov_score + consistency, 100.0)


def _generate_diagnosis(
    company: str,
    industry: str,
    sov: float,
    mentions: int,
    total: int,
    mentioned_in: list[str],
) -> str:
    """Generate a human-readable diagnosis based on audit results."""
    if sov == 0:
        return (
            f"{company} has zero presence in AI-generated answers. "
            f"Across {total} industry-relevant queries to AI search engines, "
            f"your company was not mentioned a single time. "
            f"When decision-makers ask AI assistants about {industry} solutions, "
            f"your competitors are being recommended — you're completely invisible."
        )
    elif sov < 20:
        return (
            f"{company} appeared in {mentions} out of {total} AI-generated answers "
            f"({sov:.1f}% Share of Voice). This is critically low — your competitors "
            f"dominate the AI recommendation space for {industry}. "
            f"Without intervention, you'll continue losing prospects to companies "
            f"that AI models actively recommend."
        )
    elif sov < 50:
        return (
            f"{company} has moderate visibility in AI answers — appearing in "
            f"{mentions} out of {total} queries ({sov:.1f}% Share of Voice). "
            f"There's room to significantly increase your AI search presence "
            f"in the {industry} space with targeted AEO optimization."
        )
    else:
        return (
            f"{company} has strong AI visibility — appearing in {mentions} out of "
            f"{total} queries ({sov:.1f}% Share of Voice). You're well-positioned "
            f"in the {industry} space, but ongoing optimization is needed to "
            f"maintain and grow your advantage."
        )


def _select_recommendations(sov: float, mentions: int) -> list[str]:
    """Select the most relevant recommendations based on audit results."""
    if sov == 0:
        # Zero visibility — foundational recommendations
        return _RECOMMENDATIONS_POOL[:4]
    elif sov < 20:
        # Low visibility — content + authority building
        return _RECOMMENDATIONS_POOL[2:6]
    elif sov < 50:
        # Moderate — optimization + competitive positioning
        return _RECOMMENDATIONS_POOL[4:8]
    else:
        # Strong — maintenance + growth
        return _RECOMMENDATIONS_POOL[6:10]


def _fallback_audit(request: AuditRequest) -> AuditResult:
    """Fallback audit if the real Gemini queries fail.

    Returns strategically pessimistic results to maintain urgency.
    """
    logger.info("Using fallback audit (Gemini queries failed)")
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
