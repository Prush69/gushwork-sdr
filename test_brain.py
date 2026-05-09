"""Latency-instrumented brain test.

Shows exact millisecond timings at every stage:
- Field extraction
- Routing decision
- Gemini TTFB (time to first token)
- Gemini total stream time
- Audit execution
- End-to-end turn time
"""

import asyncio
import sys
import os
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.agent.graph import llm_inference_stream
from app.schemas import CallState, ConversationNode
from app.agent.retell_handler import _detect_routing_intent, _extract_known_fields


def ms(t0):
    """Milliseconds since t0."""
    return (time.perf_counter() - t0) * 1000


async def timed_turn(state, user_text, turn_num):
    """Run a single conversation turn with full timing instrumentation."""

    t_turn = time.perf_counter()

    print(f"\n{'=' * 70}")
    print(f"  TURN {turn_num}")
    print(f"{'=' * 70}")
    print(f"  USER: {user_text}")

    # --- Stage 1: Add message ---
    state.messages.append({"role": "user", "content": user_text})

    # --- Stage 2: Field extraction ---
    t0 = time.perf_counter()
    _extract_known_fields(user_text, state)
    extract_ms = ms(t0)

    # --- Stage 3: Routing ---
    t0 = time.perf_counter()
    state.current_node = _detect_routing_intent(user_text, state)
    route_ms = ms(t0)

    print(f"  NODE:    {state.current_node.value}")
    if state.company_name:
        print(f"  COMPANY: {state.company_name}")
    if state.industry:
        print(f"  INDUSTRY: {state.industry}")

    print(f"\n  [TIMING] Field extraction: {extract_ms:.2f}ms")
    print(f"  [TIMING] Routing:          {route_ms:.2f}ms")

    # --- Stage 4: Audit (if triggered) ---
    if (state.current_node == ConversationNode.AEO_AUDIT
            and state.company_name and state.industry):
        print(f"\n  >> AUDIT TRIGGERED")
        t0 = time.perf_counter()

        from app.tools.audit import run_aeo_audit
        from app.schemas import AuditRequest

        try:
            result = await run_aeo_audit(
                AuditRequest(company_name=state.company_name, industry=state.industry)
            )
            audit_ms = ms(t0)
            state.audit_result = result
            state.current_node = ConversationNode.AUDIT_RESULTS
            print(f"  [TIMING] AEO Audit:        {audit_ms:.1f}ms ({audit_ms/1000:.1f}s)")
            print(f"           SoV={result.share_of_voice_pct}% | Score={result.aeo_score}/100")

            state.messages.append({
                "role": "assistant",
                "content": (
                    f"[AUDIT: {result.company_name} has "
                    f"{result.share_of_voice_pct}% Share of Voice. "
                    f"{result.diagnosis}]"
                )
            })
        except Exception as e:
            audit_ms = ms(t0)
            print(f"  [TIMING] AEO Audit FAILED: {audit_ms:.1f}ms - {e}")

    # --- Stage 5: LLM Inference (streaming) ---
    print(f"\n  AGENT: ", end="", flush=True)

    t_llm_start = time.perf_counter()
    ttfb = None
    token_count = 0
    full_response = ""

    try:
        async for token in llm_inference_stream(state):
            if ttfb is None:
                ttfb = ms(t_llm_start)

            if token.startswith("__TOOL_CALL__:"):
                tool_name = token.split(":", 1)[1]
                print(f"\n  >> TOOL CALL: {tool_name}")
                break

            token_count += 1
            full_response += token
            print(token, end="", flush=True)

    except Exception as e:
        print(f"\n  ERROR: {e}")
        full_response = ""

    llm_total_ms = ms(t_llm_start)
    turn_total_ms = ms(t_turn)

    print()  # newline

    # --- Timing Summary ---
    print(f"\n  {'- ' * 35}")
    print(f"  [TIMING] LLM TTFB:         {ttfb:.0f}ms" if ttfb else "  [TIMING] LLM TTFB:         N/A")
    print(f"  [TIMING] LLM total:        {llm_total_ms:.0f}ms")
    print(f"  [TIMING] Tokens streamed:  {token_count}")
    if token_count > 0 and llm_total_ms > 0:
        print(f"  [TIMING] Tokens/sec:       {token_count / (llm_total_ms / 1000):.1f}")
    print(f"  [TIMING] Response length:  {len(full_response)} chars")
    print(f"  [TIMING] TURN TOTAL:       {turn_total_ms:.0f}ms ({turn_total_ms/1000:.1f}s)")
    print(f"  {'- ' * 35}")

    # Save to state
    if full_response:
        state.messages.append({"role": "assistant", "content": full_response})

    return {
        "turn": turn_num,
        "extract_ms": extract_ms,
        "route_ms": route_ms,
        "ttfb_ms": ttfb,
        "llm_total_ms": llm_total_ms,
        "turn_total_ms": turn_total_ms,
        "tokens": token_count,
    }


async def main():
    state = CallState(call_id="perf-test-001", current_node=ConversationNode.GREETING)

    user_turns = [
        "Hi, I'm looking to generate more inbound leads for my company.",
        "We're TechNova, in the B2B SaaS space.",
        "What does that mean for us?",
        "Can we set up a meeting?",
        "Tomorrow at 3pm works.",
    ]

    print("\n" + "#" * 70)
    print("  GUSHWORK SDR - LATENCY BENCHMARK")
    print(f"  Model: {os.environ.get('GEMINI_MODEL', 'unknown')}")
    print("#" * 70)

    all_timings = []
    for i, text in enumerate(user_turns, 1):
        timings = await timed_turn(state, text, i)
        all_timings.append(timings)

    # --- Final Summary ---
    print(f"\n\n{'#' * 70}")
    print("  LATENCY SUMMARY")
    print(f"{'#' * 70}")
    print(f"\n  {'Turn':<6} {'Extract':<10} {'Route':<10} {'TTFB':<10} {'LLM Total':<12} {'Turn Total':<12} {'Tokens':<8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")

    for t in all_timings:
        ttfb_str = f"{t['ttfb_ms']:.0f}ms" if t['ttfb_ms'] else "N/A"
        print(
            f"  {t['turn']:<6} "
            f"{t['extract_ms']:.1f}ms{'':<4} "
            f"{t['route_ms']:.1f}ms{'':<4} "
            f"{ttfb_str:<10} "
            f"{t['llm_total_ms']:.0f}ms{'':<6} "
            f"{t['turn_total_ms']:.0f}ms{'':<6} "
            f"{t['tokens']:<8}"
        )

    avg_ttfb = [t['ttfb_ms'] for t in all_timings if t['ttfb_ms']]
    avg_total = [t['turn_total_ms'] for t in all_timings]

    if avg_ttfb:
        print(f"\n  AVG TTFB:       {sum(avg_ttfb)/len(avg_ttfb):.0f}ms")
    print(f"  AVG Turn Total: {sum(avg_total)/len(avg_total):.0f}ms")
    print(f"  Total Time:     {sum(avg_total):.0f}ms ({sum(avg_total)/1000:.1f}s)")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
