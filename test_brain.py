"""Direct test script for the text-to-text brain.

Simulates what Retell sends us (text transcript) and shows
what we send back (text response). No voice, no WebSocket.

Usage:
    python test_brain.py
"""

import asyncio
import sys
import os

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure app is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agent.graph import llm_inference_stream
from app.schemas import CallState, ConversationNode
from app.agent.retell_handler import _detect_routing_intent, _extract_known_fields


async def simulate_conversation():
    """Simulate a full call as text-to-text, no voice."""

    state = CallState(
        call_id="test-001",
        current_node=ConversationNode.GREETING,
    )

    user_turns = [
        "Hi, I'm looking to generate more inbound leads for my company.",
        "Yeah, my company is called TechNova and we're in the B2B SaaS space.",
        "That's interesting, what does that mean for us exactly?",
        "Yeah I'd love to learn more, can we set up a meeting?",
        "How about tomorrow at 3pm?",
    ]

    print("\n" + "=" * 60)
    print("  GUSHWORK SDR - TEXT-TO-TEXT BRAIN TEST")
    print("=" * 60)

    for i, user_text in enumerate(user_turns):
        print(f"\n{'-' * 60}")
        print(f"  TURN {i + 1}")
        print(f"{'-' * 60}")
        print(f"\n  USER: {user_text}")

        # Add user message to state
        state.messages.append({"role": "user", "content": user_text})

        # Extract fields (like the real handler does)
        _extract_known_fields(user_text, state)

        # Route to correct node
        state.current_node = _detect_routing_intent(user_text, state)

        print(f"  NODE: {state.current_node.value}")

        if state.company_name:
            print(f"  Company: {state.company_name}")
        if state.industry:
            print(f"  Industry: {state.industry}")

        # Check if this is an audit trigger
        if (state.current_node == ConversationNode.AEO_AUDIT
                and state.company_name and state.industry):
            print(f"\n  >> AUDIT TRIGGERED for {state.company_name} ({state.industry})")
            print(f"     Running AEO audit... (5-15 seconds)")

            from app.tools.audit import run_aeo_audit
            from app.schemas import AuditRequest

            try:
                result = await run_aeo_audit(
                    AuditRequest(company_name=state.company_name, industry=state.industry)
                )
                state.audit_result = result
                state.current_node = ConversationNode.AUDIT_RESULTS
                print(f"     SoV: {result.share_of_voice_pct}% | Score: {result.aeo_score}/100")
                print(f"     Diagnosis: {result.diagnosis[:120]}...")

                # Inject audit into messages so Gemini can present it
                state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"[AUDIT RESULT: {result.company_name} has "
                        f"{result.share_of_voice_pct}% Share of Voice. "
                        f"{result.diagnosis}]"
                    )
                })
            except Exception as e:
                print(f"     AUDIT FAILED: {e}")

        # Get Gemini response (streaming)
        print(f"\n  AGENT: ", end="", flush=True)
        full_response = ""
        try:
            async for token in llm_inference_stream(state):
                if token.startswith("__TOOL_CALL__:"):
                    tool_name = token.split(":", 1)[1]
                    print(f"\n  [Tool call: {tool_name}]")
                    break
                print(token, end="", flush=True)
                full_response += token
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            full_response = f"Error: {e}"

        print()  # newline after streaming

        # Save response to state
        if full_response:
            state.messages.append({"role": "assistant", "content": full_response})

    print(f"\n{'=' * 60}")
    print("  FINAL STATE")
    print(f"{'=' * 60}")
    print(f"  Company:  {state.company_name}")
    print(f"  Industry: {state.industry}")
    print(f"  Email:    {state.email}")
    print(f"  Node:     {state.current_node.value}")
    print(f"  Turns:    {len(state.messages)}")
    if state.audit_result:
        print(f"  Audit:    {state.audit_result.share_of_voice_pct}% SoV")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(simulate_conversation())
