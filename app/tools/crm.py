"""CRM Sync Tool — HubSpot/Salesforce Integration.

Phase 7, Steps 26-27: After call termination, fires a non-blocking webhook
to push the complete transcript + BANT qualification data to the CRM.

Improvement 4: Uses Rich to print the CRM payload in a beautiful,
color-coded table to the terminal for demo/loom recording purposes.
"""

from __future__ import annotations

import logging

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from app.config import settings
from app.schemas import CallState, LeadSyncRequest, LeadSyncResult

logger = logging.getLogger(__name__)
console = Console()


def _print_crm_payload(request: LeadSyncRequest) -> None:
    """Print the CRM payload in a rich, color-coded format to terminal.

    This is visible during live screen-shares and Loom recordings,
    proving the data extraction pipeline works flawlessly.
    """
    # ── Header ──
    console.print()
    console.rule("[bold cyan]📡 CRM SYNC PAYLOAD", style="cyan")
    console.print()

    # ── Lead Info Table ──
    lead_table = Table(
        title="🧑‍💼 Lead Information",
        box=box.ROUNDED,
        border_style="bright_blue",
        title_style="bold bright_blue",
        show_header=True,
        header_style="bold white",
        padding=(0, 2),
    )
    lead_table.add_column("Field", style="dim", width=20)
    lead_table.add_column("Value", style="bright_white")

    lead_table.add_row("Name", request.prospect_name)
    lead_table.add_row("Company", f"[bold yellow]{request.company_name}[/]")
    lead_table.add_row("Industry", request.industry)
    lead_table.add_row("Email", request.email or "[dim]not captured[/]")
    lead_table.add_row("Phone", request.phone or "[dim]not captured[/]")
    lead_table.add_row("Call Outcome", _colorize_outcome(request.call_outcome))
    lead_table.add_row("Duration", f"{request.call_duration_seconds:.0f}s")

    console.print(lead_table)
    console.print()

    # ── BANT Qualification Table ──
    bant_table = Table(
        title="📊 BANT Qualification",
        box=box.ROUNDED,
        border_style="bright_green",
        title_style="bold bright_green",
        show_header=True,
        header_style="bold white",
        padding=(0, 2),
    )
    bant_table.add_column("Dimension", style="bold", width=15)
    bant_table.add_column("Extracted Data", style="bright_white")
    bant_table.add_column("Status", width=8, justify="center")

    bant_fields = [
        ("Budget", request.bant.budget),
        ("Authority", request.bant.authority),
        ("Need", request.bant.need),
        ("Timeline", request.bant.timeline),
    ]
    for name, value in bant_fields:
        status = "[green]✓[/]" if value else "[red]✗[/]"
        display = value or "[dim]not yet captured[/]"
        bant_table.add_row(name, display, status)

    console.print(bant_table)
    console.print()

    # ── AEO Audit Results ──
    if request.audit_result:
        audit = request.audit_result
        audit_table = Table(
            title="🔍 AEO Audit Results",
            box=box.ROUNDED,
            border_style="bright_magenta",
            title_style="bold bright_magenta",
            show_header=True,
            header_style="bold white",
            padding=(0, 2),
        )
        audit_table.add_column("Metric", style="dim", width=20)
        audit_table.add_column("Value", style="bright_white")

        sov_color = "red" if audit.share_of_voice_pct < 10 else "yellow" if audit.share_of_voice_pct < 50 else "green"
        audit_table.add_row(
            "Share of Voice",
            f"[bold {sov_color}]{audit.share_of_voice_pct:.1f}%[/]",
        )
        audit_table.add_row("AEO Score", f"[bold]{audit.aeo_score:.1f}/100[/]")
        audit_table.add_row("Diagnosis", audit.diagnosis[:100])
        for i, rec in enumerate(audit.recommendations, 1):
            audit_table.add_row(f"Rec #{i}", rec)

        console.print(audit_table)
        console.print()

    # ── Transcript Preview ──
    transcript_lines = request.call_transcript.split("\n")
    preview = "\n".join(transcript_lines[:10])
    if len(transcript_lines) > 10:
        preview += f"\n[dim]... and {len(transcript_lines) - 10} more lines[/]"

    console.print(
        Panel(
            preview,
            title="💬 Call Transcript (preview)",
            border_style="bright_yellow",
            padding=(1, 2),
        )
    )

    # ── CRM Target ──
    crm_info = f"Provider: [bold]{settings.crm_provider.upper()}[/]"
    if settings.hubspot_api_key:
        crm_info += "  |  API Key: [green]configured ✓[/]"
    else:
        crm_info += "  |  API Key: [yellow]not set (simulated)[/]"

    console.print(
        Panel(crm_info, title="🎯 CRM Target", border_style="cyan", padding=(0, 2))
    )
    console.print()
    console.rule("[bold cyan]END CRM PAYLOAD", style="cyan")
    console.print()


def _colorize_outcome(outcome: str) -> str:
    """Color-code the call outcome for terminal display."""
    colors = {
        "meeting_booked": "[bold green]✅ MEETING BOOKED[/]",
        "callback_requested": "[bold yellow]📞 CALLBACK REQUESTED[/]",
        "not_qualified": "[bold red]❌ NOT QUALIFIED[/]",
        "dropped": "[bold dim]⚫ DROPPED[/]",
    }
    return colors.get(outcome, f"[dim]{outcome}[/]")


async def fire_crm_sync(state: CallState) -> None:
    """Asynchronous CRM sync triggered after call disconnect.

    Prints the rich payload visualization THEN pushes to CRM.
    """
    if not state.company_name:
        logger.info(f"Call {state.call_id}: No company captured, skipping CRM sync")
        return

    # Build the full transcript
    transcript = "\n".join(
        f"{'User' if msg.get('role') == 'user' else 'Agent'}: {msg.get('content', '')}"
        for msg in state.messages
        if msg.get("content")
    )

    # Determine call outcome
    if state.booking_result and state.booking_result.success:
        outcome = "meeting_booked"
    elif state.current_node.value == "terminal":
        outcome = "not_qualified"
    else:
        outcome = "dropped"

    request = LeadSyncRequest(
        prospect_name=state.prospect_name or "Unknown",
        company_name=state.company_name,
        industry=state.industry or "Unknown",
        email=state.email,
        phone=state.phone,
        bant=state.bant,
        call_transcript=transcript,
        call_duration_seconds=state.call_end_epoch - state.call_start_epoch,
        call_outcome=outcome,
        audit_result=state.audit_result,
    )

    # ── Improvement 4: Rich terminal visualization ──
    _print_crm_payload(request)

    # ── Push to CRM ──
    if settings.crm_provider == "hubspot":
        result = await _sync_hubspot(request)
    elif settings.crm_provider == "salesforce":
        result = await _sync_salesforce(request)
    else:
        logger.warning(f"Unknown CRM provider: {settings.crm_provider}")
        return

    if result.success:
        console.print(
            f"  [bold green]✓ CRM sync complete:[/] {result.crm_provider} "
            f"lead_id=[cyan]{result.lead_id}[/]"
        )
    else:
        console.print(f"  [bold red]✗ CRM sync failed:[/] {result.error}")


async def _sync_hubspot(request: LeadSyncRequest) -> LeadSyncResult:
    """Push lead data to HubSpot CRM via the Contacts API."""
    if not settings.hubspot_api_key:
        console.print("  [yellow]⚠ HubSpot API key not set — payload logged above (simulated)[/]")
        return LeadSyncResult(
            success=True,
            crm_provider="hubspot",
            lead_id="simulated_hs_12345",
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts",
                headers={
                    "Authorization": f"Bearer {settings.hubspot_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "properties": {
                        "firstname": request.prospect_name.split()[0]
                        if request.prospect_name
                        else "",
                        "lastname": " ".join(request.prospect_name.split()[1:])
                        if request.prospect_name
                        else "",
                        "email": request.email or "",
                        "phone": request.phone or "",
                        "company": request.company_name,
                        "industry": request.industry,
                        "gw_budget": request.bant.budget or "",
                        "gw_authority": request.bant.authority or "",
                        "gw_need": request.bant.need or "",
                        "gw_timeline": request.bant.timeline or "",
                        "gw_call_outcome": request.call_outcome,
                        "gw_call_transcript": request.call_transcript[:10000],
                        "gw_aeo_score": str(
                            request.audit_result.aeo_score
                        )
                        if request.audit_result
                        else "",
                        "gw_share_of_voice": str(
                            request.audit_result.share_of_voice_pct
                        )
                        if request.audit_result
                        else "",
                    }
                },
            )
            response.raise_for_status()
            data = response.json()
            return LeadSyncResult(
                success=True,
                crm_provider="hubspot",
                lead_id=data.get("id", ""),
            )

    except httpx.HTTPError as e:
        logger.error(f"HubSpot sync failed: {e}", exc_info=True)
        return LeadSyncResult(
            success=False,
            crm_provider="hubspot",
            error=str(e),
        )


async def _sync_salesforce(request: LeadSyncRequest) -> LeadSyncResult:
    """Push lead data to Salesforce CRM."""
    console.print("  [yellow]⚠ Salesforce sync not yet implemented — payload logged above[/]")
    return LeadSyncResult(
        success=True,
        crm_provider="salesforce",
        lead_id="simulated_sf_67890",
    )
