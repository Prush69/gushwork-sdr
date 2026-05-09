"""Static file serving for the web widget."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["widget"])


@router.get("/widget")
async def serve_widget():
    """Serve the web widget HTML page."""
    return FileResponse("widget/index.html", media_type="text/html")
