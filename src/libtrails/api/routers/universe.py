"""Universe (galaxy visualization) API endpoint."""

import json

from fastapi import APIRouter, HTTPException

from ...config import UNIVERSE_JSON_PATH
from ..schemas import UniverseData

router = APIRouter()


@router.get("/universe", response_model=UniverseData)
def get_universe():
    """Serve pre-generated universe coordinates for the galaxy visualization."""
    if not UNIVERSE_JSON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Universe data not generated yet. Run: uv run libtrails generate-universe",
        )

    with open(UNIVERSE_JSON_PATH) as f:
        data = json.load(f)

    return data
