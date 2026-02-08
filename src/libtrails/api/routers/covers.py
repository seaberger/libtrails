"""Book cover image endpoints."""

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...config import CALIBRE_DB_PATH, CALIBRE_LIBRARY_PATH
from ..dependencies import DBConnection

router = APIRouter()


def _find_cover_path(calibre_id: int) -> Path | None:
    """Find cover image path in Calibre library via metadata.db lookup."""
    conn = sqlite3.connect(f"file:{CALIBRE_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT path FROM books WHERE id = ?", (calibre_id,)).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    cover_path = CALIBRE_LIBRARY_PATH / row["path"] / "cover.jpg"
    if cover_path.exists():
        return cover_path

    return None


@router.get("/covers/{calibre_id}")
def get_cover(calibre_id: int):
    """Serve cover image from Calibre library."""
    cover_path = _find_cover_path(calibre_id)

    if not cover_path:
        raise HTTPException(status_code=404, detail="Cover not found")

    return FileResponse(
        cover_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/covers/book/{book_id}")
def get_cover_by_book_id(db: DBConnection, book_id: int):
    """Serve cover image by book ID (looks up calibre_id)."""
    cursor = db.cursor()
    cursor.execute("SELECT calibre_id FROM books WHERE id = ?", (book_id,))
    row = cursor.fetchone()

    if not row or not row["calibre_id"]:
        raise HTTPException(status_code=404, detail="Book not found or no calibre_id")

    return get_cover(row["calibre_id"])
