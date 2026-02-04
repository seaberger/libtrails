"""Book cover image endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...config import CALIBRE_LIBRARY_PATH
from ..dependencies import DBConnection

router = APIRouter()


def _find_cover_path(calibre_id: int) -> Path | None:
    """Find cover image path in Calibre library."""
    # Calibre stores books in: Author Name/Title (ID)/cover.jpg
    # We need to find the directory with the matching ID

    for author_dir in CALIBRE_LIBRARY_PATH.iterdir():
        if not author_dir.is_dir() or author_dir.name.startswith("."):
            continue

        for book_dir in author_dir.iterdir():
            if not book_dir.is_dir():
                continue

            # Check if directory name ends with (ID)
            if book_dir.name.endswith(f"({calibre_id})"):
                cover_path = book_dir / "cover.jpg"
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
        headers={"Cache-Control": "public, max-age=86400"},  # Cache for 1 day
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
