"""FastAPI application factory and configuration."""

import logging
import sqlite3
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import IPAD_DB_PATH
from ..database import init_chunks_table
from ..embeddings import embed_text
from .routers import books, communities, covers, domains, search, themes, universe

logger = logging.getLogger(__name__)


def _warmup_database() -> None:
    """Run representative queries to warm the OS file cache for key DB pages.

    The demo DB is ~900 MB but the server only has ~900 MB RAM. Without this,
    the first user request pages in FTS5 and sqlite-vec indexes from disk,
    causing 10-15s latency. This warmup loads those index pages into the OS
    buffer cache so real requests are fast.
    """
    import sqlite_vec

    from ..hybrid_search import hybrid_search_books

    conn = sqlite3.connect(IPAD_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    try:
        t0 = time.monotonic()
        # One hybrid search touches all hot tables: FTS5 indexes (books, topics,
        # chunks), sqlite-vec indexes (topic_vectors, book_vectors, chunk_vectors,
        # book_theme_vectors), and the key join tables (chunk_topic_links, etc.)
        hybrid_search_books(conn, "philosophy history science", limit=5)
        elapsed = time.monotonic() - t0
        logger.info("Database warmup complete in %.1fs", elapsed)
    except Exception:
        logger.warning("Database warmup failed (non-fatal)", exc_info=True)
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure schema is up-to-date and warm up models and DB on startup."""
    init_chunks_table()
    logger.info("Warming up embedding model...")
    embed_text("warmup")
    logger.info("Embedding model ready.")
    logger.info("Warming up database cache...")
    _warmup_database()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LibTrails API",
        description="API for browsing book library by themes and topics",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4321", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(domains.router, prefix="/api/v1", tags=["domains"])
    app.include_router(communities.router, prefix="/api/v1", tags=["communities"])
    app.include_router(themes.router, prefix="/api/v1", tags=["themes"])
    app.include_router(books.router, prefix="/api/v1", tags=["books"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])
    app.include_router(covers.router, prefix="/api/v1", tags=["covers"])
    app.include_router(universe.router, prefix="/api/v1", tags=["universe"])

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    return app
