"""FastAPI application factory and configuration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..embeddings import embed_text
from .routers import books, covers, domains, search, themes, universe

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the embedding model on startup."""
    logger.info("Warming up embedding model...")
    embed_text("warmup")
    logger.info("Embedding model ready.")
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
    app.include_router(themes.router, prefix="/api/v1", tags=["themes"])
    app.include_router(books.router, prefix="/api/v1", tags=["books"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])
    app.include_router(covers.router, prefix="/api/v1", tags=["covers"])
    app.include_router(universe.router, prefix="/api/v1", tags=["universe"])

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    return app
