"""FastAPI application factory and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import books, covers, domains, search, themes


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LibTrails API",
        description="API for browsing book library by themes and topics",
        version="0.1.0",
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

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    return app
