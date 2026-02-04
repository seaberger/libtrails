"""FastAPI dependencies for database connections."""

import sqlite3
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends

from ..config import IPAD_DB_PATH


def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a database connection with Row factory."""
    conn = sqlite3.connect(IPAD_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


DBConnection = Annotated[sqlite3.Connection, Depends(get_db_connection)]
