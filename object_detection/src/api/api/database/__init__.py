from .connection import DbConnection
from .enums import HomographyType, VideoStatus
from .exceptions import VideoNotFoundException

__all__ = [
    # database connection
    "DbConnection",
    "init_db_connection",
    "get_db_connection",
    # enums
    "VideoStatus",
    "HomographyType",
    # exceptions
    "VideoNotFoundException",
]

db_connection = None


def init_db_connection(db_url: str) -> DbConnection:
    """Initialize database connection and store it as a global module variable."""
    global db_connection
    if db_connection is None:
        db_connection = DbConnection(db_url=db_url)
    return get_db_connection()


def get_db_connection() -> DbConnection:
    """Get database connection."""
    assert db_connection is not None, "Database connection was not initialized."
    return db_connection
