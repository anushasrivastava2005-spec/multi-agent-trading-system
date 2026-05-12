"""
Database engine and session management.
Uses SQLite with WAL mode for concurrent reads.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from database.models import Base


# Enable WAL mode for better concurrent read performance in SQLite
def _set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64 MB cache
    cursor.close()


engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # SQLite + FastAPI threads
    pool_pre_ping=True,
)

event.listen(engine, "connect", _set_sqlite_pragma)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
