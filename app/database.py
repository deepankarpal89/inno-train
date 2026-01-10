"""
SQLAlchemy database configuration and utilities.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Get database URL from settings
DATABASE_URL = settings.db_url

# Log database connection info
print(f"[DATABASE] Using {settings.db_type} database: {DATABASE_URL}")

# Create engine with async support
engine = create_async_engine(
    DATABASE_URL,
    echo=settings.debug,  # Log SQL statements if in debug mode
    future=True,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncSession:
    """Dependency for FastAPI to get a database session.

    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize the database, creating tables if they don't exist."""
    logger.info("Initializing database...")

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")


async def close_db():
    """Close database connections."""
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed")
