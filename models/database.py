"""
Database configuration for Tortoise ORM
"""

from typing import Optional
from tortoise import Tortoise
from app.config import settings


import os
from pathlib import Path

# Ensure database directory exists
db_dir = Path("./")
db_dir.mkdir(exist_ok=True)

# Tortoise ORM configuration
# Using SQLite for development, can switch to PostgreSQL for production
TORTOISE_ORM = {
    "connections": {
        "default": {
            # SQLite configuration (development)
            "engine": "tortoise.backends.sqlite",
            "credentials": {
                "file_path": "./innotrain.db",
                # Add SQLite specific options for better reliability
                "journal_mode": "WAL",
                "synchronous": "NORMAL",
            },
            # PostgreSQL configuration (production) - uncomment when ready
            # "engine": "tortoise.backends.asyncpg",
            # "credentials": {
            #     "host": settings.db_host,
            #     "port": int(settings.db_port),
            #     "user": settings.db_user,
            #     "password": settings.db_password,
            #     "database": settings.db_name,
            # },
        }
    },
    "apps": {
        "models": {
            "models": ["models", "aerich.models"],
            "default_connection": "default",
        }
    },
}


async def init_db() -> None:
    """Initialize database connection with error handling"""
    try:
        await Tortoise.init(config=TORTOISE_ORM)
        await Tortoise.generate_schemas(safe=True)  # Use safe=True to avoid conflicts
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Try to clean up and retry once
        try:
            await Tortoise.close_connections()
        except:
            pass

        # Remove potentially corrupted database files
        db_files = ["./innotrain.db", "./innotrain.db-shm", "./innotrain.db-wal"]
        for db_file in db_files:
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                    print(f"Removed corrupted database file: {db_file}")
                except:
                    pass

        # Retry initialization
        await Tortoise.init(config=TORTOISE_ORM)
        await Tortoise.generate_schemas(safe=True)


async def close_db() -> None:
    """Close database connections"""
    await Tortoise.close_connections()
