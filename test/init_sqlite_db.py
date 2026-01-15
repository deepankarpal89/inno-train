#!/usr/bin/env python3
"""
Script to initialize SQLite database with SQLAlchemy models.
This will create all tables from scratch.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Set environment variables for SQLite
os.environ["DB_TYPE"] = "sqlite"
os.environ["DB_NAME"] = "innotrain.db"

# Import after setting environment variables
from app.database import Base, engine, init_db, close_db
from models.training_job import TrainingJob
from models.training_iteration import TrainingIteration
from models.epoch_train import EpochTrain
from models.eval import Eval

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("init_sqlite_db")


async def create_tables():
    """Create all tables in the database."""
    logger.info("Creating database tables...")

    # Drop all tables first to ensure clean state
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created successfully!")


async def main():
    """Main function to initialize the database."""
    try:
        # Initialize database connection
        await init_db()

        # Create tables
        await create_tables()

        logger.info("SQLite database initialization complete!")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        # Close database connections
        await close_db()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
