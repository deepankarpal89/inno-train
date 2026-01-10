"""
Test script to verify the DatabaseService implementation
"""

import asyncio
import logging
from dotenv import load_dotenv
import uuid
import time
import threading
import concurrent.futures

from app.database import init_db, close_db, async_session_maker
from models.training_job import TrainingJob, TrainingJobStatus
from app.services.db_service import db_service
from sqlalchemy import select
from scripts.utils import ist_now

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_db_service")


async def test_main_thread_db_operations():
    """Test database operations in the main thread"""
    logger.info("Testing database operations in the main thread")

    # Create a job
    job_uuid = str(uuid.uuid4())

    async with async_session_maker() as session:
        # Create job
        job = TrainingJob(
            uuid=job_uuid,
            project_id="test-project",
            training_run_id="test-run",
            status=TrainingJobStatus.PENDING,
            created_at=ist_now(),
        )

        session.add(job)
        await session.commit()
        await session.refresh(job)

    logger.info(f"Created job with UUID: {job_uuid}")

    # Update the job
    async with async_session_maker() as session:
        # Get the job
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        job = result.scalars().first()

        # Update it
        job.status = TrainingJobStatus.RUNNING
        job.started_at = ist_now()
        await session.commit()

    logger.info(f"Updated job status to: {job.status}")

    # Retrieve the job
    async with async_session_maker() as session:
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        retrieved_job = result.scalars().first()
        logger.info(f"Retrieved job status: {retrieved_job.status}")

    # Delete the job
    async with async_session_maker() as session:
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        job_to_delete = result.scalars().first()

        await session.delete(job_to_delete)
        await session.commit()

    logger.info(f"Deleted job with UUID: {job_uuid}")

    return True


def background_thread_function():
    """Function to run in a background thread using async methods in sync context"""
    logger.info("Starting background thread")

    try:
        # Create a job
        job_uuid = str(uuid.uuid4())

        # Run async operations in the background thread
        async def bg_operations():
            # Create job
            async with async_session_maker() as session:
                job = TrainingJob(
                    uuid=job_uuid,
                    project_id="test-project-bg",
                    training_run_id="test-run-bg",
                    status=TrainingJobStatus.PENDING,
                    created_at=ist_now(),
                )
                session.add(job)
                await session.commit()

            logger.info(f"Background thread created job with UUID: {job_uuid}")

            # Update job
            async with async_session_maker() as session:
                stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
                result = await session.execute(stmt)
                job = result.scalars().first()

                job.status = TrainingJobStatus.RUNNING
                job.started_at = ist_now()
                await session.commit()

            logger.info(f"Background thread updated job status")

            # Retrieve job
            async with async_session_maker() as session:
                stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
                result = await session.execute(stmt)
                retrieved_job = result.scalars().first()
                logger.info(
                    f"Background thread retrieved job status: {retrieved_job.status.value}"
                )

            # Delete job
            async with async_session_maker() as session:
                stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
                result = await session.execute(stmt)
                job_to_delete = result.scalars().first()

                await session.delete(job_to_delete)
                await session.commit()

            logger.info(f"Background thread deleted job with UUID: {job_uuid}")
            return True

        # Run the async operations in this thread's event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bg_operations())
        loop.close()

        logger.info("Background thread completed successfully")
    except Exception as e:
        logger.error(f"Error in background thread: {e}")


async def main():
    """Main test function"""
    try:
        # Initialize database
        logger.info("Initializing database")
        await init_db()

        # No need to set main event loop with SQLAlchemy
        logger.info("Using SQLAlchemy for database operations")

        # Test database operations in the main thread
        await test_main_thread_db_operations()

        # Test database operations in a background thread
        bg_thread = threading.Thread(target=background_thread_function)
        bg_thread.start()
        bg_thread.join()  # Wait for the background thread to complete

        logger.info("All tests completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Close database connections
        await close_db()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
