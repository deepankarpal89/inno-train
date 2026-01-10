"""
Test script to verify the DatabaseService implementation with Tortoise ORM
"""

import asyncio
import logging
import threading
import uuid
import time
from datetime import datetime

from dotenv import load_dotenv
from models.database import init_db, close_db
from models.training_job import TrainingJob, TrainingJobStatus
from app.services.db_service import db_service
from scripts.utils import ist_now

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_tortoise_db_service")


async def test_main_thread_db_operations():
    """Test database operations in the main thread"""
    logger.info("Testing database operations in the main thread")

    # Create a job
    job_uuid = str(uuid.uuid4())
    job = await db_service.execute(
        TrainingJob.create(
            uuid=job_uuid,
            project_id="test-project",
            training_run_id="test-run",
            status=TrainingJobStatus.PENDING,
            created_at=ist_now(),
        )
    )

    logger.info(f"Created job with UUID: {job_uuid}")

    # Update the job
    job.status = TrainingJobStatus.RUNNING
    job.started_at = ist_now()
    await db_service.execute(job.save())

    logger.info(f"Updated job status to: {job.status}")

    # Retrieve the job
    retrieved_job = await db_service.execute(TrainingJob.get(uuid=job_uuid))
    logger.info(f"Retrieved job status: {retrieved_job.status}")

    # Delete the job
    await db_service.execute(job.delete())
    logger.info(f"Deleted job with UUID: {job_uuid}")

    return True


def background_thread_function():
    """Function to run in a background thread"""
    logger.info("Starting background thread")

    try:
        # Create a job using the synchronous method
        job_uuid = str(uuid.uuid4())
        job = db_service.run_sync(
            TrainingJob.create(
                uuid=job_uuid,
                project_id="test-project-bg",
                training_run_id="test-run-bg",
                status=TrainingJobStatus.PENDING,
                created_at=ist_now(),
            )
        )

        logger.info(f"Background thread created job with UUID: {job_uuid}")

        # Update the job
        job.status = TrainingJobStatus.RUNNING
        job.started_at = ist_now()
        db_service.run_sync(job.save())

        logger.info(f"Background thread updated job status to: {job.status}")

        # Retrieve the job
        retrieved_job = db_service.run_sync(TrainingJob.get(uuid=job_uuid))
        logger.info(f"Background thread retrieved job status: {retrieved_job.status}")

        # Delete the job
        db_service.run_sync(job.delete())
        logger.info(f"Background thread deleted job with UUID: {job_uuid}")

        logger.info("Background thread completed successfully")
    except Exception as e:
        logger.error(f"Error in background thread: {e}")


async def main():
    """Main test function"""
    try:
        # Initialize database
        logger.info("Initializing database")
        await init_db()

        # Set the main event loop in db_service
        db_service.main_loop = asyncio.get_running_loop()
        logger.info("Main event loop set in db_service")

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
