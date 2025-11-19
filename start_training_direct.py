#!/usr/bin/env python3
"""
Script to start a training job directly using TrainingWorkflow.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Import database initialization
from app.db import init_db, close_db
from app.services.training_workflow import TrainingWorkflow

# Set up logging with both console and file handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training_workflow.log")],
)
logger = logging.getLogger(__name__)


async def load_training_request(file_path: str) -> dict:
    """Load training request data from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading training request: {e}")
        raise


async def main():
    """Main function to start the training job."""
    try:
        # Load environment variables
        load_dotenv()

        # Initialize database
        logger.info("Initializing database connection...")
        await init_db()
        logger.info("✅ Database connected successfully")

        # Load training request data
        request_file = "app/api/train_request.json"
        request_data = await load_training_request(request_file)

        if not request_data:
            logger.error("No valid training request data found")
            return

        logger.info("Starting training workflow...")
        workflow = TrainingWorkflow()

        # Start the training job (returns immediately with job UUID)
        completed_job_uuid = await workflow.run_complete_training(request_data)
        logger.info(f"✅ Training job started with UUID: {completed_job_uuid}")
        logger.info("Training is running in background with automatic output download")

        # Optional: Wait for completion if you want to block until done
        # Uncomment the lines below if you want to wait
        logger.info("Waiting for training to complete...")

        logger.info(f"✅ Training job {completed_job_uuid} completed successfully")

    except Exception as e:
        logger.error(f"Error in training job: {e}", exc_info=True)
        raise
    finally:
        # Clean up database connection
        logger.info("Closing database connection...")
        await close_db()
        logger.info("✅ Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())
