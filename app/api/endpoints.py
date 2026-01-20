from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import asyncio
from scripts.utils import ist_now,calculate_duration
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import datetime
from datetime import timezone
import os

from app.database import get_session
from app.services.training_workflow import TrainingWorkflow, JobNotFoundError
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger with the given name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Configure logger if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Create formatter with the specified format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Initialize the logger for this module
logger = setup_logger("APIEndpoints")


class TrainingRequest(BaseModel):
    """Training request model matching train_request.json format"""

    success: Optional[bool] = True
    message: Optional[str] = "Training run started successfully"
    status_code: Optional[int] = 200
    data: Dict[str, Any] = Field(..., description="Training configuration data")


class TrainingResponse(BaseModel):
    """Response model for training job creation"""

    success: bool
    message: str
    job_uuid: str
    status: str


class JobStatusResponse(BaseModel):
    """Response model for job status"""

    job_uuid: str
    status: str
    success: bool


class CancelTrainingResponse(BaseModel):
    """Response model for cancel training job"""

    success: bool
    message: str
    job_uuid: str
    status: str


class TrainingTimeResponse(BaseModel):
    """Response model for training time information"""

    job_uuid: str
    status: str
    elapsed_time: float
    remaining_time: float
    estimated_time: float
    success: bool
    message: str


async def _run_training_job_background(request_data: Dict[str, Any], job_uuid: str) -> str:
    """Run complete training job in background.

    Args:
        request_data: Training job request data
        job_uuid: Pre-created job UUID to use

    Returns:
        str: Job UUID

    Raises:
        Exception: If training fails
    """
    workflow = await TrainingWorkflow.for_existing_job(job_uuid=job_uuid, logger=logger)

    try:
        # Run complete training workflow (skip job initialization since it's already done)
        completed_job_uuid = await workflow.run_complete_training(request_data)
        logger.info(f"✅ Training job {completed_job_uuid} completed successfully")
        return completed_job_uuid

    except Exception as e:
        logger.error(f"Training job {job_uuid} failed: {str(e)}")
        # The workflow handles job status updates in _handle_failure
        raise


def _run_training_job_background_sync(request_data: Dict[str, Any], job_uuid: str) -> str:
    """Run training job in a new event loop in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _run_training_job_background(request_data, job_uuid)
        )
    finally:
        loop.close()


@router.get("/hello")
async def hello_world():
    """
    A simple Hello World endpoint.
    Returns a welcome message.
    """
    return {"message": "Hello, World from InnoTrain API!"}


@router.post("/v1/training/start", response_model=TrainingResponse)
async def start_training_job(request: TrainingRequest, background_tasks: BackgroundTasks):

    try:
        print("request", request)
        # Initialize job record first
        workflow = TrainingWorkflow.for_new_job(logger=logger)
        job_uuid = await workflow._initialize_job(request.model_dump())

        #future = TRAINING_EXECUTOR.submit(_run_training_job_background_sync, request.model_dump(), job_uuid)

        # # Optional: Add callback to handle completion
        # future.add_done_callback(
        #     lambda f: logger.info(f"Job {job_uuid} completed with status {f.result()}")
        # )
        logger.info(f"Training job {job_uuid} queued for background execution")

        return TrainingResponse(
            success=True,
            message="Training job started successfully",
            job_uuid=job_uuid,
            status="pending",
        )

    except Exception as e:
        logger.error(f"Failed to start training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/training/jobs/{job_uuid}", response_model=JobStatusResponse)
async def get_job_status(job_uuid: str, session: AsyncSession = Depends(get_session)):
    try:
        # Use SQLAlchemy to query the job
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
        # Include completed_at if status is completed, failed, or cancelled
        completed_at = None
        if job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.CANCELLED]:
            completed_at = job.completed_at
        return JobStatusResponse(
            job_uuid=str(job.uuid), status=job.status.value, success=True,completed_at=completed_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/training/jobs/{job_uuid}/cancel", response_model=CancelTrainingResponse)
async def cancel_training_job(job_uuid: str, background_tasks: BackgroundTasks, session: AsyncSession = Depends(get_session),) -> CancelTrainingResponse:

    try:
        # Check if job exists first using SQLAlchemy
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            raise JobNotFoundError(f"Job {job_uuid} not found")

        # If job is already in terminal state, just return
        if job.status in [
            TrainingJobStatus.COMPLETED,
            TrainingJobStatus.FAILED,
            TrainingJobStatus.CANCELLED,
        ]:
            return CancelTrainingResponse(
                success=True,
                message=f"Job is already in terminal state: {job.status.value}",
                job_uuid=job_uuid,
                status=job.status.value,
            )

        # Mark job as cancelling immediately
        job.status = TrainingJobStatus.CANCELLED
        session.add(job)
        await session.commit()

        # Run actual cleanup in background
        async def perform_cleanup():
            try:
                workflow = await TrainingWorkflow.for_existing_job(
                    job_uuid=job_uuid, logger=logger
                )
                await workflow._cleanup_resources()
                logger.info(f"🛑 Background cleanup completed for job {job_uuid}")
            except Exception as e:
                logger.error(f"Background cleanup failed for job {job_uuid}: {str(e)}")

        # background_tasks.add_task(perform_cleanup)

        return CancelTrainingResponse(
            success=True,
            message=f"Job {job_uuid} cancellation initiated",
            job_uuid=job_uuid,
            status="cancelling",
        )

    except JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_uuid}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while cancelling the job: {str(e)}",
        )


@router.get("/v1/training/jobs/{job_uuid}/time", response_model=TrainingTimeResponse)
async def get_job_training_time(job_uuid: str, session: AsyncSession = Depends(get_session)):
    """
    Get training time information for a job.

    For running jobs, provides elapsed time and estimated time remaining.
    For completed jobs, provides total time taken.

    Args:
        job_uuid: UUID of the training job
        session: Database session

    Returns:
        TrainingTimeResponse: Training time information
    """
    try:
        # Query the job
        stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
        result = await session.execute(stmt)
        job = result.scalars().first()
        elapsed_time = 0
        estimated_time = 0
        remaining_time = 0
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")

        # Calculate elapsed time
        no_iterations = (job.project_yaml_config or {}).get("no_iterations", 2)
        gpu_setup_time = int(os.getenv('GPU_SETUP_TIME',5))
        task_estimated_time = int(os.getenv('TEXT_CLASSIFICATION_ESTIMATED_TIME',30))
        
        
        estimated_time = no_iterations * task_estimated_time + gpu_setup_time
        remaining_time = estimated_time - elapsed_time
        
        
        if job.status == TrainingJobStatus.PENDING:
            current_time = ist_now()
            elapsed_time = calculate_duration(job.created_at, current_time)
            remaining_time = estimated_time - elapsed_time
        elif job.status == TrainingJobStatus.RUNNING:
            current_time = ist_now()
            elapsed_time = calculate_duration(job.created_at, current_time)

            #check if a single iteration has been completed
            stmt = select(TrainingIteration).where(
                TrainingIteration.training_job_uuid == job_uuid,
                TrainingIteration.step_type == StepType.ITERATION,
                TrainingIteration.completed_at.isnot(None),
                ).order_by(TrainingIteration.iteration_number.desc())
            
            result = await session.execute(stmt)
            latest_iteration = result.scalars().first()

            if latest_iteration and latest_iteration.time_taken:
                #use actual iteration time to estimate remaining time
                completed_iterations = latest_iteration.iteration_number
                remaining_iterations = no_iterations - completed_iterations

                #estimate remaining time based on actual iteration performance
                remaining_time = remaining_iterations * latest_iteration.time_taken + gpu_setup_time
                estimated_time = elapsed_time + remaining_time
            else:
                remaining_time = estimated_time - elapsed_time

        else:
            current_time = ist_now()
            if job.completed_at:
                elapsed_time = calculate_duration(job.created_at, job.completed_at)
            else:
                elapsed_time = calculate_duration(job.created_at, current_time)
            remaining_time = 0
            estimated_time = elapsed_time
        resp = TrainingTimeResponse(
            job_uuid=job_uuid,
            status=job.status,
            elapsed_time=elapsed_time,
            remaining_time=remaining_time,
            estimated_time=estimated_time,
            success=True,
            message="Training time information retrieved successfully"
        )
        print(resp)
        return resp
    except Exception as e:
        print(f"Error Exception: {e}")
        return TrainingTimeResponse(
            job_uuid=job_uuid,
            status=TrainingJobStatus.FAILED,
            elapsed_time=0,
            remaining_time=0,
            estimated_time=0,
            success=False,
            message="Failed to retrieve training time information"
        )


