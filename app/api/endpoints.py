from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

from app.services.training_workflow import TrainingWorkflow
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration
from models.epoch_train import EpochTrain
from models.eval import Eval

router = APIRouter()


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
    project_id: str
    training_run_id: str
    created_at: Optional[str]
    updated_at: Optional[str]
    completed_at: Optional[str]
    time_taken: Optional[int]
    machine_config: Optional[Dict[str, Any]]
    training_config: Optional[Dict[str, Any]]


class IterationResponse(BaseModel):
    """Response model for training iteration"""

    id: str
    iteration_number: int
    step_type: str
    created_at: Optional[str]
    completed_at: Optional[str]
    step_time: Optional[float]
    step_config: Optional[Dict[str, Any]]


class JobDetailResponse(BaseModel):
    """Detailed response model for job with iterations"""

    job_uuid: str
    status: str
    project_id: str
    training_run_id: str
    created_at: Optional[str]
    updated_at: Optional[str]
    completed_at: Optional[str]
    time_taken: Optional[int]
    machine_config: Optional[Dict[str, Any]]
    training_config: Optional[Dict[str, Any]]
    iterations: List[IterationResponse]


async def _run_training_job_background(
    request_data: Dict[str, Any], job_uuid: str
    ) -> str:
    """Run complete training job in background.

    Args:
        request_data: Training job request data
        job_uuid: Pre-created job UUID to use

    Returns:
        str: Job UUID

    Raises:
        Exception: If training fails
    """
    orchestrator = TrainingWorkflow(logger=logger)

    # Set the pre-created job UUID in the orchestrator
    orchestrator.job = await TrainingJob.get(uuid=job_uuid)

    try:
        # Run complete training workflow (skip job initialization since it's already done)
        completed_job_uuid = await orchestrator.run_complete_training(request_data)
        logger.info(f"✅ Training job {completed_job_uuid} completed successfully")
        return completed_job_uuid

    except Exception as e:
        logger.error(f"Training job {job_uuid} failed: {str(e)}")
        # The orchestrator handles job status updates in _handle_failure
        raise


@router.get("/hello")
async def hello_world():
    """
    A simple Hello World endpoint.
    Returns a welcome message.
    """
    return {"message": "Hello, World from InnoTrain API!"}


@router.post("/v1/training/start", response_model=TrainingResponse)
async def start_training_job(
    request: TrainingRequest, background_tasks: BackgroundTasks
    ):
    """
    Start a new GPU training job.

    This endpoint:
    1. Creates a database record for the training job
    2. Starts the complete training workflow in background
    3. Returns immediately with job UUID
    4. Training runs asynchronously until completion

    Use the job UUID to check status and get results.

    Example request body (matches train_request.json format):
    ```json
    {
        "data": {
            "request_data": {
                "training_run_id": "7b1be4c4-084d-46d7-948d-12b04b26b049",
                "project": {
                    "id": "7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd",
                    "name": "spam local",
                    "description": "testing spam on local",
                    "task_type": "text_classification"
                },
                "prompt": {...},
                "train_dataset": {...},
                "eval_dataset": {...}
            }
        }
    }
    ```
    """
    try:
        # Initialize job record first
        orchestrator = TrainingWorkflow(logger=logger)
        job_uuid = await orchestrator._initialize_job(request.model_dump())

        # Start training in background using FastAPI BackgroundTasks
        background_tasks.add_task(
            _run_training_job_background, request.model_dump(), job_uuid
        )

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
async def get_job_status(job_uuid: str):
    """
    Get the status of a training job.

    Returns current status, timestamps, and configuration.
    """
    try:
        job = await TrainingJob.get(uuid=job_uuid)

        return JobStatusResponse(
            job_uuid=str(job.uuid),
            status=job.status.value
        )

    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")


@router.post("/v1/training/jobs/{job_uuid}/cancel")
async def cancel_training_job(job_uuid: str):
    """
    Cancel a running training job.

    This will:
    1. Attempt to cancel the job using the TrainingWorkflow
    2. Mark the job as cancelled in the database if successful
    """
    try:
        # Create an instance of TrainingWorkflow
        workflow = TrainingWorkflow(logger=logger)

        # Attempt to cancel the training job
        cancellation_successful = await workflow.cancel_training(job_uuid)

        if not cancellation_successful:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job.status.value}",
            )

        logger.info(f"Training job {job_uuid} marked as cancelled")

        return {
            "success": True,
            "message": f"Training job {job_uuid} cancelled successfully",
            "job_uuid": job_uuid,
            "status": "cancelled",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")

