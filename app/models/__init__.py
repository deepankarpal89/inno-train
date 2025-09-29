"""
Pydantic models and state management for the InnoTrain application.
"""

from .schemas import TrainRequest, TrainResponse, JobStatus
from .state import (
    active_instances,
    create_job_state,
    update_job_status,
    get_job,
    get_all_jobs,
    delete_job,
)

__all__ = [
    "TrainRequest",
    "TrainResponse",
    "JobStatus",
    "active_instances",
    "create_job_state",
    "update_job_status",
    "get_job",
    "get_all_jobs",
    "delete_job",
]
