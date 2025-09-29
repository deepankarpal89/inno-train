from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any


class TrainRequest(BaseModel):
    """Request model for training endpoint"""

    job_name: str = "default-job"
    instance_type: str = "t2.micro"
    region: str = "us-east-1"

    class Config:
        json_schema_extra = {
            "example": {
                "job_name": "test-job",
                "instance_type": "t2.micro",
                "region": "us-east-1",
            }
        }


class TrainResponse(BaseModel):
    """Response model for training endpoint"""

    job_id: str
    status: str
    message: str
    instance_id: str
    started_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-abc12345",
                "status": "initializing",
                "message": "Training job started successfully",
                "instance_id": "i-abc12345",
                "started_at": "2024-01-15T10:30:00",
            }
        }


class JobStatus(BaseModel):
    """Job status response model"""

    job_id: str
    status: str
    job_name: str
    instance_type: str
    region: str
    instance_id: Optional[str]
    started_at: str
    completed_at: Optional[str]
    docker_result: Optional[Dict[str, Any]]
    error: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-abc12345",
                "status": "completed",
                "job_name": "test-job",
                "instance_type": "t2.micro",
                "region": "us-east-1",
                "instance_id": "i-abc12345",
                "started_at": "2024-01-15T10:30:00",
                "completed_at": "2024-01-15T10:30:15",
                "docker_result": {
                    "status": "success",
                    "output": "Hello from Docker!",
                    "container_id": "container-xyz98765",
                    "execution_time": "2.3s",
                },
            }
        }
