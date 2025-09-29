import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.models import (
    TrainRequest,
    TrainResponse,
    JobStatus,
    create_job_state,
    update_job_status,
    get_job,
    get_all_jobs,
    delete_job,
)
from app.services.ec2_simulator import EC2Simulator
from app.services.docker_simulator import DockerSimulator
from app.api.dependencies import get_ec2_simulator, get_docker_simulator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["jobs"])



async def run_training_job(
    job_id: str,
    job_name: str,
    instance_type: str,
    region: str,
    ec2_simulator: EC2Simulator,
    docker_simulator: DockerSimulator,
    ):
    """Main training job workflow"""
    try:
        # Step 1: Create EC2 instance
        update_job_status(job_id, status="creating_instance")
        instance_info = await ec2_simulator.create_instance(instance_type, region)
        update_job_status(
            job_id, status="instance_running", instance_id=instance_info["instance_id"]
        )

        # Step 2: Run Docker hello-world
        update_job_status(job_id, status="running_docker")
        docker_result = await docker_simulator.run_hello_world(
            instance_info["instance_id"]
        )
        update_job_status(job_id, docker_result=docker_result)

        # Step 3: Terminate EC2 instance
        update_job_status(job_id, status="terminating_instance")
        await ec2_simulator.terminate_instance(instance_info["instance_id"])

        # Mark job as completed
        update_job_status(
            job_id, status="completed", completed_at=datetime.now().isoformat()
        )

        logger.info(f"🎉 Training job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Training job {job_id} failed: {str(e)}")
        update_job_status(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now().isoformat(),
        )



@router.post("/train", response_model=TrainResponse)
async def train(
    request: TrainRequest,
    ec2_simulator: EC2Simulator = Depends(get_ec2_simulator),
    docker_simulator: DockerSimulator = Depends(get_docker_simulator),
    ):
    """Start a training job that creates EC2 instance and runs Docker hello-world"""
    # Create job state
    job = create_job_state(request.job_name, request.instance_type, request.region)
    job_id = job["job_id"]

    # Start the training job asynchronously
    asyncio.create_task(
        run_training_job(
            job_id=job_id,
            job_name=request.job_name,
            instance_type=request.instance_type,
            region=request.region,
            ec2_simulator=ec2_simulator,
            docker_simulator=docker_simulator,
        )
    )

    logger.info(f"🚀 Started training job {job_id} ({request.job_name})")

    return TrainResponse(
        job_id=job_id,
        status="initializing",
        message=f"Training job {request.job_name} started successfully",
        instance_id="pending",
        started_at=job["started_at"],
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    return {
        "total_jobs": len(get_all_jobs()),
        "jobs": [
            {
                "job_id": job_id,
                "job_name": job_data["job_name"],
                "status": job_data["status"],
                "started_at": job_data["started_at"],
                "instance_id": job_data.get("instance_id", "pending"),
            }
            for job_id, job_data in get_all_jobs().items()
        ],
    }


@router.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    """Delete a completed job from memory"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    if delete_job(job_id):
        return {"message": f"Job {job_id} deleted successfully"}

    raise HTTPException(status_code=500, detail="Failed to delete job")
