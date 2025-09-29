from typing import Dict, Any, Optional
import uuid
from datetime import datetime

# In-memory storage for simulation state
active_instances: Dict[str, Dict[str, Any]] = {}


def create_job_state(job_name: str, instance_type: str, region: str) -> Dict[str, Any]:
    """Create a new job state entry"""
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    active_instances[job_id] = {
        "job_name": job_name,
        "instance_type": instance_type,
        "region": region,
        "status": "initializing",
        "started_at": datetime.now().isoformat(),
        "instance_id": None,
        "docker_result": None,
        "error": None,
        "completed_at": None,
    }
    return active_instances[job_id]


def update_job_status(job_id: str, **updates) -> Optional[Dict[str, Any]]:
    """Update job status with provided fields"""
    if job_id not in active_instances:
        return None

    active_instances[job_id].update(updates)
    return active_instances[job_id]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job by ID"""
    return active_instances.get(job_id)


def get_all_jobs() -> Dict[str, Dict[str, Any]]:
    """Get all jobs"""
    return active_instances


def delete_job(job_id: str) -> bool:
    """Delete a job by ID"""
    if job_id in active_instances:
        del active_instances[job_id]
        return True
    return False
