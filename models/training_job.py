"""
TrainingJob model - Individual training jobs linked to projects
"""

from enum import Enum
from tortoise import fields
from tortoise.models import Model


class TrainingJobStatus(str, Enum):
    """Status enum for TrainingJob"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Model):
    """
    Training Job Table - Top-level entity for training jobs
    """

    uuid = fields.UUIDField(pk=True)
    created_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    started_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    completed_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string

    # Project and configuration fields
    project_id = fields.CharField(max_length=255)
    training_run_id = fields.CharField(max_length=255, index=True)
    project_yaml_config = fields.JSONField(null=True)
    training_request = fields.JSONField(null=True)  # Training request
    machine_config = fields.JSONField(null=True)  # Machine/GPU configuration
    status = fields.CharEnumField(TrainingJobStatus, default=TrainingJobStatus.PENDING)
    time_taken = fields.FloatField(null=True)  # Time taken in seconds

    # Reverse relation to iterations
    iterations: fields.ReverseRelation["TrainingIteration"]
    metadata = fields.JSONField(null=True)  # Additional metadata

    class Meta:
        table = "training_job"

    def __str__(self):
        return f"TrainingJob({self.project_id}-{self.training_run_id})"
