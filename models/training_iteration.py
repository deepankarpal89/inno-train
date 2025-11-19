"""
TrainingIteration model - Individual iterations within a training job
"""

from enum import Enum
from tortoise import fields
from tortoise.models import Model


class StepType(str, Enum):
    """Step type enum for TrainingIteration"""

    TRAJECTORY_GENERATION = "trajectory_generation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ITERATION = "iteration"
    GROUP_ITERATION = "group_iteration"


class TrainingIteration(Model):
    """
    Training Iteration Table - Individual iterations within a training job
    """

    uuid = fields.UUIDField(pk=True)
    created_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    completed_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    # Time taken in minutes (2 decimal places)
    time_taken = fields.FloatField(null=True)

    # Foreign key to TrainingJob
    training_job = fields.ForeignKeyField(
        "models.TrainingJob", related_name="iterations", on_delete=fields.CASCADE
    )

    iteration_number = fields.IntField()
    step_type = fields.CharEnumField(StepType, default=StepType.ITERATION)
    step_config = fields.JSONField(null=True)  # Configuration for this step
    metadata = fields.JSONField(null=True)  # Additional metadata

    # Reverse relation to epochs
    epochs: fields.ReverseRelation["EpochTrain"]

    class Meta:
        table = "training_iteration"
        indexes = [("training_job", "iteration_number")]

    def __str__(self):
        return f"Iteration({self.iteration_number})"
