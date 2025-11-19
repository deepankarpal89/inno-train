"""
EpochTrain model - Individual epochs within an iteration
"""

from tortoise import fields
from tortoise.models import Model


class EpochTrain(Model):
    """
    Epoch Train Table - Individual epochs within an iteration
    """

    uuid = fields.UUIDField(pk=True)
    created_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    completed_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    time_taken = fields.FloatField(null=True)  # Time in minutes (2 decimal places)
    metadata = fields.JSONField(null=True)  # Additional metadata

    # Foreign key to TrainingIteration
    iteration = fields.ForeignKeyField(
        "models.TrainingIteration", related_name="epochs", on_delete=fields.CASCADE
    )

    iteration_number = fields.IntField()  # Denormalized for easier querying
    epoch_number = fields.IntField()
    model_path = fields.CharField(max_length=512, null=True)  # Path to saved model
    optimizer_path = fields.CharField(
        max_length=512, null=True
    )  # Path to optimizer state
    metrics = fields.JSONField(null=True)  # Training metrics (loss, accuracy, etc.)

    class Meta:
        table = "epoch_train"
        indexes = [("iteration", "epoch_number")]

    def __str__(self):
        return f"Epoch({self.epoch_number})"
