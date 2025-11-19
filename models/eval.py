"""
Eval model - Evaluation results for models
"""

from tortoise import fields
from tortoise.models import Model


class Eval(Model):
    """
    Eval Table - Evaluation results for models
    """

    uuid = fields.UUIDField(pk=True)
    created_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    completed_at = fields.CharField(
        max_length=32, null=True
    )  # IST timestamp as ISO string
    time_taken = fields.FloatField(null=True)  # Time in minutes (2 decimal places)
    iteration = fields.ForeignKeyField(
        "models.TrainingIteration", related_name="evals", on_delete=fields.CASCADE
    )

    model_id = fields.CharField(
        max_length=255, index=True
    )  # Reference to model being evaluated
    dataset = fields.CharField(max_length=255)  # Dataset used for evaluation
    config = fields.JSONField(null=True)  # Evaluation configuration
    metrics = fields.JSONField(null=True)  # Evaluation metrics
    eval_data_path = fields.CharField(
        max_length=512, null=True
    )  # Path to evaluation data
    metadata = fields.JSONField(null=True)  # Additional metadata

    class Meta:
        table = "eval"
        indexes = [("iteration", "model_id")]

    def __str__(self):
        return f"Eval({self.model_id})"
