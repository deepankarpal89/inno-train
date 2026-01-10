"""
Eval model - Evaluation results for models
"""

import uuid
from sqlalchemy import Column, String, Float, JSON, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from app.database import Base


class Eval(Base):
    """
    Eval Table - Evaluation results for models
    """

    __tablename__ = "eval"

    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    completed_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    time_taken = Column(Float, nullable=True)  # Time in minutes (2 decimal places)

    # Foreign key to TrainingIteration
    iteration_uuid = Column(
        String, ForeignKey("training_iteration.uuid"), nullable=False
    )
    iteration = relationship("TrainingIteration", back_populates="evals")

    model_id = Column(String(255), index=True)  # Reference to model being evaluated
    dataset = Column(String(255), nullable=False)  # Dataset used for evaluation
    config = Column(JSON, nullable=True)  # Evaluation configuration
    metrics = Column(JSON, nullable=True)  # Evaluation metrics
    eval_data_path = Column(String(512), nullable=True)  # Path to evaluation data
    eval_metadata = Column(
        JSON, nullable=True
    )  # Additional metadata - renamed from 'metadata'

    def __str__(self):
        return f"Eval({self.model_id})"


# Create indexes
Index("idx_eval_iteration_model", Eval.iteration_uuid, Eval.model_id)
