"""
TrainingIteration model - Individual iterations within a training job
"""

import uuid
import enum
from sqlalchemy import Column, String, Float, JSON, Enum, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from app.database import Base


class StepType(enum.Enum):
    """Step type enum for TrainingIteration"""

    PROJECT = "project"
    TRAJECTORY = "trajectory"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ITERATION = "iteration"
    GROUP_ITERATION = "group_iteration"


class TrainingIteration(Base):
    """
    Training Iteration Table - Individual iterations within a training job
    """

    __tablename__ = "training_iteration"

    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    completed_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    time_taken = Column(
        Float, nullable=True
    )  # Time taken in minutes (2 decimal places)

    # Foreign key to TrainingJob
    training_job_uuid = Column(String, ForeignKey("training_job.uuid"), nullable=False)
    training_job = relationship("TrainingJob", back_populates="iterations")

    iteration_number = Column(Integer, nullable=False)
    step_type = Column(Enum(StepType), default=StepType.ITERATION)
    step_config = Column(JSON, nullable=True)  # Configuration for this step
    iteration_metadata = Column(
        JSON, nullable=True
    )  # Additional metadata - renamed from 'metadata'

    # Relationships
    epochs = relationship("EpochTrain", back_populates="iteration")
    evals = relationship("Eval", back_populates="iteration")

    def __str__(self):
        return f"Iteration({self.iteration_number})"


# Create indexes
Index(
    "idx_training_iteration_job_num",
    TrainingIteration.training_job_uuid,
    TrainingIteration.iteration_number,
)
