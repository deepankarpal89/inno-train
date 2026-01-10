"""
TrainingJob model - Individual training jobs linked to projects
"""

import uuid
import enum
from sqlalchemy import Column, String, Float, JSON, Enum, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.database import Base


class TrainingJobStatus(enum.Enum):
    """Status enum for TrainingJob"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    """
    Training Job Table - Top-level entity for training jobs
    """

    __tablename__ = "training_job"

    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    started_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    completed_at = Column(String(32), nullable=True)  # IST timestamp as ISO string

    # Project and configuration fields
    project_id = Column(String(255), nullable=False)
    training_run_id = Column(String(255), nullable=False, index=True)
    project_yaml_config = Column(JSON, nullable=True)
    training_request = Column(JSON, nullable=True)  # Training request
    machine_config = Column(JSON, nullable=True)  # Machine/GPU configuration
    status = Column(Enum(TrainingJobStatus), default=TrainingJobStatus.PENDING)
    time_taken = Column(Float, nullable=True)  # Time taken in seconds

    # Additional metadata
    job_metadata = Column(JSON, nullable=True)  # Additional metadata

    # Relationships
    iterations = relationship("TrainingIteration", back_populates="training_job")

    def __str__(self):
        return f"TrainingJob({self.project_id}-{self.training_run_id})"


# Create an index on training_run_id for faster lookups
Index("idx_training_job_run_id", TrainingJob.training_run_id)
