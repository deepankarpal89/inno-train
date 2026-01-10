"""
EpochTrain model - Individual epochs within an iteration
"""

import uuid
from sqlalchemy import Column, String, Float, JSON, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from app.database import Base


class EpochTrain(Base):
    """
    Epoch Train Table - Individual epochs within an iteration
    """

    __tablename__ = "epoch_train"

    uuid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    completed_at = Column(String(32), nullable=True)  # IST timestamp as ISO string
    time_taken = Column(Float, nullable=True)  # Time in minutes (2 decimal places)
    epoch_metadata = Column(
        JSON, nullable=True
    )  # Additional metadata - renamed from 'metadata'

    # Foreign key to TrainingIteration
    iteration_uuid = Column(
        String, ForeignKey("training_iteration.uuid"), nullable=False
    )
    iteration = relationship("TrainingIteration", back_populates="epochs")

    iteration_number = Column(
        Integer, nullable=False
    )  # Denormalized for easier querying
    epoch_number = Column(Integer, nullable=False)
    model_path = Column(String(512), nullable=True)  # Path to saved model
    optimizer_path = Column(String(512), nullable=True)  # Path to optimizer state
    metrics = Column(JSON, nullable=True)  # Training metrics (loss, accuracy, etc.)

    def __str__(self):
        return f"Epoch({self.epoch_number})"


# Create indexes
Index(
    "idx_epoch_train_iteration_num", EpochTrain.iteration_uuid, EpochTrain.epoch_number
)
