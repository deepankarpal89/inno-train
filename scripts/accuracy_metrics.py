"""
Accuracy Metrics Calculation

This module provides functions to calculate and cache accuracy metrics
for training iterations. It selects the best performing epoch based on evaluation dataset
performance and stores the results in iteration metadata for efficient retrieval.
"""

from typing import Optional, Dict, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval
from scripts.utils import ist_now


async def get_best_eval_for_iteration(
    evaluation_iteration_uuid: str, session: AsyncSession
) -> Tuple[Optional[float], Optional[int]]:
    """
    Find the eval with maximum accuracy for an evaluation iteration.
    Only considers cv (eval) dataset, not train dataset.

    Args:
        evaluation_iteration_uuid: UUID of the EVALUATION step iteration
        session: Database session

    Returns:
        Tuple of (best_eval_accuracy_percentage, best_epoch_number)
    """
    stmt = select(Eval).where(
        Eval.iteration_uuid == evaluation_iteration_uuid, Eval.dataset == "cv"
    )
    result = await session.execute(stmt)
    evals = result.scalars().all()

    best_eval_accuracy = None
    best_epoch_num = None

    for eval_record in evals:
        if not eval_record.metrics:
            continue

        answer_reward = eval_record.metrics.get("answer_reward", {})
        percentage = answer_reward.get("percentage")

        if percentage is None:
            continue

        model_id = eval_record.model_id or ""
        epoch_num = None
        if "epoch_" in model_id:
            try:
                epoch_num = int(model_id.split("epoch_")[-1])
            except (ValueError, IndexError):
                continue

        if epoch_num and (
            best_eval_accuracy is None or percentage > best_eval_accuracy
        ):
            best_eval_accuracy = percentage
            best_epoch_num = epoch_num

    return best_eval_accuracy, best_epoch_num


async def get_train_accuracy_for_epoch(
    evaluation_iteration_uuid: str, epoch_number: int, session: AsyncSession
) -> Optional[float]:
    """
    Get training accuracy for a specific epoch.
    Note: Train evaluation is also stored in Eval table with dataset='train'.

    Args:
        evaluation_iteration_uuid: UUID of the EVALUATION step iteration
        epoch_number: Epoch number to fetch
        session: Database session

    Returns:
        Training accuracy percentage or None
    """
    stmt = select(Eval).where(
        Eval.iteration_uuid == evaluation_iteration_uuid, Eval.dataset == "train"
    )
    result = await session.execute(stmt)
    evals = result.scalars().all()

    # Find the eval for the specific epoch
    for eval_record in evals:
        if not eval_record.metrics or not eval_record.model_id:
            continue

        # Extract epoch number from model_id
        model_id = eval_record.model_id
        if "epoch_" in model_id:
            try:
                epoch_num = int(model_id.split("epoch_")[-1])
                if epoch_num == epoch_number:
                    answer_reward = eval_record.metrics.get("answer_reward", {})
                    return answer_reward.get("percentage")
            except (ValueError, IndexError):
                continue

    return None


async def get_iteration_steps(
    job_uuid: str, iteration_number: int, session: AsyncSession
) -> Tuple[Optional[TrainingIteration], Optional[TrainingIteration]]:
    """
    Get both ITERATION and EVALUATION steps for a given iteration number.

    Args:
        job_uuid: Training job UUID
        iteration_number: Iteration number
        session: Database session

    Returns:
        Tuple of (training_iteration, evaluation_iteration)
    """
    stmt = select(TrainingIteration).where(
        TrainingIteration.training_job_uuid == job_uuid,
        TrainingIteration.iteration_number == iteration_number,
        TrainingIteration.step_type == StepType.ITERATION,
    )
    result = await session.execute(stmt)
    training_iteration = result.scalars().first()

    stmt = select(TrainingIteration).where(
        TrainingIteration.training_job_uuid == job_uuid,
        TrainingIteration.iteration_number == iteration_number,
        TrainingIteration.step_type == StepType.EVALUATION,
    )
    result = await session.execute(stmt)
    evaluation_iteration = result.scalars().first()

    return training_iteration, evaluation_iteration


async def calculate_best_epoch_metadata(
    training_iteration: TrainingIteration,
    evaluation_iteration: TrainingIteration,
    session: AsyncSession,
) -> Dict:
    """
    Calculate best epoch metrics based on eval performance.

    Args:
        training_iteration: TrainingIteration with ITERATION step_type
        evaluation_iteration: TrainingIteration with EVALUATION step_type
        session: Database session

    Returns:
        Dictionary with best_epoch data
    """
    best_eval_accuracy, best_epoch_num = await get_best_eval_for_iteration(
        evaluation_iteration.uuid, session
    )

    best_train_accuracy = None
    if best_epoch_num:
        best_train_accuracy = await get_train_accuracy_for_epoch(
            evaluation_iteration.uuid, best_epoch_num, session
        )

    best_epoch_data = {
        "epoch_number": best_epoch_num,
        "train_accuracy": best_train_accuracy,
        "eval_accuracy": best_eval_accuracy,
        "calculated_at": ist_now().isoformat(),
    }

    return best_epoch_data


async def cache_best_epoch_in_metadata(
    training_iteration: TrainingIteration, best_epoch_data: Dict, session: AsyncSession
) -> None:
    """
    Cache best epoch data in training iteration metadata.

    Args:
        training_iteration: TrainingIteration to update
        best_epoch_data: Best epoch data dictionary
        session: Database session
    """
    if training_iteration.iteration_metadata is None:
        training_iteration.iteration_metadata = {}

    training_iteration.iteration_metadata["best_epoch"] = best_epoch_data
    session.add(training_iteration)


async def get_or_calculate_best_epoch(
    job_uuid: str, iteration_number: int, session: AsyncSession
) -> Dict:
    """
    Get best epoch data from cache or calculate if not present.

    This is the main entry point for getting best epoch metrics.
    It will use cached data if available, otherwise calculate and cache.

    Args:
        job_uuid: Training job UUID
        iteration_number: Iteration number
        session: Database session

    Returns:
        Dictionary with best_epoch data
    """
    training_iteration, evaluation_iteration = await get_iteration_steps(
        job_uuid, iteration_number, session
    )

    if not training_iteration or not evaluation_iteration:
        return {"epoch_number": None, "train_accuracy": None, "eval_accuracy": None}

    metadata = training_iteration.iteration_metadata or {}
    best_epoch_data = metadata.get("best_epoch")

    if best_epoch_data:
        return best_epoch_data

    best_epoch_data = await calculate_best_epoch_metadata(
        training_iteration, evaluation_iteration, session
    )

    await cache_best_epoch_in_metadata(training_iteration, best_epoch_data, session)

    return best_epoch_data
