import asyncio
from typing import Dict, Any, Optional, Optional as OptionalType
from models.training_iteration import TrainingIteration, StepType
from models.training_job import TrainingJob
from models.eval import Eval
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from app.database import async_session_maker
from scripts.file_logger import get_file_logger
import logging


class AccuracyMetricsCalculator:
    def __init__(self, training_job_uuid: str, logger: Optional[logging.Logger] = None):
        self.training_job_uuid = training_job_uuid
        self.data: Dict[int, Dict[str, Any]] = {}
        self.logger = logger or get_file_logger(f"accuracy_metrics_{training_job_uuid}")

    async def get_training_iterations(self) -> None:
        stmt = select(TrainingIteration).where(
            TrainingIteration.training_job_uuid == self.training_job_uuid,
            TrainingIteration.step_type == StepType.EVALUATION,
        )

        async with async_session_maker() as session:
            result = await session.execute(stmt)
            training_iterations = result.scalars().all()

            for training_iteration in training_iterations:
                if training_iteration.iteration_number not in self.data:
                    self.data[training_iteration.iteration_number] = {}
                self.logger.info(
                    f"Processing iteration: step_type={training_iteration.step_type}, "
                    f"iteration_number={training_iteration.iteration_number}, "
                    f"dataset={training_iteration.step_config['dataset']}"
                )
                self.data[training_iteration.iteration_number][
                    training_iteration.step_config["dataset"]
                ] = {"iteration_uuid": training_iteration.uuid}

    async def get_eval_for_iteration(self, iteration_number: int) -> None:
        iteration_uuid = self.data[iteration_number]["eval"]["iteration_uuid"]
        stmt = select(Eval).where(Eval.iteration_uuid == iteration_uuid)
        async with async_session_maker() as session:
            result = await session.execute(stmt)
            eval = result.scalars().all()

            max_reward = -1.0
            best_eval = None
            for e in eval:
                reward_percentage = e.metrics["answer_reward"]["percentage"]
                self.logger.info(
                    f"Eval: uuid={e.uuid}, model_id={e.model_id}, "
                    f"dataset={e.dataset}, reward={reward_percentage}"
                )

                if reward_percentage > max_reward:
                    max_reward = reward_percentage
                    best_eval = e
                    iteration_number = int(e.model_id.split("_")[1])
            self.logger.info(f"Best run: {best_eval.uuid if best_eval else 'None'}")
            self.logger.info(f"Highest reward: {max_reward}")
            self.data[iteration_number]["best_eval_eval"] = {
                "model_id": best_eval.model_id,
                "accuracy": max_reward,
                "eval_uuid": best_eval.uuid,
            }

        train_iteration_uuid = self.data[iteration_number]["train"]["iteration_uuid"]
        stmt = select(Eval).where(
            Eval.iteration_uuid == train_iteration_uuid,
            Eval.model_id == best_eval.model_id,
        )
        async with async_session_maker() as session:
            result = await session.execute(stmt)
            train_eval = result.scalars().first()
            self.logger.info("Train best run:")
            self.logger.info(
                f"uuid={train_eval.uuid}, model_id={train_eval.model_id}, "
                f"dataset={train_eval.dataset}, "
                f"reward={train_eval.metrics['answer_reward']['percentage']}"
            )
            self.data[iteration_number]["best_eval_train"] = {
                "model_id": train_eval.model_id,
                "accuracy": train_eval.metrics["answer_reward"]["percentage"],
                "eval_uuid": train_eval.uuid,
            }

    async def calculate_overall_best_model(self) -> Dict[str, Any]:
        best_iteration = None
        best_accuracy = -1.0

        for iteration_number, iteration_data in self.data.items():
            if "best_eval_eval" in iteration_data:
                eval_accuracy = iteration_data["best_eval_eval"]["accuracy"]
                self.logger.info(
                    f"Iteration {iteration_number}: eval Accuracy = {eval_accuracy}"
                )

                if eval_accuracy > best_accuracy:
                    best_accuracy = eval_accuracy
                    best_iteration = iteration_number

        self.logger.info(f"Best iteration: {best_iteration}")
        self.logger.info(f"Highest eval accuracy: {best_accuracy}")

        if best_iteration is None:
            self.logger.warning("No valid iterations found with evaluation data")
            return {
            "best_iteration": None,
            "best_eval_eval": None,
            "best_eval_train": None,
            }

        return {
            "best_iteration": best_iteration,
            "best_eval_eval": self.data[best_iteration]["best_eval_eval"],
            "best_eval_train": self.data[best_iteration]["best_eval_train"],
        }

    async def update_training_iteration_metadata(self) -> None:
        stmt = select(TrainingIteration).where(
            TrainingIteration.training_job_uuid == self.training_job_uuid,
            TrainingIteration.step_type == StepType.ITERATION,
        )
        async with async_session_maker() as session:
            result = await session.execute(stmt)
            training_iterations = result.scalars().all()
            for training_iteration in training_iterations:
                iteration_number = training_iteration.iteration_number
                if iteration_number not in self.data:
                    continue
                existing_metadata = training_iteration.iteration_metadata or {}
                existing_metadata["metrics"] = {
                    "best_eval_eval": self.data[iteration_number]["best_eval_eval"],
                    "best_eval_train": self.data[iteration_number]["best_eval_train"],
                }
                training_iteration.iteration_metadata = existing_metadata
                flag_modified(training_iteration, "iteration_metadata")
                self.logger.info(
                    f"Updated iteration {iteration_number} metadata: {training_iteration.iteration_metadata}"
                )
            await session.commit()

    async def update_training_job_metadata(self) -> None:
        best_model_info = await self.calculate_overall_best_model()
        stmt = select(TrainingJob).where(TrainingJob.uuid == self.training_job_uuid)
        async with async_session_maker() as session:
            result = await session.execute(stmt)
            training_job = result.scalars().first()
            existing_metadata = training_job.job_metadata or {}
            existing_metadata["metrics"] = best_model_info
            training_job.job_metadata = existing_metadata
            flag_modified(training_job, "job_metadata")
            await session.commit()

    async def verify_db_content(self) -> None:
        stmt = select(TrainingIteration).where(
            TrainingIteration.training_job_uuid == self.training_job_uuid,
            TrainingIteration.step_type == StepType.ITERATION,
        )
        async with async_session_maker() as session:
            result = await session.execute(stmt)
            iterations = result.scalars().all()

            self.logger.info("=== DATABASE VERIFICATION ===")
            for iteration in iterations:
                self.logger.info(
                    f"Iteration {iteration.iteration_number}: {iteration.iteration_metadata}"
                )
            self.logger.info("=== END VERIFICATION ===")

    async def run(self) -> None:
        await self.get_training_iterations()
        for iteration_number in self.data.keys():
            await self.get_eval_for_iteration(iteration_number)

        await self.update_training_iteration_metadata()
        await self.update_training_job_metadata()


if __name__ == "__main__":
    training_job_uuid = "b02190c9-099d-468b-81d4-da07678e8f95"
    calculator = AccuracyMetricsCalculator(training_job_uuid)
    asyncio.run(calculator.run())
