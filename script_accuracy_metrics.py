import asyncio
from models.training_iteration import TrainingIteration, StepType
from models.training_job import TrainingJob
from models.eval import Eval
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import async_session_maker

training_job_uuid = "b02190c9-099d-468b-81d4-da07678e8f95"
data = {}


async def get_training_iterations():

    stmt = select(TrainingIteration).where(
        TrainingIteration.training_job_uuid == training_job_uuid,
        TrainingIteration.step_type == StepType.EVALUATION,
    )

    async with async_session_maker() as session:
        result = await session.execute(stmt)
        training_iterations = result.scalars().all()

        for training_iteration in training_iterations:
            if training_iteration.iteration_number not in data:
                data[training_iteration.iteration_number] = {}
            print(
                training_iteration.step_type,
                training_iteration.iteration_number,
                training_iteration.step_config["dataset"],
            )
            data[training_iteration.iteration_number][
                training_iteration.step_config["dataset"]
            ] = {"iteration_uuid": training_iteration.uuid}


async def get_eval_for_iteration(iteration_number):

    iteration_uuid = data[iteration_number]["cv"]["iteration_uuid"]
    stmt = select(Eval).where(Eval.iteration_uuid == iteration_uuid)
    async with async_session_maker() as session:
        result = await session.execute(stmt)
        eval = result.scalars().all()

        max_reward = -1.0
        best_eval = None
        for e in eval:
            reward_percentage = e.metrics["answer_reward"]["percentage"]
            print(
                e.uuid,
                e.model_id,
                e.dataset,
                reward_percentage,
                type(reward_percentage),
            )

            if reward_percentage > max_reward:
                max_reward = reward_percentage
                best_eval = e
                iteration_number = int(e.model_id.split("_")[1])
        print(f"\nBest run: {best_eval.uuid if best_eval else 'None'}")
        print(f"Highest reward: {max_reward}")
        data[iteration_number]["best_eval_cv"] = {
            "model_id": best_eval.model_id,
            "accuracy": max_reward,
            "eval_uuid": best_eval.uuid,
        }

    train_iteration_uuid = data[iteration_number]["train"]["iteration_uuid"]
    stmt = select(Eval).where(
        Eval.iteration_uuid == train_iteration_uuid, Eval.model_id == best_eval.model_id
    )
    async with async_session_maker() as session:
        result = await session.execute(stmt)
        train_eval = result.scalars().first()
        print("train best run")
        print(
            train_eval.uuid,
            train_eval.model_id,
            train_eval.dataset,
            train_eval.metrics["answer_reward"]["percentage"],
        )
        data[iteration_number]["best_eval_train"] = {
            "model_id": train_eval.model_id,
            "accuracy": train_eval.metrics["answer_reward"]["percentage"],
            "eval_uuid": train_eval.uuid,
        }


async def calculate_overall_best_model(data):
    best_iteration = None
    best_accuracy = -1.0

    for iteration_number, iteration_data in data.items():
        if "best_eval_cv" in iteration_data:
            cv_accuracy = iteration_data["best_eval_cv"]["accuracy"]
            print(f"Iteration {iteration_number}: CV Accuracy = {cv_accuracy}")

            if cv_accuracy > best_accuracy:
                best_accuracy = cv_accuracy
                best_iteration = iteration_number

    print(f"\nBest iteration: {best_iteration}")
    print(f"Highest CV accuracy: {best_accuracy}")

    return {
        "best_iteration": best_iteration,
        "best_eval_cv": data[best_iteration]["best_eval_cv"],
        "best_eval_train": data[best_iteration]["best_eval_train"],
    }


async def update_training_iteration_metadata(training_job_uuid):
    from sqlalchemy.orm.attributes import flag_modified

    stmt = select(TrainingIteration).where(
        TrainingIteration.training_job_uuid == training_job_uuid,
        TrainingIteration.step_type == StepType.ITERATION,
    )
    async with async_session_maker() as session:
        result = await session.execute(stmt)
        training_iterations = result.scalars().all()
        for training_iteration in training_iterations:
            iteration_number = training_iteration.iteration_number
            if iteration_number not in data:
                continue
            existing_metadata = training_iteration.iteration_metadata or {}
            existing_metadata["metrics"] = {
                "best_eval_cv": data[iteration_number]["best_eval_cv"],
                "best_eval_train": data[iteration_number]["best_eval_train"],
            }
            training_iteration.iteration_metadata = existing_metadata
            # Explicitly flag the field as modified
            flag_modified(training_iteration, "iteration_metadata")
            print("-----------------Iteration Metadata-----------------------")
            print(training_iteration.iteration_metadata)
        await session.commit()


async def update_training_job_metdata(training_job_uuid):
    from sqlalchemy.orm.attributes import flag_modified

    best_model_info = await calculate_overall_best_model(data)
    stmt = select(TrainingJob).where(TrainingJob.uuid == training_job_uuid)
    async with async_session_maker() as session:
        result = await session.execute(stmt)
        training_job = result.scalars().first()
        existing_metadata = training_job.job_metadata or {}
        existing_metadata["metrics"] = best_model_info
        training_job.job_metadata = existing_metadata
        # Explicitly flag the field as modified
        flag_modified(training_job, "job_metadata")
        await session.commit()


async def verify_db_content(training_job_uuid):
    """Verify the actual content in the database"""
    stmt = select(TrainingIteration).where(
        TrainingIteration.training_job_uuid == training_job_uuid,
        TrainingIteration.step_type == StepType.ITERATION,
    )
    async with async_session_maker() as session:
        result = await session.execute(stmt)
        iterations = result.scalars().all()

        print("\n=== DATABASE VERIFICATION ===")
        for iteration in iterations:
            print(
                f"Iteration {iteration.iteration_number}: {iteration.iteration_metadata}"
            )
        print("=== END VERIFICATION ===\n")


if __name__ == "__main__":
    asyncio.run(get_training_iterations())

    print("-" * 200)
    asyncio.run(get_eval_for_iteration(1))
    asyncio.run(get_eval_for_iteration(2))
    print("-" * 200)
    print(data)
    asyncio.run(update_training_iteration_metadata(training_job_uuid))

    asyncio.run(update_training_job_metdata(training_job_uuid))

    asyncio.run(verify_db_content(training_job_uuid))
