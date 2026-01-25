"""
Test script for accuracy_metrics.py functions

This script tests each function in the accuracy_metrics module with real database data.
"""

import asyncio
import sys
from dotenv import load_dotenv
from sqlalchemy import select

load_dotenv()

from app.database import async_session_maker
from models.training_job import TrainingJob
from models.training_iteration import TrainingIteration, StepType
from scripts.accuracy_metrics import (
    get_best_eval_for_iteration,
    get_train_accuracy_for_epoch,
    get_iteration_steps,
    calculate_best_epoch_metadata,
    cache_best_epoch_in_metadata,
    get_or_calculate_best_epoch,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def test_get_best_eval_for_iteration():
    """Test get_best_eval_for_iteration function."""
    print_section("TEST 1: get_best_eval_for_iteration()")

    async with async_session_maker() as session:
        # Get a sample evaluation iteration
        stmt = (
            select(TrainingIteration)
            .where(TrainingIteration.step_type == StepType.EVALUATION)
            .limit(1)
        )
        result = await session.execute(stmt)
        eval_iteration = result.scalars().first()

        if not eval_iteration:
            print("❌ No evaluation iterations found in database")
            return False

        print(f"📊 Testing with evaluation iteration: {eval_iteration.uuid}")
        print(f"   Iteration number: {eval_iteration.iteration_number}")

        # Test the function
        best_accuracy, best_epoch = await get_best_eval_for_iteration(
            eval_iteration.uuid, session
        )

        print(f"\n✅ Results:")
        print(f"   Best eval accuracy: {best_accuracy}%")
        print(f"   Best epoch number: {best_epoch}")

        return True


async def test_get_train_accuracy_for_epoch():
    """Test get_train_accuracy_for_epoch function."""
    print_section("TEST 2: get_train_accuracy_for_epoch()")

    async with async_session_maker() as session:
        # Get a sample training iteration
        stmt = (
            select(TrainingIteration)
            .where(TrainingIteration.step_type == StepType.ITERATION)
            .limit(1)
        )
        result = await session.execute(stmt)
        train_iteration = result.scalars().first()

        if not train_iteration:
            print("❌ No training iterations found in database")
            return False

        print(f"📊 Testing with training iteration: {train_iteration.uuid}")
        print(f"   Iteration number: {train_iteration.iteration_number}")

        # Test with epoch 1
        epoch_number = 1
        train_accuracy = await get_train_accuracy_for_epoch(
            train_iteration.uuid, epoch_number, session
        )

        print(f"\n✅ Results:")
        print(f"   Training accuracy for epoch {epoch_number}: {train_accuracy}%")

        return True


async def test_get_iteration_steps():
    """Test get_iteration_steps function."""
    print_section("TEST 3: get_iteration_steps()")

    async with async_session_maker() as session:
        # Get a sample job
        stmt = select(TrainingJob).limit(1)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            print("❌ No training jobs found in database")
            return False

        # Get an iteration number from this job
        stmt = (
            select(TrainingIteration)
            .where(
                TrainingIteration.training_job_uuid == job.uuid,
                TrainingIteration.step_type == StepType.ITERATION,
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        iteration = result.scalars().first()

        if not iteration:
            print("❌ No iterations found for this job")
            return False

        print(f"📊 Testing with job: {job.uuid}")
        print(f"   Iteration number: {iteration.iteration_number}")

        # Test the function
        training_iter, evaluation_iter = await get_iteration_steps(
            job.uuid, iteration.iteration_number, session
        )

        print(f"\n✅ Results:")
        print(
            f"   Training iteration UUID: {training_iter.uuid if training_iter else 'None'}"
        )
        print(
            f"   Training iteration step_type: {training_iter.step_type if training_iter else 'None'}"
        )
        print(
            f"   Evaluation iteration UUID: {evaluation_iter.uuid if evaluation_iter else 'None'}"
        )
        print(
            f"   Evaluation iteration step_type: {evaluation_iter.step_type if evaluation_iter else 'None'}"
        )

        return True


async def test_calculate_best_epoch_metadata():
    """Test calculate_best_epoch_metadata function."""
    print_section("TEST 4: calculate_best_epoch_metadata()")

    async with async_session_maker() as session:
        # Get a sample job
        stmt = select(TrainingJob).limit(1)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            print("❌ No training jobs found in database")
            return False

        # Get an iteration
        stmt = (
            select(TrainingIteration)
            .where(
                TrainingIteration.training_job_uuid == job.uuid,
                TrainingIteration.step_type == StepType.ITERATION,
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        iteration = result.scalars().first()

        if not iteration:
            print("❌ No iterations found")
            return False

        print(f"📊 Testing with job: {job.uuid}")
        print(f"   Iteration number: {iteration.iteration_number}")

        # Get both steps
        training_iter, evaluation_iter = await get_iteration_steps(
            job.uuid, iteration.iteration_number, session
        )

        if not training_iter or not evaluation_iter:
            print("❌ Could not find both training and evaluation steps")
            return False

        # Test the function
        metadata = await calculate_best_epoch_metadata(
            training_iter, evaluation_iter, session
        )

        print(f"\n✅ Results:")
        print(f"   Best epoch number: {metadata.get('epoch_number')}")
        print(f"   Train accuracy: {metadata.get('train_accuracy')}%")
        print(f"   Eval accuracy: {metadata.get('eval_accuracy')}%")
        print(f"   Calculated at: {metadata.get('calculated_at')}")

        return True


async def test_cache_best_epoch_in_metadata():
    """Test cache_best_epoch_in_metadata function."""
    print_section("TEST 5: cache_best_epoch_in_metadata()")

    async with async_session_maker() as session:
        # Get a sample training iteration
        stmt = (
            select(TrainingIteration)
            .where(TrainingIteration.step_type == StepType.ITERATION)
            .limit(1)
        )
        result = await session.execute(stmt)
        train_iteration = result.scalars().first()

        if not train_iteration:
            print("❌ No training iterations found")
            return False

        print(f"📊 Testing with training iteration: {train_iteration.uuid}")

        # Create test metadata
        test_metadata = {
            "epoch_number": 2,
            "train_accuracy": 85.5,
            "eval_accuracy": 82.3,
            "calculated_at": "2026-01-24T18:00:00+05:30",
        }

        print(f"   Test metadata: {test_metadata}")

        # Test the function (without committing)
        await cache_best_epoch_in_metadata(train_iteration, test_metadata, session)

        # Check if it was added
        cached_data = (
            train_iteration.iteration_metadata.get("best_epoch")
            if train_iteration.iteration_metadata
            else None
        )

        print(f"\n✅ Results:")
        print(f"   Metadata cached: {cached_data is not None}")
        print(f"   Cached data: {cached_data}")

        # Rollback to not affect database
        await session.rollback()
        print(f"   (Changes rolled back - not committed to database)")

        return True


async def test_get_or_calculate_best_epoch():
    """Test get_or_calculate_best_epoch function (main entry point)."""
    print_section("TEST 6: get_or_calculate_best_epoch() - Main Function")

    async with async_session_maker() as session:
        # Get a sample job
        stmt = select(TrainingJob).limit(1)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            print("❌ No training jobs found in database")
            return False

        # Get an iteration
        stmt = (
            select(TrainingIteration)
            .where(
                TrainingIteration.training_job_uuid == job.uuid,
                TrainingIteration.step_type == StepType.ITERATION,
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        iteration = result.scalars().first()

        if not iteration:
            print("❌ No iterations found")
            return False

        print(f"📊 Testing with job: {job.uuid}")
        print(f"   Iteration number: {iteration.iteration_number}")

        # Check if already cached
        metadata_before = iteration.iteration_metadata or {}
        had_cache = "best_epoch" in metadata_before
        print(f"   Had cached data before: {had_cache}")

        # Test the function
        best_epoch_data = await get_or_calculate_best_epoch(
            job.uuid, iteration.iteration_number, session
        )

        print(f"\n✅ Results:")
        print(f"   Best epoch number: {best_epoch_data.get('epoch_number')}")
        print(f"   Train accuracy: {best_epoch_data.get('train_accuracy')}%")
        print(f"   Eval accuracy: {best_epoch_data.get('eval_accuracy')}%")
        print(f"   Calculated at: {best_epoch_data.get('calculated_at')}")

        if not had_cache:
            print(f"\n   ℹ️  Data was calculated and cached (commit needed)")
            await session.rollback()
            print(f"   (Changes rolled back - not committed to database)")
        else:
            print(f"\n   ℹ️  Data was retrieved from cache (no calculation needed)")

        return True


async def test_full_workflow():
    """Test the complete workflow with multiple iterations."""
    print_section("TEST 7: Full Workflow - Multiple Iterations")

    async with async_session_maker() as session:
        # Get a sample job
        stmt = select(TrainingJob).limit(1)
        result = await session.execute(stmt)
        job = result.scalars().first()

        if not job:
            print("❌ No training jobs found in database")
            return False

        print(f"📊 Testing with job: {job.uuid}")

        # Get all iterations for this job
        stmt = (
            select(TrainingIteration)
            .where(
                TrainingIteration.training_job_uuid == job.uuid,
                TrainingIteration.step_type == StepType.ITERATION,
            )
            .order_by(TrainingIteration.iteration_number)
        )

        result = await session.execute(stmt)
        iterations = result.scalars().all()

        print(f"   Found {len(iterations)} iterations")

        train_accuracies = []
        eval_accuracies = []
        best_epochs = []

        for iteration in iterations:
            best_epoch_data = await get_or_calculate_best_epoch(
                job.uuid, iteration.iteration_number, session
            )

            train_accuracies.append(best_epoch_data.get("train_accuracy"))
            eval_accuracies.append(best_epoch_data.get("eval_accuracy"))
            best_epochs.append(best_epoch_data.get("epoch_number"))

        print(f"\n✅ Results for all iterations:")
        print(f"   Train accuracies: {train_accuracies}")
        print(f"   Eval accuracies: {eval_accuracies}")
        print(f"   Best epochs: {best_epochs}")

        # Rollback to not affect database
        await session.rollback()
        print(f"\n   (Changes rolled back - not committed to database)")

        return True


async def main():
    """Run all tests."""
    print("\n" + "🧪" * 35)
    print("  ACCURACY METRICS TEST SUITE")
    print("🧪" * 35)

    tests = [
        ("get_best_eval_for_iteration", test_get_best_eval_for_iteration),
        ("get_train_accuracy_for_epoch", test_get_train_accuracy_for_epoch),
        ("get_iteration_steps", test_get_iteration_steps),
        ("calculate_best_epoch_metadata", test_calculate_best_epoch_metadata),
        ("cache_best_epoch_in_metadata", test_cache_best_epoch_in_metadata),
        ("get_or_calculate_best_epoch", test_get_or_calculate_best_epoch),
        ("Full Workflow", test_full_workflow),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")

    print(f"\n📊 Total: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
