"""
Phase 1.1: Essential Database Operations Testing

Simple focused tests for:
1. Database connection
2. Create TrainingJob
3. Create TrainingIteration with foreign key
4. Create EpochTrain
5. Create Eval
6. Query and filter jobs

Run: python tests/test_01_database.py
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tortoise import Tortoise
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    # Use SQLite for testing (simpler, no server needed)
    # Database file: innotrain.db in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "innotrain.db")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite://{db_path}")

    await Tortoise.init(
        db_url=DATABASE_URL,
        modules={
            "models": [
                "models.training_job",
                "models.training_iteration",
                "models.epoch_train",
                "models.eval",
            ]
        },
    )

    # Generate schemas (creates tables if they don't exist)
    await Tortoise.generate_schemas()

    print("✅ Database connected!")
    print(f"   Type: SQLite")
    print(f"   File: {DATABASE_URL}\n")


async def close_db():
    """Close database connection"""
    await Tortoise.close_connections()
    print("\n🔌 Database connection closed")


async def test_1_create_training_job():
    """Test 1: Create a TrainingJob record"""
    print("=" * 70)
    print("TEST 1: Create TrainingJob")
    print("=" * 70)

    try:
        job = await TrainingJob.create(
            project_id="test-project-001",
            training_run_id="test-run-001",
            status=TrainingJobStatus.PENDING,
            machine_config={
                "instance_type": "gpu_1x_a10",
                "instance_id": "i-12345",
                "ip": "192.168.1.100",
            },
            training_config={"iterations": 2, "epochs": 3},
        )

        print(f"\n📝 Created TrainingJob:")
        print(f"   UUID: {job.uuid}")
        print(f"   Project: {job.project_id}")
        print(f"   Status: {job.status.value}")
        print(f"   Created: {job.created_at}")

        # Verify
        fetched = await TrainingJob.get(uuid=job.uuid)
        assert fetched.project_id == "test-project-001"

        print("\n✅ PASSED")
        return job

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_2_create_iteration(job):
    """Test 2: Create TrainingIteration linked to job"""
    print("\n" + "=" * 70)
    print("TEST 2: Create TrainingIteration")
    print("=" * 70)

    if not job:
        print("❌ SKIPPED: No job available")
        return None

    try:
        iteration = await TrainingIteration.create(
            training_job=job,
            iteration_number=1,
            step_type=StepType.TRAINING,
            step_config={"epochs": 3, "lr": 0.001},
        )

        print(f"\n📝 Created TrainingIteration:")
        print(f"   UUID: {iteration.uuid}")
        print(f"   Iteration #: {iteration.iteration_number}")
        print(f"   Step Type: {iteration.step_type.value}")

        # Verify foreign key
        await iteration.fetch_related("training_job")
        assert iteration.training_job.uuid == job.uuid

        print("\n✅ PASSED - Foreign key works!")
        return iteration

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_3_create_epoch(iteration):
    """Test 3: Create EpochTrain records"""
    print("\n" + "=" * 70)
    print("TEST 3: Create EpochTrain")
    print("=" * 70)

    if not iteration:
        print("❌ SKIPPED: No iteration available")
        return None

    try:
        epochs = []
        for i in range(1, 4):
            epoch = await EpochTrain.create(
                iteration=iteration,
                iteration_number=iteration.iteration_number,
                epoch_number=i,
                metrics={"avg_loss": 0.5 - (i * 0.1), "accuracy": 0.7 + (i * 0.05)},
            )
            epochs.append(epoch)
            print(f"\n📝 Epoch {i}: loss={epoch.metrics['avg_loss']:.2f}")

        print(f"\n✅ PASSED - Created {len(epochs)} epochs")
        return epochs

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_4_create_eval():
    """Test 4: Create Eval record (standalone, no iteration FK)"""
    print("\n" + "=" * 70)
    print("TEST 4: Create Eval")
    print("=" * 70)

    try:
        eval_record = await Eval.create(
            model_id="model-v1-iter1",
            dataset="eval_dataset",
            metrics={"accuracy": 0.85, "f1_score": 0.83},
            config={"batch_size": 16},
        )

        print(f"\n📝 Created Eval:")
        print(f"   Model: {eval_record.model_id}")
        print(f"   Dataset: {eval_record.dataset}")
        print(f"   Accuracy: {eval_record.metrics['accuracy']}")

        print("\n✅ PASSED")
        return eval_record

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_5_query_and_filter():
    """Test 5: Query and filter jobs"""
    print("\n" + "=" * 70)
    print("TEST 5: Query and Filter")
    print("=" * 70)

    try:
        # Create additional jobs for filtering
        await TrainingJob.create(
            project_id="project-A",
            training_run_id="run-A1",
            status=TrainingJobStatus.RUNNING,
        )

        await TrainingJob.create(
            project_id="project-B",
            training_run_id="run-B1",
            status=TrainingJobStatus.COMPLETED,
        )

        # Test queries
        total = await TrainingJob.all().count()
        print(f"\n📊 Total jobs: {total}")

        pending = await TrainingJob.filter(status=TrainingJobStatus.PENDING).count()
        print(f"📊 Pending jobs: {pending}")

        running = await TrainingJob.filter(status=TrainingJobStatus.RUNNING).count()
        print(f"📊 Running jobs: {running}")

        # Test pagination
        page1 = await TrainingJob.all().limit(2)
        print(f"\n📄 Page 1 (limit=2): {len(page1)} jobs")

        # Test ordering
        recent = await TrainingJob.all().order_by("-created_at").limit(3)
        print(f"📄 Recent jobs (desc): {len(recent)} jobs")

        print("\n✅ PASSED - Queries work!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_6_relationships():
    """Test 6: Verify relationships work"""
    print("\n" + "=" * 70)
    print("TEST 6: Relationships")
    print("=" * 70)

    try:
        # Get a job with iterations
        jobs = await TrainingJob.all().prefetch_related("iterations").limit(1)

        if not jobs:
            print("❌ No jobs found")
            return False

        job = jobs[0]
        print(f"\n📊 Job {job.uuid}:")
        print(f"   Iterations: {len(job.iterations)}")

        for iteration in job.iterations:
            # Count epochs
            epoch_count = await EpochTrain.filter(iteration=iteration).count()

            print(f"\n   Iteration {iteration.iteration_number}:")
            print(f"      Epochs: {epoch_count}")

        print("\n✅ PASSED - Relationships work!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 1.1: ESSENTIAL DATABASE OPERATIONS".center(70))
    print("#" * 70)

    results = {}

    # Initialize DB
    try:
        await init_db()
        results["connection"] = True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Run tests
    job = await test_1_create_training_job()
    results["training_job"] = job is not None

    iteration = await test_2_create_iteration(job)
    results["iteration"] = iteration is not None

    epochs = await test_3_create_epoch(iteration)
    results["epochs"] = epochs is not None

    eval_record = await test_4_create_eval()
    results["eval"] = eval_record is not None

    results["query"] = await test_5_query_and_filter()
    results["relationships"] = await test_6_relationships()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    await close_db()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
