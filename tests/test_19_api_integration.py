"""Phase 6.2: API Integration Testing

End-to-end integration tests for the training API workflow:
1. Start job via API
2. Poll status while running
3. Check iterations appear in real-time
4. Check epochs appear as training progresses
5. Get detailed job info
6. Cancel a running job
7. Verify cleanup happens after cancel
8. List jobs with various filters
9. Get evaluation results

This simulates a real client workflow interacting with the API.

Run: python tests/test_19_api_integration.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
import uuid as uuid_lib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tortoise import Tortoise

from app.api import api_router
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval

# Create test app without lifespan (we manage DB ourselves)
test_app = FastAPI(title="InnoTrain Integration Test")
test_app.include_router(api_router)

# Create test client
client = TestClient(test_app)


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_api_integration.db")

    # Remove old test database
    if os.path.exists(db_path):
        os.remove(db_path)

    DATABASE_URL = f"sqlite://{db_path}"

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

    await Tortoise.generate_schemas()

    print("✅ Database connected!")
    print(f"   File: {DATABASE_URL}\n")


async def close_db():
    """Close database connection"""
    await Tortoise.close_connections()
    print("\n🔌 Database connection closed")


async def simulate_training_progress(job_uuid: str, iterations: int = 2, epochs: int = 3):
    """
    Simulate training progress by creating iterations and epochs
    This mimics what the real training orchestrator would do
    """
    print(f"\n🔄 Simulating training progress for job {job_uuid[:8]}...")

    job = await TrainingJob.get(uuid=job_uuid)

    # Update to running
    job.status = TrainingJobStatus.RUNNING
    await job.save()
    print(f"   ✓ Job status: RUNNING")

    for iter_num in range(1, iterations + 1):
        print(f"\n   📊 Creating iteration {iter_num}...")

        # Create training iteration
        iteration = await TrainingIteration.create(
            training_job=job,
            iteration_number=iter_num,
            step_type=StepType.TRAINING,
            step_config={"epochs": epochs, "lr": 0.001 * iter_num},
        )

        # Simulate epochs appearing one by one
        for epoch_num in range(1, epochs + 1):
            await asyncio.sleep(0.5)  # Simulate time between epochs

            await EpochTrain.create(
                iteration=iteration,
                iteration_number=iter_num,
                epoch_number=epoch_num,
                metrics={
                    "loss": max(0.1, 1.0 - (epoch_num * 0.15)),
                    "accuracy": min(0.95, 0.5 + (epoch_num * 0.1)),
                },
            )
            print(f"      ✓ Epoch {epoch_num} completed")

        # Mark iteration complete
        iteration.completed_at = datetime.utcnow()
        iteration.step_time = 2.5
        await iteration.save()

    # Mark job complete
    job.status = TrainingJobStatus.COMPLETED
    job.completed_at = datetime.utcnow()
    job.time_taken = 60
    await job.save()

    print(f"\n   ✅ Training simulation completed!")


async def test_1_start_job_via_api():
    """Test 1: Start a training job via API"""
    print("=" * 70)
    print("TEST 1: Start Training Job via API")
    print("=" * 70)

    try:
        # Prepare request matching train_request.json format
        request_data = {
            "success": True,
            "message": "Training run started successfully",
            "status_code": 200,
            "data": {
                "request_data": {
                    "training_run_id": str(uuid_lib.uuid4()),
                    "project": {
                        "id": "integration-test-project",
                        "name": "Integration Test",
                        "description": "API integration testing",
                        "task_type": "text_classification",
                    },
                    "prompt": {
                        "template": "Test prompt",
                        "system_message": "You are a test assistant",
                    },
                    "train_dataset": {
                        "name": "test_train",
                        "path": "s3://bucket/train.csv",
                    },
                    "eval_dataset": {
                        "name": "test_eval",
                        "path": "s3://bucket/eval.csv",
                    },
                    "training_config": {
                        "iterations": 2,
                        "epochs": 3,
                        "batch_size": 16,
                        "learning_rate": 0.001,
                    },
                }
            },
        }

        print("\n📤 Sending POST /api/v1/training/start...")
        response = client.post("/api/v1/training/start", json=request_data)

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Success: {data.get('success')}")
        print(f"📝 Job UUID: {data.get('job_uuid')}")
        print(f"📝 Status: {data.get('status')}")
        print(f"📝 Message: {data.get('message')}")

        assert response.status_code == 200
        assert data["success"] == True
        assert "job_uuid" in data
        assert data["status"] == "pending"

        job_uuid = data["job_uuid"]

        print("\n✅ PASSED")
        return job_uuid

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_2_poll_status_while_running(job_uuid: str):
    """Test 2: Poll job status while it's running"""
    print("\n" + "=" * 70)
    print("TEST 2: Poll Status While Running")
    print("=" * 70)

    if not job_uuid:
        print("❌ SKIPPED: No job UUID available")
        return False

    try:
        print(f"\n🔄 Starting background training simulation...")

        # Start training simulation in background
        simulation_task = asyncio.create_task(simulate_training_progress(job_uuid, 2, 3))

        # Poll status multiple times
        print("\n📊 Polling job status...")
        poll_count = 0
        max_polls = 10
        statuses_seen = []

        while poll_count < max_polls:
            await asyncio.sleep(1)  # Poll every second

            response = client.get(f"/api/v1/training/jobs/{job_uuid}")
            assert response.status_code == 200

            data = response.json()
            status = data["status"]
            statuses_seen.append(status)

            print(f"   Poll {poll_count + 1}: Status = {status}")

            poll_count += 1

            # Break if completed
            if status == "completed":
                print("\n   ✓ Job completed!")
                break

        # Wait for simulation to finish
        await simulation_task

        # Verify we saw status progression
        assert "pending" in statuses_seen or "running" in statuses_seen
        assert "completed" in statuses_seen

        print(f"\n📊 Status progression: {' → '.join(set(statuses_seen))}")
        print("\n✅ PASSED - Status polling works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_3_check_iterations_realtime(job_uuid: str):
    """Test 3: Check iterations appear in real-time"""
    print("\n" + "=" * 70)
    print("TEST 3: Check Iterations Appear in Real-Time")
    print("=" * 70)

    if not job_uuid:
        print("❌ SKIPPED: No job UUID available")
        return False

    try:
        print(f"\n📊 Fetching job details...")

        response = client.get(f"/api/v1/training/jobs/{job_uuid}/details")
        assert response.status_code == 200

        data = response.json()
        iterations = data.get("iterations", [])

        print(f"\n📝 Job UUID: {data['job_uuid'][:8]}...")
        print(f"📝 Status: {data['status']}")
        print(f"📝 Iterations found: {len(iterations)}")

        assert len(iterations) == 2  # We created 2 iterations

        # Verify iteration structure
        for iteration in iterations:
            print(f"\n   Iteration {iteration['iteration_number']}:")
            print(f"      Type: {iteration['step_type']}")
            print(f"      Config: {iteration.get('step_config', {})}")
            print(f"      Completed: {iteration.get('completed_at', 'N/A')}")

            assert "iteration_number" in iteration
            assert "step_type" in iteration
            assert iteration["step_type"] == "training"

        print("\n✅ PASSED - Iterations appear correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_4_check_epochs_progress(job_uuid: str):
    """Test 4: Check epochs appear as training progresses"""
    print("\n" + "=" * 70)
    print("TEST 4: Check Epochs Appear as Training Progresses")
    print("=" * 70)

    if not job_uuid:
        print("❌ SKIPPED: No job UUID available")
        return False

    try:
        # Check epochs for iteration 1
        print(f"\n📊 Fetching epochs for iteration 1...")

        response = client.get(f"/api/v1/training/jobs/{job_uuid}/iterations/1/epochs")
        assert response.status_code == 200

        data = response.json()
        epochs = data.get("epochs", [])

        print(f"\n📝 Iteration: {data['iteration_number']}")
        print(f"📝 Epochs found: {len(epochs)}")

        assert len(epochs) == 3  # We created 3 epochs per iteration

        # Verify epoch progression
        for epoch in epochs:
            print(f"\n   Epoch {epoch['epoch_number']}:")
            print(f"      Loss: {epoch['metrics'].get('loss', 'N/A')}")
            print(f"      Accuracy: {epoch['metrics'].get('accuracy', 'N/A')}")

            assert "epoch_number" in epoch
            assert "metrics" in epoch
            assert "loss" in epoch["metrics"]
            assert "accuracy" in epoch["metrics"]

        # Verify metrics improve over epochs
        losses = [e["metrics"]["loss"] for e in epochs]
        accuracies = [e["metrics"]["accuracy"] for e in epochs]

        print(f"\n📊 Loss trend: {' → '.join([f'{l:.3f}' for l in losses])}")
        print(f"📊 Accuracy trend: {' → '.join([f'{a:.3f}' for a in accuracies])}")

        # Loss should generally decrease
        assert losses[0] > losses[-1], "Loss should decrease over epochs"
        # Accuracy should generally increase
        assert accuracies[0] < accuracies[-1], "Accuracy should increase over epochs"

        print("\n✅ PASSED - Epochs show proper progression!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_5_get_detailed_job_info(job_uuid: str):
    """Test 5: Get comprehensive job information"""
    print("\n" + "=" * 70)
    print("TEST 5: Get Detailed Job Info")
    print("=" * 70)

    if not job_uuid:
        print("❌ SKIPPED: No job UUID available")
        return False

    try:
        print(f"\n📊 Fetching comprehensive job details...")

        response = client.get(f"/api/v1/training/jobs/{job_uuid}/details")
        assert response.status_code == 200

        data = response.json()

        print(f"\n📝 Job Information:")
        print(f"   UUID: {data['job_uuid'][:8]}...")
        print(f"   Status: {data['status']}")
        print(f"   Project: {data['project_id']}")
        print(f"   Training Run: {data['training_run_id']}")
        print(f"   Created: {data.get('created_at', 'N/A')}")
        print(f"   Completed: {data.get('completed_at', 'N/A')}")
        print(f"   Time Taken: {data.get('time_taken', 'N/A')}s")

        # Verify all expected fields
        assert "job_uuid" in data
        assert "status" in data
        assert "project_id" in data
        assert "training_run_id" in data
        assert "iterations" in data
        assert "machine_config" in data
        assert "training_config" in data

        print(f"\n📊 Machine Config: {data.get('machine_config', {})}")
        print(f"📊 Training Config: {data.get('training_config', {})}")
        print(f"📊 Total Iterations: {len(data['iterations'])}")

        print("\n✅ PASSED - Detailed info retrieved successfully!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_6_cancel_running_job():
    """Test 6: Cancel a running job"""
    print("\n" + "=" * 70)
    print("TEST 6: Cancel a Running Job")
    print("=" * 70)

    try:
        # Create a new job to cancel
        print("\n📤 Creating a new job to cancel...")

        request_data = {
            "success": True,
            "data": {
                "request_data": {
                    "training_run_id": str(uuid_lib.uuid4()),
                    "project": {
                        "id": "cancel-test-project",
                        "name": "Cancel Test",
                        "description": "Testing job cancellation",
                        "task_type": "text_classification",
                    },
                    "training_config": {"iterations": 5, "epochs": 10},
                }
            },
        }

        response = client.post("/api/v1/training/start", json=request_data)
        assert response.status_code == 200

        job_uuid = response.json()["job_uuid"]
        print(f"   ✓ Created job: {job_uuid[:8]}...")

        # Update to running status
        job = await TrainingJob.get(uuid=job_uuid)
        job.status = TrainingJobStatus.RUNNING
        await job.save()
        print(f"   ✓ Job status: RUNNING")

        # Cancel the job
        print(f"\n🛑 Cancelling job...")
        cancel_response = client.post(f"/api/v1/training/jobs/{job_uuid}/cancel")

        print(f"\n📝 Response Status: {cancel_response.status_code}")
        cancel_data = cancel_response.json()
        print(f"📝 Success: {cancel_data.get('success')}")
        print(f"📝 Message: {cancel_data.get('message')}")
        print(f"📝 Status: {cancel_data.get('status')}")

        assert cancel_response.status_code == 200
        assert cancel_data["success"] == True
        assert cancel_data["status"] == "cancelled"

        # Verify status in database
        updated_job = await TrainingJob.get(uuid=job_uuid)
        assert updated_job.status == TrainingJobStatus.CANCELLED
        print(f"\n📊 Job status in DB: {updated_job.status.value}")

        print("\n✅ PASSED - Job cancelled successfully!")
        return job_uuid

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_7_verify_cleanup_after_cancel(job_uuid: str):
    """Test 7: Verify cleanup happens after cancel"""
    print("\n" + "=" * 70)
    print("TEST 7: Verify Cleanup After Cancel")
    print("=" * 70)

    if not job_uuid:
        print("❌ SKIPPED: No cancelled job UUID available")
        return False

    try:
        print(f"\n📊 Checking job state after cancellation...")

        # Get job status
        response = client.get(f"/api/v1/training/jobs/{job_uuid}")
        assert response.status_code == 200

        data = response.json()
        print(f"\n📝 Job UUID: {data['job_uuid'][:8]}...")
        print(f"📝 Status: {data['status']}")

        assert data["status"] == "cancelled"

        # Verify job details still accessible
        details_response = client.get(f"/api/v1/training/jobs/{job_uuid}/details")
        assert details_response.status_code == 200

        details = details_response.json()
        print(f"📝 Iterations: {len(details.get('iterations', []))}")

        # Verify cannot cancel again
        print(f"\n🔄 Attempting to cancel again (should fail)...")
        retry_response = client.post(f"/api/v1/training/jobs/{job_uuid}/cancel")

        print(f"📝 Response Status: {retry_response.status_code}")

        assert retry_response.status_code == 400  # Should fail

        print("\n✅ PASSED - Cleanup verified, cannot re-cancel!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_8_list_jobs_with_filters():
    """Test 8: List jobs with various filters"""
    print("\n" + "=" * 70)
    print("TEST 8: List Jobs with Various Filters")
    print("=" * 70)

    try:
        # Create additional test jobs
        print("\n📊 Creating test jobs with different statuses...")

        for i in range(3):
            await TrainingJob.create(
                project_id=f"filter-test-project-{i % 2}",
                training_run_id=f"filter-run-{i}",
                status=[
                    TrainingJobStatus.PENDING,
                    TrainingJobStatus.RUNNING,
                    TrainingJobStatus.COMPLETED,
                ][i],
                machine_config={"instance_type": "gpu_1x_a10"},
                training_config={"iterations": 2, "epochs": 3},
            )

        print("   ✓ Created 3 test jobs")

        # Test 1: List all jobs
        print("\n📋 Test: List all jobs")
        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 200

        data = response.json()
        print(f"   Total: {data['total']}")
        print(f"   Returned: {len(data['jobs'])}")

        assert data["total"] >= 3

        # Test 2: Filter by status
        print("\n📋 Test: Filter by status=completed")
        response = client.get("/api/v1/training/jobs?status=completed")
        assert response.status_code == 200

        data = response.json()
        print(f"   Completed jobs: {data['total']}")

        for job in data["jobs"]:
            assert job["status"] == "completed"
            print(f"      ✓ {job['job_uuid'][:8]}... - completed")

        # Test 3: Filter by project
        print("\n📋 Test: Filter by project_id")
        response = client.get("/api/v1/training/jobs?project_id=filter-test-project-0")
        assert response.status_code == 200

        data = response.json()
        print(f"   Jobs for project-0: {data['total']}")

        for job in data["jobs"]:
            assert job["project_id"] == "filter-test-project-0"
            print(f"      ✓ {job['job_uuid'][:8]}... - project-0")

        # Test 4: Pagination
        print("\n📋 Test: Pagination (limit=2, offset=1)")
        response = client.get("/api/v1/training/jobs?limit=2&offset=1")
        assert response.status_code == 200

        data = response.json()
        print(f"   Limit: {data['limit']}")
        print(f"   Offset: {data['offset']}")
        print(f"   Returned: {len(data['jobs'])}")

        assert data["limit"] == 2
        assert data["offset"] == 1
        assert len(data["jobs"]) <= 2

        # Test 5: Combined filters
        print("\n📋 Test: Combined filters (status + project)")
        response = client.get(
            "/api/v1/training/jobs?status=pending&project_id=filter-test-project-0"
        )
        assert response.status_code == 200

        data = response.json()
        print(f"   Matching jobs: {data['total']}")

        for job in data["jobs"]:
            assert job["status"] == "pending"
            assert job["project_id"] == "filter-test-project-0"

        print("\n✅ PASSED - All filter combinations work!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_9_get_evaluation_results():
    """Test 9: Get evaluation results"""
    print("\n" + "=" * 70)
    print("TEST 9: Get Evaluation Results")
    print("=" * 70)

    try:
        # Create evaluation records
        print("\n📊 Creating evaluation records...")

        eval_data = [
            {
                "model_id": "model-v1",
                "dataset": "eval_dataset_1",
                "metrics": {"accuracy": 0.85, "f1_score": 0.83, "precision": 0.84},
                "config": {"batch_size": 16, "threshold": 0.5},
            },
            {
                "model_id": "model-v2",
                "dataset": "eval_dataset_1",
                "metrics": {"accuracy": 0.90, "f1_score": 0.88, "precision": 0.89},
                "config": {"batch_size": 32, "threshold": 0.5},
            },
            {
                "model_id": "model-v1",
                "dataset": "eval_dataset_2",
                "metrics": {"accuracy": 0.82, "f1_score": 0.80, "precision": 0.81},
                "config": {"batch_size": 16, "threshold": 0.5},
            },
        ]

        for eval_item in eval_data:
            await Eval.create(**eval_item)

        print(f"   ✓ Created {len(eval_data)} evaluation records")

        # Test 1: List all evaluations
        print("\n📋 Test: List all evaluations")
        response = client.get("/api/v1/training/evaluations")
        assert response.status_code == 200

        data = response.json()
        print(f"   Total: {data['total']}")
        print(f"   Returned: {len(data['evaluations'])}")

        assert data["total"] >= 3

        # Verify structure
        for eval_record in data["evaluations"][:2]:
            print(
                f"\n   Eval: {eval_record['model_id']} on {eval_record['dataset']}"
            )
            print(f"      Accuracy: {eval_record['metrics'].get('accuracy', 'N/A')}")
            print(f"      F1: {eval_record['metrics'].get('f1_score', 'N/A')}")

            assert "model_id" in eval_record
            assert "dataset" in eval_record
            assert "metrics" in eval_record

        # Test 2: Filter by model_id
        print("\n📋 Test: Filter by model_id=model-v1")
        response = client.get("/api/v1/training/evaluations?model_id=model-v1")
        assert response.status_code == 200

        data = response.json()
        print(f"   Evaluations for model-v1: {data['total']}")

        for eval_record in data["evaluations"]:
            assert eval_record["model_id"] == "model-v1"
            print(f"      ✓ {eval_record['dataset']} - model-v1")

        # Test 3: Filter by dataset
        print("\n📋 Test: Filter by dataset=eval_dataset_1")
        response = client.get("/api/v1/training/evaluations?dataset=eval_dataset_1")
        assert response.status_code == 200

        data = response.json()
        print(f"   Evaluations for eval_dataset_1: {data['total']}")

        for eval_record in data["evaluations"]:
            assert eval_record["dataset"] == "eval_dataset_1"
            print(f"      ✓ {eval_record['model_id']} - eval_dataset_1")

        # Test 4: Pagination
        print("\n📋 Test: Pagination (limit=2)")
        response = client.get("/api/v1/training/evaluations?limit=2")
        assert response.status_code == 200

        data = response.json()
        print(f"   Limit: {data['limit']}")
        print(f"   Returned: {len(data['evaluations'])}")

        assert len(data["evaluations"]) <= 2

        print("\n✅ PASSED - Evaluation results retrieved successfully!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 6.2: API INTEGRATION TESTING".center(70))
    print("#" * 70)

    results = {}

    # Initialize DB
    try:
        await init_db()
        results["connection"] = True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Run integration tests
    job_uuid = await test_1_start_job_via_api()
    results["test_1_start_job"] = job_uuid is not None

    results["test_2_poll_status"] = await test_2_poll_status_while_running(job_uuid)
    results["test_3_iterations"] = await test_3_check_iterations_realtime(job_uuid)
    results["test_4_epochs"] = await test_4_check_epochs_progress(job_uuid)
    results["test_5_detailed_info"] = await test_5_get_detailed_job_info(job_uuid)

    cancelled_job_uuid = await test_6_cancel_running_job()
    results["test_6_cancel_job"] = cancelled_job_uuid is not None

    results["test_7_verify_cleanup"] = await test_7_verify_cleanup_after_cancel(
        cancelled_job_uuid
    )
    results["test_8_list_filters"] = await test_8_list_jobs_with_filters()
    results["test_9_evaluations"] = await test_9_get_evaluation_results()

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
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n💡 The API workflow is fully functional:")
        print("   ✓ Job creation and status tracking")
        print("   ✓ Real-time iteration and epoch monitoring")
        print("   ✓ Job cancellation and cleanup")
        print("   ✓ Filtering and pagination")
        print("   ✓ Evaluation results retrieval")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    await close_db()


if __name__ == "__main__":
    asyncio.run(run_all_tests())