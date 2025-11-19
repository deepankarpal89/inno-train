"""Phase 6.1: API Endpoint Testing

Tests for all FastAPI endpoints:
1. POST /v1/training/start - Create job
2. GET /v1/training/jobs/{uuid} - Get status
3. GET /v1/training/jobs/{uuid}/details - Get iterations
4. GET /v1/training/jobs - List all jobs
5. GET /v1/training/jobs?status=X - Filter by status
6. GET /v1/training/jobs?project_id=X - Filter by project
7. GET /v1/training/jobs?limit=X&offset=Y - Pagination
8. POST /v1/training/jobs/{uuid}/cancel - Cancel job
9. GET /v1/training/jobs/{uuid}/iterations/{n}/epochs - Get epochs
10. GET /v1/training/evaluations - List evaluations

Run: python tests/test_18_api_endpoints.py
"""

import asyncio
import os
import sys
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
test_app = FastAPI(title="InnoTrain Test")
test_app.include_router(api_router)

# Create test client with app as positional argument
client = TestClient(test_app)


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_api.db")

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


async def setup_test_data():
    """Create test data for API testing"""
    print("\n📊 Setting up test data...")

    # Create multiple jobs with different statuses and projects
    jobs = []

    # Job 1: Pending
    job1 = await TrainingJob.create(
        project_id="project-A",
        training_run_id="run-A1",
        status=TrainingJobStatus.PENDING,
        machine_config={"instance_type": "gpu_1x_a10"},
        training_config={"iterations": 2, "epochs": 3},
    )
    jobs.append(job1)

    # Job 2: Running with iterations
    job2 = await TrainingJob.create(
        project_id="project-A",
        training_run_id="run-A2",
        status=TrainingJobStatus.RUNNING,
        machine_config={"instance_type": "gpu_1x_a100"},
        training_config={"iterations": 3, "epochs": 5},
    )
    jobs.append(job2)

    # Create iterations and epochs for job2
    iteration1 = await TrainingIteration.create(
        training_job=job2,
        iteration_number=1,
        step_type=StepType.TRAINING,
        step_config={"epochs": 5, "lr": 0.001},
    )

    # Create epochs for iteration1
    for epoch_num in range(1, 6):
        await EpochTrain.create(
            iteration=iteration1,
            iteration_number=1,
            epoch_number=epoch_num,
            metrics={
                "loss": 1.0 - (epoch_num * 0.1),
                "accuracy": 0.5 + (epoch_num * 0.08),
            },
        )

    # Job 3: Completed
    job3 = await TrainingJob.create(
        project_id="project-B",
        training_run_id="run-B1",
        status=TrainingJobStatus.COMPLETED,
        machine_config={"instance_type": "gpu_1x_a10"},
        training_config={"iterations": 1, "epochs": 2},
    )
    jobs.append(job3)

    # Job 4: Failed
    job4 = await TrainingJob.create(
        project_id="project-B",
        training_run_id="run-B2",
        status=TrainingJobStatus.FAILED,
        machine_config={"instance_type": "gpu_1x_a10"},
        training_config={"iterations": 2, "epochs": 3},
    )
    jobs.append(job4)

    # Job 5: Cancelled
    job5 = await TrainingJob.create(
        project_id="project-C",
        training_run_id="run-C1",
        status=TrainingJobStatus.CANCELLED,
        machine_config={"instance_type": "gpu_1x_a100"},
        training_config={"iterations": 2, "epochs": 4},
    )
    jobs.append(job5)

    # Create evaluation records
    await Eval.create(
        model_id="model-v1",
        dataset="eval_dataset_1",
        metrics={"accuracy": 0.85, "f1_score": 0.83},
        config={"batch_size": 16},
    )

    await Eval.create(
        model_id="model-v2",
        dataset="eval_dataset_2",
        metrics={"accuracy": 0.90, "f1_score": 0.88},
        config={"batch_size": 32},
    )

    print(f"✅ Created {len(jobs)} test jobs and 2 evaluations\n")
    return jobs


def test_1_hello_endpoint():
    """Test 1: Hello World endpoint"""
    print("=" * 70)
    print("TEST 1: GET /api/hello")
    print("=" * 70)

    try:
        response = client.get("/api/hello")

        print(f"\n📝 Response Status: {response.status_code}")
        print(f"📝 Response Body: {response.json()}")

        assert response.status_code == 200
        assert "message" in response.json()

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_2_get_job_status():
    """Test 2: GET /v1/training/jobs/{uuid} - Get job status"""
    print("\n" + "=" * 70)
    print("TEST 2: GET /v1/training/jobs/{uuid}")
    print("=" * 70)

    try:
        # Get a job from database
        job = await TrainingJob.filter(status=TrainingJobStatus.RUNNING).first()

        if not job:
            print("❌ SKIPPED: No running job found")
            return False

        response = client.get(f"/api/v1/training/jobs/{job.uuid}")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Job UUID: {data.get('job_uuid')}")
        print(f"📝 Status: {data.get('status')}")
        print(f"📝 Project ID: {data.get('project_id')}")

        assert response.status_code == 200
        assert data["job_uuid"] == str(job.uuid)
        assert data["status"] == job.status.value
        assert data["project_id"] == job.project_id

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_3_get_job_status_not_found():
    """Test 3: GET /v1/training/jobs/{uuid} - 404 for non-existent job"""
    print("\n" + "=" * 70)
    print("TEST 3: GET /v1/training/jobs/{uuid} - Not Found")
    print("=" * 70)

    try:
        fake_uuid = str(uuid_lib.uuid4())
        response = client.get(f"/api/v1/training/jobs/{fake_uuid}")

        print(f"\n📝 Response Status: {response.status_code}")
        print(f"📝 Response: {response.json()}")

        assert response.status_code == 404

        print("\n✅ PASSED - Correctly returns 404")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_4_get_job_details():
    """Test 4: GET /v1/training/jobs/{uuid}/details - Get iterations"""
    print("\n" + "=" * 70)
    print("TEST 4: GET /v1/training/jobs/{uuid}/details")
    print("=" * 70)

    try:
        # Get job with iterations
        job = await TrainingJob.filter(status=TrainingJobStatus.RUNNING).first()

        if not job:
            print("❌ SKIPPED: No running job found")
            return False

        response = client.get(f"/api/v1/training/jobs/{job.uuid}/details")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Job UUID: {data.get('job_uuid')}")
        print(f"📝 Status: {data.get('status')}")
        print(f"📝 Iterations: {len(data.get('iterations', []))}")

        assert response.status_code == 200
        assert data["job_uuid"] == str(job.uuid)
        assert "iterations" in data
        assert isinstance(data["iterations"], list)

        if data["iterations"]:
            iteration = data["iterations"][0]
            print(
                f"📝 First Iteration: #{iteration['iteration_number']}, Type: {iteration['step_type']}"
            )
            assert "iteration_number" in iteration
            assert "step_type" in iteration

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_5_list_all_jobs():
    """Test 5: GET /v1/training/jobs - List all jobs"""
    print("\n" + "=" * 70)
    print("TEST 5: GET /v1/training/jobs - List All")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/jobs")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total Jobs: {data.get('total')}")
        print(f"📝 Returned: {len(data.get('jobs', []))}")
        print(f"📝 Limit: {data.get('limit')}")
        print(f"📝 Offset: {data.get('offset')}")

        assert response.status_code == 200
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)
        assert data["total"] >= 5  # We created 5 jobs

        # Verify job structure
        if data["jobs"]:
            job = data["jobs"][0]
            assert "job_uuid" in job
            assert "status" in job
            assert "project_id" in job
            print(f"📝 First Job: {job['job_uuid'][:8]}... - {job['status']}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_6_filter_by_status():
    """Test 6: GET /v1/training/jobs?status=running - Filter by status"""
    print("\n" + "=" * 70)
    print("TEST 6: GET /v1/training/jobs?status=running")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/jobs?status=running")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total Running Jobs: {data.get('total')}")
        print(f"📝 Returned: {len(data.get('jobs', []))}")

        assert response.status_code == 200
        assert data["total"] >= 1  # We created 1 running job

        # Verify all jobs have running status
        for job in data["jobs"]:
            assert job["status"] == "running"
            print(f"   ✓ Job {job['job_uuid'][:8]}... is running")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_7_filter_by_project():
    """Test 7: GET /v1/training/jobs?project_id=X - Filter by project"""
    print("\n" + "=" * 70)
    print("TEST 7: GET /v1/training/jobs?project_id=project-A")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/jobs?project_id=project-A")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total Jobs for project-A: {data.get('total')}")
        print(f"📝 Returned: {len(data.get('jobs', []))}")

        assert response.status_code == 200
        assert data["total"] >= 2  # We created 2 jobs for project-A

        # Verify all jobs belong to project-A
        for job in data["jobs"]:
            assert job["project_id"] == "project-A"
            print(f"   ✓ Job {job['job_uuid'][:8]}... belongs to project-A")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_8_pagination():
    """Test 8: GET /v1/training/jobs?limit=2&offset=1 - Pagination"""
    print("\n" + "=" * 70)
    print("TEST 8: GET /v1/training/jobs?limit=2&offset=1")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/jobs?limit=2&offset=1")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total Jobs: {data.get('total')}")
        print(f"📝 Limit: {data.get('limit')}")
        print(f"📝 Offset: {data.get('offset')}")
        print(f"📝 Returned: {len(data.get('jobs', []))}")

        assert response.status_code == 200
        assert data["limit"] == 2
        assert data["offset"] == 1
        assert len(data["jobs"]) <= 2  # Should return at most 2 jobs

        # Show returned jobs
        for job in data["jobs"]:
            print(f"   ✓ Job {job['job_uuid'][:8]}... - {job['status']}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_9_get_iteration_epochs():
    """Test 9: GET /v1/training/jobs/{uuid}/iterations/{n}/epochs"""
    print("\n" + "=" * 70)
    print("TEST 9: GET /v1/training/jobs/{uuid}/iterations/1/epochs")
    print("=" * 70)

    try:
        # Get job with iterations
        job = await TrainingJob.filter(status=TrainingJobStatus.RUNNING).first()

        if not job:
            print("❌ SKIPPED: No running job found")
            return False

        response = client.get(f"/api/v1/training/jobs/{job.uuid}/iterations/1/epochs")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Job UUID: {data.get('job_uuid')}")
        print(f"📝 Iteration: {data.get('iteration_number')}")
        print(f"📝 Epochs: {len(data.get('epochs', []))}")

        assert response.status_code == 200
        assert data["job_uuid"] == str(job.uuid)
        assert data["iteration_number"] == 1
        assert "epochs" in data
        assert isinstance(data["epochs"], list)
        assert len(data["epochs"]) == 5  # We created 5 epochs

        # Verify epoch structure
        if data["epochs"]:
            epoch = data["epochs"][0]
            assert "epoch_number" in epoch
            assert "metrics" in epoch
            print(
                f"📝 First Epoch: #{epoch['epoch_number']}, Metrics: {epoch['metrics']}"
            )

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_10_get_iteration_epochs_not_found():
    """Test 10: GET iterations/epochs - 404 for non-existent iteration"""
    print("\n" + "=" * 70)
    print("TEST 10: GET iterations/epochs - Not Found")
    print("=" * 70)

    try:
        job = await TrainingJob.first()

        if not job:
            print("❌ SKIPPED: No job found")
            return False

        response = client.get(f"/api/v1/training/jobs/{job.uuid}/iterations/999/epochs")

        print(f"\n📝 Response Status: {response.status_code}")
        print(f"📝 Response: {response.json()}")

        assert response.status_code == 404

        print("\n✅ PASSED - Correctly returns 404")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_11_list_evaluations():
    """Test 11: GET /v1/training/evaluations - List evaluations"""
    print("\n" + "=" * 70)
    print("TEST 11: GET /v1/training/evaluations")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/evaluations")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total Evaluations: {data.get('total')}")
        print(f"📝 Returned: {len(data.get('evaluations', []))}")
        print(f"📝 Limit: {data.get('limit')}")
        print(f"📝 Offset: {data.get('offset')}")

        assert response.status_code == 200
        assert "evaluations" in data
        assert "total" in data
        assert isinstance(data["evaluations"], list)
        assert data["total"] >= 2  # We created 2 evaluations

        # Verify evaluation structure
        if data["evaluations"]:
            eval_record = data["evaluations"][0]
            assert "model_id" in eval_record
            assert "dataset" in eval_record
            assert "metrics" in eval_record
            print(
                f"📝 First Eval: Model={eval_record['model_id']}, Dataset={eval_record['dataset']}"
            )

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_12_filter_evaluations_by_model():
    """Test 12: GET /v1/training/evaluations?model_id=X"""
    print("\n" + "=" * 70)
    print("TEST 12: GET /v1/training/evaluations?model_id=model-v1")
    print("=" * 70)

    try:
        response = client.get("/api/v1/training/evaluations?model_id=model-v1")

        print(f"\n📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Total: {data.get('total')}")
        print(f"📝 Returned: {len(data.get('evaluations', []))}")

        assert response.status_code == 200
        assert data["total"] >= 1  # We created 1 evaluation for model-v1

        # Verify all evaluations belong to model-v1
        for eval_record in data["evaluations"]:
            assert eval_record["model_id"] == "model-v1"
            print(f"   ✓ Eval for model-v1: {eval_record['dataset']}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_13_cancel_job():
    """Test 13: POST /v1/training/jobs/{uuid}/cancel"""
    print("\n" + "=" * 70)
    print("TEST 13: POST /v1/training/jobs/{uuid}/cancel")
    print("=" * 70)

    try:
        # Get a pending job to cancel
        job = await TrainingJob.filter(status=TrainingJobStatus.PENDING).first()

        if not job:
            print("❌ SKIPPED: No pending job found")
            return False

        print(f"\n📝 Cancelling job: {job.uuid}")
        response = client.post(f"/api/v1/training/jobs/{job.uuid}/cancel")

        print(f"📝 Response Status: {response.status_code}")
        data = response.json()
        print(f"📝 Success: {data.get('success')}")
        print(f"📝 Message: {data.get('message')}")
        print(f"📝 Status: {data.get('status')}")

        assert response.status_code == 200
        assert data["success"] == True
        assert data["status"] == "cancelled"

        # Verify job status updated in database
        updated_job = await TrainingJob.get(uuid=job.uuid)
        assert updated_job.status == TrainingJobStatus.CANCELLED
        print(f"📝 Job status in DB: {updated_job.status.value}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_14_cancel_completed_job():
    """Test 14: POST cancel - 400 for completed job"""
    print("\n" + "=" * 70)
    print("TEST 14: POST cancel - Cannot Cancel Completed Job")
    print("=" * 70)

    try:
        # Get a completed job
        job = await TrainingJob.filter(status=TrainingJobStatus.COMPLETED).first()

        if not job:
            print("❌ SKIPPED: No completed job found")
            return False

        print(f"\n📝 Attempting to cancel completed job: {job.uuid}")
        response = client.post(f"/api/v1/training/jobs/{job.uuid}/cancel")

        print(f"📝 Response Status: {response.status_code}")
        print(f"📝 Response: {response.json()}")

        assert response.status_code == 400

        print("\n✅ PASSED - Correctly returns 400")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 6.1: API ENDPOINT TESTING".center(70))
    print("#" * 70)

    results = {}

    # Initialize DB
    try:
        await init_db()
        results["connection"] = True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Setup test data
    try:
        await setup_test_data()
        results["setup"] = True
    except Exception as e:
        print(f"❌ Test data setup failed: {e}")
        import traceback

        traceback.print_exc()
        await close_db()
        return

    # Run tests
    results["test_1_hello"] = test_1_hello_endpoint()
    results["test_2_get_status"] = await test_2_get_job_status()
    results["test_3_not_found"] = await test_3_get_job_status_not_found()
    results["test_4_job_details"] = await test_4_get_job_details()
    results["test_5_list_all"] = test_5_list_all_jobs()
    results["test_6_filter_status"] = test_6_filter_by_status()
    results["test_7_filter_project"] = test_7_filter_by_project()
    results["test_8_pagination"] = test_8_pagination()
    results["test_9_iteration_epochs"] = await test_9_get_iteration_epochs()
    results["test_10_epochs_not_found"] = await test_10_get_iteration_epochs_not_found()
    results["test_11_list_evals"] = test_11_list_evaluations()
    results["test_12_filter_evals"] = test_12_filter_evaluations_by_model()
    results["test_13_cancel_job"] = await test_13_cancel_job()
    results["test_14_cancel_completed"] = await test_14_cancel_completed_job()

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
