"""
Test 14: EVAL Event Processing - Phase 4.5

Tests database updates from EVAL phase events:
- EVAL_TRAINING start: Creates evaluation step
- EVAL_MODEL metrics: Creates Eval record
- EVAL_MODEL metrics: Stores all metrics
- EVAL_TRAINING end: Records duration

Run: python tests/test_14_event_eval.py
"""

import asyncio
import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tortoise import Tortoise
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.eval import Eval
from app.services.training_job_monitor import TrainingJobMonitor
from scripts.ssh_executor import SshExecutor, CommandResult


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_event_eval.db")

    # Remove existing test database
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
    print("✅ Database connected!\n")


async def close_db():
    """Close database connection"""
    await Tortoise.close_connections()
    print("\n🔌 Database connection closed")


def create_mock_ssh_executor():
    """Create a mock SSH executor for testing"""
    mock_ssh = Mock(spec=SshExecutor)
    mock_ssh.ip = "192.168.1.100"
    mock_ssh.username = "ubuntu"
    return mock_ssh


async def test_1_eval_training_start_creates_evaluation_step():
    """Test 1: EVAL_TRAINING start event creates evaluation step"""
    print("=" * 70)
    print("TEST 1: EVAL_TRAINING Start Creates Evaluation Step")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-1",
            training_run_id="test-run-1",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # First, create an iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 14:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 3}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process EVAL_TRAINING start event
        eval_start_event = {
            "timestamp": "2025-11-06 14:30:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {
                "config": {
                    "dataset": "cv",
                    "batch_size": 32,
                    "metrics": ["accuracy", "precision", "recall", "f1"],
                }
            },
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Verify evaluation step was created
        eval_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.EVALUATION
        ).all()

        print(f"\n📊 Number of evaluation steps created: {len(eval_steps)}")

        assert len(eval_steps) == 1
        eval_step = eval_steps[0]

        print(f"📊 Step type: {eval_step.step_type}")
        print(f"📊 Iteration number: {eval_step.iteration_number}")
        print(f"📊 Created at: {eval_step.created_at}")
        print(f"📊 Step config: {eval_step.step_config}")

        assert eval_step.step_type == StepType.EVALUATION
        assert eval_step.iteration_number == 1
        assert eval_step.step_config is not None
        assert eval_step.step_config.get("dataset") == "cv"
        assert eval_step.step_config.get("batch_size") == 32
        assert "accuracy" in eval_step.step_config.get("metrics", [])

        print("\n✅ PASSED - evaluation step created successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_2_eval_model_metrics_creates_eval_record():
    """Test 2: EVAL_MODEL metrics event creates Eval record"""
    print("\n" + "=" * 70)
    print("TEST 2: EVAL_MODEL metrics Creates Eval Record")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-2",
            training_run_id="test-run-2",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Create iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 14:10:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 3}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process EVAL_TRAINING start
        eval_start_event = {
            "timestamp": "2025-11-06 14:35:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {"config": {"dataset": "cv"}},
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Process EVAL_MODEL metrics event
        eval_metrics_event = {
            "timestamp": "2025-11-06 14:40:00",
            "level": "INFO",
            "phase": "EVAL_MODEL",
            "event": "metrics",
            "message": "Evaluation metrics computed",
            "data": {
                "config": {"dataset": "cv"},
                "metrics": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.91,
                    "f1": 0.90,
                },
                "model_path": "/models/iteration_1.pth",
                "metrics_json_path": "/output/metrics/iteration_1.json",
            },
        }

        await monitor._process_log_line(json.dumps(eval_metrics_event))

        # Verify Eval record was created
        eval_records = await Eval.all()

        print(f"\n📊 Number of eval records created: {len(eval_records)}")

        assert len(eval_records) == 1
        eval_record = eval_records[0]

        print(f"📊 Model ID: {eval_record.model_id}")
        print(f"📊 Dataset: {eval_record.dataset}")
        print(f"📊 Metrics: {eval_record.metrics}")
        print(f"📊 Eval data path: {eval_record.eval_data_path}")

        assert eval_record.model_id == "iteration_1"
        assert eval_record.dataset == "cv"
        assert eval_record.metrics is not None
        assert eval_record.eval_data_path == "/output/metrics/iteration_1.json"

        print("\n✅ PASSED - Eval record created successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_3_eval_model_metrics_stores_all_metrics():
    """Test 3: EVAL_MODEL metrics stores all metrics correctly"""
    print("\n" + "=" * 70)
    print("TEST 3: EVAL_MODEL metrics Stores All Metrics")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-3",
            training_run_id="test-run-3",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Create iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 15:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 2",
            "data": {"config": {"current_iteration": 2, "total_iterations": 3}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process EVAL_TRAINING start
        eval_start_event = {
            "timestamp": "2025-11-06 15:05:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {"config": {"dataset": "test"}},
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Process EVAL_MODEL metrics with comprehensive metrics
        eval_metrics_event = {
            "timestamp": "2025-11-06 15:10:00",
            "level": "INFO",
            "phase": "EVAL_MODEL",
            "event": "metrics",
            "message": "Evaluation metrics computed",
            "data": {
                "config": {"dataset": "test", "threshold": 0.5},
                "metrics": {
                    "accuracy": 0.945,
                    "precision": 0.923,
                    "recall": 0.956,
                    "f1": 0.939,
                    "auc": 0.982,
                    "loss": 0.234,
                    "confusion_matrix": [[850, 50], [30, 920]],
                },
                "model_path": "/models/iteration_2.pth",
                "metrics_json_path": "/output/metrics/iteration_2.json",
            },
        }

        await monitor._process_log_line(json.dumps(eval_metrics_event))

        # Verify all metrics are stored
        # Get the most recent eval record (from this test)
        eval_record = await Eval.all().order_by("-created_at").first()

        assert eval_record is not None

        print(f"\n📊 Metrics stored: {eval_record.metrics}")
        print(f"📊 Accuracy: {eval_record.metrics.get('accuracy')}")
        print(f"📊 Precision: {eval_record.metrics.get('precision')}")
        print(f"📊 Recall: {eval_record.metrics.get('recall')}")
        print(f"📊 F1: {eval_record.metrics.get('f1')}")
        print(f"📊 AUC: {eval_record.metrics.get('auc')}")
        print(f"📊 Loss: {eval_record.metrics.get('loss')}")

        assert eval_record.metrics is not None
        assert eval_record.metrics.get("accuracy") == 0.945
        assert eval_record.metrics.get("precision") == 0.923
        assert eval_record.metrics.get("recall") == 0.956
        assert eval_record.metrics.get("f1") == 0.939
        assert eval_record.metrics.get("auc") == 0.982
        assert eval_record.metrics.get("loss") == 0.234
        assert eval_record.metrics.get("confusion_matrix") is not None

        print("\n✅ PASSED - All metrics stored correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_4_eval_training_end_records_duration():
    """Test 4: EVAL_TRAINING end event records duration"""
    print("\n" + "=" * 70)
    print("TEST 4: EVAL_TRAINING End Records Duration")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-4",
            training_run_id="test-run-4",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Create iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 16:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 1}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process EVAL_TRAINING start
        eval_start_event = {
            "timestamp": "2025-11-06 16:05:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {"config": {"dataset": "cv"}},
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Process EVAL_TRAINING end with duration
        eval_end_event = {
            "timestamp": "2025-11-06 16:20:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "end",
            "message": "Evaluation completed",
            "data": {"duration": 15.0},  # 15 minutes
        }

        await monitor._process_log_line(json.dumps(eval_end_event))

        # Verify duration was recorded
        eval_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.EVALUATION
        ).all()

        assert len(eval_steps) == 1
        eval_step = eval_steps[0]

        print(f"\n📊 Step time: {eval_step.step_time} minutes")
        print(f"📊 Completed at: {eval_step.completed_at}")

        assert eval_step.step_time is not None
        assert eval_step.step_time == 15.0
        assert eval_step.completed_at is not None
        assert eval_step.completed_at.year == 2025
        assert eval_step.completed_at.month == 11
        assert eval_step.completed_at.day == 6
        assert eval_step.completed_at.hour == 16
        assert eval_step.completed_at.minute == 20

        print("\n✅ PASSED - Duration recorded correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_5_eval_timestamp_accuracy():
    """Test 5: EVAL_TRAINING start uses event timestamp for created_at"""
    print("\n" + "=" * 70)
    print("TEST 5: EVAL_TRAINING Start Uses Event Timestamp")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-5",
            training_run_id="test-run-5",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Create iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 17:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 1}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process EVAL_TRAINING start with specific timestamp
        eval_start_event = {
            "timestamp": "2025-11-06 17:25:30",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {"config": {"dataset": "cv"}},
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Verify created_at matches event timestamp
        eval_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.EVALUATION
        ).all()

        assert len(eval_steps) == 1
        eval_step = eval_steps[0]

        print(f"\n📊 Created at: {eval_step.created_at}")
        print(f"📊 Expected: 2025-11-06 17:25:30")

        assert eval_step.created_at.year == 2025
        assert eval_step.created_at.month == 11
        assert eval_step.created_at.day == 6
        assert eval_step.created_at.hour == 17
        assert eval_step.created_at.minute == 25
        assert eval_step.created_at.second == 30

        print("\n✅ PASSED - created_at uses event timestamp")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_6_eval_without_iteration_context():
    """Test 6: EVAL_TRAINING without iteration context is handled gracefully"""
    print("\n" + "=" * 70)
    print("TEST 6: EVAL_TRAINING Without Iteration Context")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-6",
            training_run_id="test-run-6",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Process EVAL_TRAINING start WITHOUT iteration context
        eval_start_event = {
            "timestamp": "2025-11-06 18:00:00",
            "level": "INFO",
            "phase": "EVAL_TRAINING",
            "event": "start",
            "message": "Starting evaluation",
            "data": {"config": {"dataset": "cv"}},
        }

        await monitor._process_log_line(json.dumps(eval_start_event))

        # Verify no evaluation step was created (no iteration context)
        eval_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.EVALUATION
        ).all()

        print(f"\n📊 Number of evaluation steps created: {len(eval_steps)}")
        print(f"📊 Expected: 0 (no iteration context)")

        assert len(eval_steps) == 0

        print("\n✅ PASSED - No step created without iteration context")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_7_multiple_eval_iterations():
    """Test 7: Multiple evaluation iterations are handled correctly"""
    print("\n" + "=" * 70)
    print("TEST 7: Multiple Evaluation Iterations Handled Correctly")
    print("=" * 70)

    try:
        mock_ssh = create_mock_ssh_executor()

        job = await TrainingJob.create(
            project_id="test-project-7",
            training_run_id="test-run-7",
            status=TrainingJobStatus.RUNNING,
        )

        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )

        # Process multiple iterations with evaluations
        for iter_num in range(1, 4):
            # Create iteration context
            iteration_start_event = {
                "timestamp": f"2025-11-06 {10 + iter_num}:00:00",
                "level": "INFO",
                "phase": "ITERATION",
                "event": "start",
                "message": f"Starting iteration {iter_num}",
                "data": {
                    "config": {"current_iteration": iter_num, "total_iterations": 3}
                },
            }

            await monitor._process_log_line(json.dumps(iteration_start_event))

            # Process EVAL_TRAINING start
            eval_start_event = {
                "timestamp": f"2025-11-06 {10 + iter_num}:30:00",
                "level": "INFO",
                "phase": "EVAL_TRAINING",
                "event": "start",
                "message": "Starting evaluation",
                "data": {"config": {"dataset": "cv"}},
            }

            await monitor._process_log_line(json.dumps(eval_start_event))

            # Process EVAL_MODEL metrics
            eval_metrics_event = {
                "timestamp": f"2025-11-06 {10 + iter_num}:35:00",
                "level": "INFO",
                "phase": "EVAL_MODEL",
                "event": "metrics",
                "message": "Evaluation metrics computed",
                "data": {
                    "config": {"dataset": "cv"},
                    "metrics": {
                        "accuracy": 0.85 + (iter_num * 0.03),  # Improving accuracy
                        "f1": 0.82 + (iter_num * 0.03),
                    },
                    "model_path": f"/models/iteration_{iter_num}.pth",
                    "metrics_json_path": f"/output/metrics/iteration_{iter_num}.json",
                },
            }

            await monitor._process_log_line(json.dumps(eval_metrics_event))

        # Verify all evaluation steps and records were created
        eval_steps = (
            await TrainingIteration.filter(
                training_job=job, step_type=StepType.EVALUATION
            )
            .order_by("iteration_number")
            .all()
        )

        # Get the 3 most recent eval records (from this test)
        eval_records = await Eval.all().order_by("-created_at").limit(3)
        # Reverse to get chronological order
        eval_records = list(reversed(eval_records))

        print(f"\n📊 Total evaluation steps created: {len(eval_steps)}")
        print(f"📊 Total eval records created: {len(eval_records)}")

        assert len(eval_steps) == 3
        assert len(eval_records) == 3

        for i, (eval_step, eval_record) in enumerate(zip(eval_steps, eval_records), 1):
            print(f"\n📊 Iteration {i}:")
            print(f"   Step iteration number: {eval_step.iteration_number}")
            print(f"   Eval model_id: {eval_record.model_id}")
            print(f"   Accuracy: {eval_record.metrics.get('accuracy'):.3f}")
            print(f"   F1: {eval_record.metrics.get('f1'):.3f}")

            assert eval_step.iteration_number == i
            assert eval_record.model_id == f"iteration_{i}"
            assert abs(eval_record.metrics.get("accuracy") - (0.85 + i * 0.03)) < 0.001
            assert abs(eval_record.metrics.get("f1") - (0.82 + i * 0.03)) < 0.001

        print("\n✅ PASSED - Multiple evaluation iterations handled correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  TEST 14: EVAL EVENT PROCESSING".center(70))
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
    results["eval_training_start_creates_step"] = (
        await test_1_eval_training_start_creates_evaluation_step()
    )
    results["eval_model_metrics_creates_record"] = (
        await test_2_eval_model_metrics_creates_eval_record()
    )
    results["eval_model_metrics_stores_all"] = (
        await test_3_eval_model_metrics_stores_all_metrics()
    )
    results["eval_training_end_records_duration"] = (
        await test_4_eval_training_end_records_duration()
    )
    results["eval_timestamp"] = await test_5_eval_timestamp_accuracy()
    results["eval_without_context"] = await test_6_eval_without_iteration_context()
    results["multiple_eval_iterations"] = await test_7_multiple_eval_iterations()

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

    # Clean up test database
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_event_eval.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
