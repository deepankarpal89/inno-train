"""
Test 13: TRAINING Event Processing - Phase 4.4

Tests database updates from TRAINING phase events:
- TRAINING start: Creates training step
- TRAINING epoch_complete: Creates EpochTrain record
- TRAINING epoch_complete: Records avg_loss
- TRAINING end: Records duration
- Multiple epochs handled correctly

Run: python tests/test_13_event_training.py
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
from models.epoch_train import EpochTrain
from app.services.training_job_monitor import TrainingJobMonitor
from scripts.ssh_executor import SshExecutor, CommandResult


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_event_training.db")

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


async def test_1_training_start_creates_training_step():
    """Test 1: TRAINING start event creates training step"""
    print("=" * 70)
    print("TEST 1: TRAINING Start Creates Training Step")
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

        # Process TRAINING start event
        training_start_event = {
            "timestamp": "2025-11-06 14:05:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {
                "config": {
                    "num_epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                }
            },
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Verify training step was created
        training_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).all()

        print(f"\n📊 Number of training steps created: {len(training_steps)}")

        assert len(training_steps) == 1
        training_step = training_steps[0]

        print(f"📊 Step type: {training_step.step_type}")
        print(f"📊 Iteration number: {training_step.iteration_number}")
        print(f"📊 Created at: {training_step.created_at}")
        print(f"📊 Step config: {training_step.step_config}")

        assert training_step.step_type == StepType.TRAINING
        assert training_step.iteration_number == 1
        assert training_step.step_config is not None
        assert training_step.step_config.get("num_epochs") == 5
        assert training_step.step_config.get("batch_size") == 32
        assert training_step.step_config.get("learning_rate") == 0.001
        assert training_step.step_config.get("optimizer") == "adam"

        print("\n✅ PASSED - training step created successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_2_training_epoch_complete_creates_epoch_record():
    """Test 2: TRAINING epoch_complete event creates EpochTrain record"""
    print("\n" + "=" * 70)
    print("TEST 2: TRAINING epoch_complete Creates EpochTrain Record")
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

        # Process TRAINING start
        training_start_event = {
            "timestamp": "2025-11-06 14:15:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 3}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Process TRAINING epoch_complete event
        epoch_complete_event = {
            "timestamp": "2025-11-06 14:20:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "epoch_complete",
            "message": "Epoch 1 completed",
            "data": {
                "epoch": 1,
                "avg_loss": 0.523,
                "model_path": "/models/epoch_1.pth",
                "optimizer_path": "/models/optimizer_1.pth",
            },
        }

        await monitor._process_log_line(json.dumps(epoch_complete_event))

        # Verify EpochTrain record was created
        epochs = await EpochTrain.all()

        print(f"\n📊 Number of epoch records created: {len(epochs)}")

        assert len(epochs) == 1
        epoch = epochs[0]

        print(f"📊 Epoch number: {epoch.epoch_number}")
        print(f"📊 Iteration number: {epoch.iteration_number}")
        print(f"📊 Model path: {epoch.model_path}")
        print(f"📊 Optimizer path: {epoch.optimizer_path}")
        print(f"📊 Metrics: {epoch.metrics}")

        assert epoch.epoch_number == 1
        assert epoch.iteration_number == 1
        assert epoch.model_path == "/models/epoch_1.pth"
        assert epoch.optimizer_path == "/models/optimizer_1.pth"
        assert epoch.metrics is not None

        print("\n✅ PASSED - EpochTrain record created successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_3_training_epoch_complete_records_avg_loss():
    """Test 3: TRAINING epoch_complete records avg_loss in metrics"""
    print("\n" + "=" * 70)
    print("TEST 3: TRAINING epoch_complete Records avg_loss")
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
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 1}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process TRAINING start
        training_start_event = {
            "timestamp": "2025-11-06 15:05:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 1}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Process TRAINING epoch_complete with avg_loss
        epoch_complete_event = {
            "timestamp": "2025-11-06 15:10:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "epoch_complete",
            "message": "Epoch 1 completed",
            "data": {
                "epoch": 1,
                "avg_loss": 0.342,
                "accuracy": 0.89,
                "learning_rate": 0.001,
            },
        }

        await monitor._process_log_line(json.dumps(epoch_complete_event))

        # Verify avg_loss is recorded in metrics
        # Get the training step first to filter epochs correctly
        training_step = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).first()

        epochs = await EpochTrain.filter(iteration=training_step).all()

        assert len(epochs) == 1
        epoch = epochs[0]

        print(f"\n📊 Metrics: {epoch.metrics}")
        print(f"📊 avg_loss: {epoch.metrics.get('avg_loss')}")
        print(f"📊 accuracy: {epoch.metrics.get('accuracy')}")
        print(f"📊 learning_rate: {epoch.metrics.get('learning_rate')}")

        assert epoch.metrics is not None
        assert epoch.metrics.get("avg_loss") == 0.342
        assert epoch.metrics.get("accuracy") == 0.89
        assert epoch.metrics.get("learning_rate") == 0.001

        print("\n✅ PASSED - avg_loss recorded correctly in metrics")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_4_training_end_records_duration():
    """Test 4: TRAINING end event records duration"""
    print("\n" + "=" * 70)
    print("TEST 4: TRAINING End Records Duration")
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

        # Process TRAINING start
        training_start_event = {
            "timestamp": "2025-11-06 16:05:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 5}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Process TRAINING end with duration
        training_end_event = {
            "timestamp": "2025-11-06 16:25:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "end",
            "message": "Training completed",
            "data": {"duration": 20.5},  # 20.5 minutes
        }

        await monitor._process_log_line(json.dumps(training_end_event))

        # Verify duration was recorded
        training_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).all()

        assert len(training_steps) == 1
        training_step = training_steps[0]

        print(f"\n📊 Step time: {training_step.step_time} minutes")
        print(f"📊 Completed at: {training_step.completed_at}")

        assert training_step.step_time is not None
        assert training_step.step_time == 20.5
        assert training_step.completed_at is not None
        assert training_step.completed_at.year == 2025
        assert training_step.completed_at.month == 11
        assert training_step.completed_at.day == 6
        assert training_step.completed_at.hour == 16
        assert training_step.completed_at.minute == 25

        print("\n✅ PASSED - Duration recorded correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_5_multiple_epochs_handled_correctly():
    """Test 5: Multiple epochs are handled correctly"""
    print("\n" + "=" * 70)
    print("TEST 5: Multiple Epochs Handled Correctly")
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

        # Process TRAINING start
        training_start_event = {
            "timestamp": "2025-11-06 17:05:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 5, "batch_size": 64}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Process multiple epoch_complete events
        for epoch_num in range(1, 6):
            epoch_complete_event = {
                "timestamp": f"2025-11-06 17:{5 + epoch_num * 3:02d}:00",
                "level": "INFO",
                "phase": "TRAINING",
                "event": "epoch_complete",
                "message": f"Epoch {epoch_num} completed",
                "data": {
                    "epoch": epoch_num,
                    "avg_loss": 1.0 - (epoch_num * 0.15),  # Decreasing loss
                    "accuracy": 0.5 + (epoch_num * 0.08),  # Increasing accuracy
                    "model_path": f"/models/epoch_{epoch_num}.pth",
                    "optimizer_path": f"/models/optimizer_{epoch_num}.pth",
                },
            }

            await monitor._process_log_line(json.dumps(epoch_complete_event))

        # Verify all epochs were created
        # Get the training step first to filter epochs correctly
        training_step = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).first()

        epochs = (
            await EpochTrain.filter(iteration=training_step)
            .order_by("epoch_number")
            .all()
        )

        print(f"\n📊 Total epochs created: {len(epochs)}")

        assert len(epochs) == 5

        for i, epoch in enumerate(epochs, 1):
            print(f"\n📊 Epoch {i}:")
            print(f"   Epoch number: {epoch.epoch_number}")
            print(f"   avg_loss: {epoch.metrics.get('avg_loss'):.3f}")
            accuracy = epoch.metrics.get("accuracy")
            print(
                f"   accuracy: {accuracy:.3f}"
                if accuracy is not None
                else "   accuracy: None"
            )
            print(f"   Model path: {epoch.model_path}")

            assert epoch.epoch_number == i
            assert epoch.iteration_number == 1
            assert abs(epoch.metrics.get("avg_loss") - (1.0 - i * 0.15)) < 0.001
            assert abs(epoch.metrics.get("accuracy") - (0.5 + i * 0.08)) < 0.001
            assert epoch.model_path == f"/models/epoch_{i}.pth"
            assert epoch.optimizer_path == f"/models/optimizer_{i}.pth"

        print("\n✅ PASSED - Multiple epochs handled correctly")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_6_training_timestamp_accuracy():
    """Test 6: TRAINING start uses event timestamp for created_at"""
    print("\n" + "=" * 70)
    print("TEST 6: TRAINING Start Uses Event Timestamp")
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

        # Create iteration context
        iteration_start_event = {
            "timestamp": "2025-11-06 18:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {"config": {"current_iteration": 1, "total_iterations": 1}},
        }

        await monitor._process_log_line(json.dumps(iteration_start_event))

        # Process TRAINING start with specific timestamp
        training_start_event = {
            "timestamp": "2025-11-06 18:15:45",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 3}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Verify created_at matches event timestamp
        training_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).all()

        assert len(training_steps) == 1
        training_step = training_steps[0]

        print(f"\n📊 Created at: {training_step.created_at}")
        print(f"📊 Expected: 2025-11-06 18:15:45")

        assert training_step.created_at.year == 2025
        assert training_step.created_at.month == 11
        assert training_step.created_at.day == 6
        assert training_step.created_at.hour == 18
        assert training_step.created_at.minute == 15
        assert training_step.created_at.second == 45

        print("\n✅ PASSED - created_at uses event timestamp")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_7_training_without_iteration_context():
    """Test 7: TRAINING without iteration context is handled gracefully"""
    print("\n" + "=" * 70)
    print("TEST 7: TRAINING Without Iteration Context")
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

        # Process TRAINING start WITHOUT iteration context
        training_start_event = {
            "timestamp": "2025-11-06 19:00:00",
            "level": "INFO",
            "phase": "TRAINING",
            "event": "start",
            "message": "Starting training",
            "data": {"config": {"num_epochs": 5}},
        }

        await monitor._process_log_line(json.dumps(training_start_event))

        # Verify no training step was created (no iteration context)
        training_steps = await TrainingIteration.filter(
            training_job=job, step_type=StepType.TRAINING
        ).all()

        print(f"\n📊 Number of training steps created: {len(training_steps)}")
        print(f"📊 Expected: 0 (no iteration context)")

        assert len(training_steps) == 0

        print("\n✅ PASSED - No step created without iteration context")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  TEST 13: TRAINING EVENT PROCESSING".center(70))
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
    results["training_start_creates_step"] = (
        await test_1_training_start_creates_training_step()
    )
    results["epoch_complete_creates_record"] = (
        await test_2_training_epoch_complete_creates_epoch_record()
    )
    results["epoch_complete_records_avg_loss"] = (
        await test_3_training_epoch_complete_records_avg_loss()
    )
    results["training_end_records_duration"] = (
        await test_4_training_end_records_duration()
    )
    results["multiple_epochs_handled"] = (
        await test_5_multiple_epochs_handled_correctly()
    )
    results["training_timestamp"] = await test_6_training_timestamp_accuracy()
    results["training_without_context"] = (
        await test_7_training_without_iteration_context()
    )

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
    db_path = os.path.join(project_root, "test_event_training.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
