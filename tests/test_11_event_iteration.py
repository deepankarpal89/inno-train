"""
Test 11: ITERATION Event Processing - Phase 4.2

Tests database updates from ITERATION phase events:
- GROUP_ITERATION start: Updates training_config
- ITERATION start: Creates TrainingIteration record
- ITERATION end: Updates completed_at and time_taken
- Multiple iterations handled correctly

Run: python tests/test_11_event_iteration.py
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
from app.services.training_job_monitor import TrainingJobMonitor
from scripts.ssh_executor import SshExecutor, CommandResult


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_event_iteration.db")
    
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


async def test_1_group_iteration_updates_training_config():
    """Test 1: GROUP_ITERATION start event updates training_config"""
    print("=" * 70)
    print("TEST 1: GROUP_ITERATION Start Updates training_config")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Create job
        job = await TrainingJob.create(
            project_id="test-project-1",
            training_run_id="test-run-1",
            status=TrainingJobStatus.RUNNING,
        )
        
        print(f"\n📊 Initial training_config: {job.training_config}")
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process GROUP_ITERATION start event
        group_iteration_event = {
            "timestamp": "2025-11-06 10:05:00",
            "level": "INFO",
            "phase": "GROUP_ITERATION",
            "event": "start",
            "message": "Starting group iteration",
            "data": {
                "config": {
                    "no_iterations": 3,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "model_name": "test_model"
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(group_iteration_event))
        
        # Verify training_config was updated
        job = await TrainingJob.get(uuid=job.uuid)
        print(f"📊 Updated training_config: {job.training_config}")
        
        assert job.training_config is not None
        assert job.training_config.get("no_iterations") == 3
        assert job.training_config.get("learning_rate") == 0.001
        assert job.training_config.get("batch_size") == 32
        assert job.training_config.get("model_name") == "test_model"
        
        print("\n✅ PASSED - training_config updated from GROUP_ITERATION start")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_iteration_start_creates_record():
    """Test 2: ITERATION start event creates TrainingIteration record"""
    print("\n" + "=" * 70)
    print("TEST 2: ITERATION Start Creates TrainingIteration Record")
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
        
        # Process ITERATION start event
        iteration_start_event = {
            "timestamp": "2025-11-06 10:10:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 3
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Verify TrainingIteration record was created
        iterations = await TrainingIteration.filter(training_job=job).all()
        print(f"\n📊 Number of iterations created: {len(iterations)}")
        
        assert len(iterations) == 1
        iteration = iterations[0]
        
        print(f"📊 Iteration number: {iteration.iteration_number}")
        print(f"📊 Step type: {iteration.step_type}")
        print(f"📊 Created at: {iteration.created_at}")
        print(f"📊 Step config: {iteration.step_config}")
        
        assert iteration.iteration_number == 1
        assert iteration.step_type == StepType.ITERATION
        assert iteration.step_config.get("current_iteration") == 1
        assert iteration.step_config.get("total_iterations") == 3
        
        print("\n✅ PASSED - TrainingIteration record created")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_iteration_end_updates_completed_at():
    """Test 3: ITERATION end event updates completed_at"""
    print("\n" + "=" * 70)
    print("TEST 3: ITERATION End Updates completed_at")
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
        
        # Process ITERATION start
        iteration_start_event = {
            "timestamp": "2025-11-06 10:15:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 3
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process ITERATION end
        iteration_end_event = {
            "timestamp": "2025-11-06 10:25:30",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "end",
            "message": "Iteration 1 completed",
            "data": {
                "duration": 10.5  # 10.5 minutes
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_end_event))
        
        # Verify completed_at is set
        iterations = await TrainingIteration.filter(training_job=job).all()
        assert len(iterations) == 1
        iteration = iterations[0]
        
        print(f"\n📊 Completed at: {iteration.completed_at}")
        
        assert iteration.completed_at is not None
        assert iteration.completed_at.year == 2025
        assert iteration.completed_at.month == 11
        assert iteration.completed_at.day == 6
        assert iteration.completed_at.hour == 10
        assert iteration.completed_at.minute == 25
        
        print("\n✅ PASSED - completed_at updated correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_iteration_end_updates_time_taken():
    """Test 4: ITERATION end event updates time_taken"""
    print("\n" + "=" * 70)
    print("TEST 4: ITERATION End Updates time_taken")
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
        
        # Process ITERATION start
        iteration_start_event = {
            "timestamp": "2025-11-06 10:30:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 3
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process ITERATION end with duration
        iteration_end_event = {
            "timestamp": "2025-11-06 10:45:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "end",
            "message": "Iteration 1 completed",
            "data": {
                "duration": 15.0  # 15 minutes
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_end_event))
        
        # Verify time_taken is set
        iterations = await TrainingIteration.filter(training_job=job).all()
        assert len(iterations) == 1
        iteration = iterations[0]
        
        print(f"\n📊 Time taken: {iteration.time_taken} minutes")
        print(f"📊 Expected: 15.0 minutes")
        
        assert iteration.time_taken is not None
        assert iteration.time_taken == 15.0
        
        print("\n✅ PASSED - time_taken updated correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_multiple_iterations_handled():
    """Test 5: Multiple iterations are handled correctly"""
    print("\n" + "=" * 70)
    print("TEST 5: Multiple Iterations Handled Correctly")
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
        
        # Process GROUP_ITERATION start
        group_iteration_event = {
            "timestamp": "2025-11-06 11:00:00",
            "level": "INFO",
            "phase": "GROUP_ITERATION",
            "event": "start",
            "message": "Starting group iteration",
            "data": {
                "config": {
                    "no_iterations": 3,
                    "learning_rate": 0.001
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(group_iteration_event))
        
        # Process 3 iterations
        for i in range(1, 4):
            # Start iteration
            iteration_start_event = {
                "timestamp": f"2025-11-06 11:{i*10:02d}:00",
                "level": "INFO",
                "phase": "ITERATION",
                "event": "start",
                "message": f"Starting iteration {i}",
                "data": {
                    "config": {
                        "current_iteration": i,
                        "total_iterations": 3
                    }
                }
            }
            
            await monitor._process_log_line(json.dumps(iteration_start_event))
            
            # End iteration
            iteration_end_event = {
                "timestamp": f"2025-11-06 11:{i*10 + 5:02d}:00",
                "level": "INFO",
                "phase": "ITERATION",
                "event": "end",
                "message": f"Iteration {i} completed",
                "data": {
                    "duration": 5.0 + i  # Variable duration
                }
            }
            
            await monitor._process_log_line(json.dumps(iteration_end_event))
        
        # Verify all iterations were created
        iterations = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.ITERATION
        ).order_by("iteration_number").all()
        
        print(f"\n📊 Total iterations created: {len(iterations)}")
        
        assert len(iterations) == 3
        
        for i, iteration in enumerate(iterations, 1):
            print(f"\n📊 Iteration {i}:")
            print(f"   Number: {iteration.iteration_number}")
            print(f"   Completed: {iteration.completed_at is not None}")
            print(f"   Time taken: {iteration.time_taken} minutes")
            
            assert iteration.iteration_number == i
            assert iteration.completed_at is not None
            assert iteration.time_taken == 5.0 + i
        
        # Verify training_config was updated
        job = await TrainingJob.get(uuid=job.uuid)
        assert job.training_config is not None
        assert job.training_config.get("no_iterations") == 3
        
        print("\n✅ PASSED - Multiple iterations handled correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_6_iteration_start_timestamp():
    """Test 6: ITERATION start uses event timestamp for created_at"""
    print("\n" + "=" * 70)
    print("TEST 6: ITERATION Start Uses Event Timestamp")
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
        
        # Process ITERATION start with specific timestamp
        iteration_start_event = {
            "timestamp": "2025-11-06 12:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 1
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Verify created_at matches event timestamp
        iterations = await TrainingIteration.filter(training_job=job).all()
        assert len(iterations) == 1
        iteration = iterations[0]
        
        print(f"\n📊 Created at: {iteration.created_at}")
        print(f"📊 Expected: 2025-11-06 12:00:00")
        
        assert iteration.created_at.year == 2025
        assert iteration.created_at.month == 11
        assert iteration.created_at.day == 6
        assert iteration.created_at.hour == 12
        assert iteration.created_at.minute == 0
        
        print("\n✅ PASSED - created_at uses event timestamp")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_7_iteration_config_stored():
    """Test 7: ITERATION config is stored in step_config"""
    print("\n" + "=" * 70)
    print("TEST 7: ITERATION Config Stored in step_config")
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
        
        # Process ITERATION start with detailed config
        iteration_start_event = {
            "timestamp": "2025-11-06 13:00:00",
            "level": "INFO",
            "phase": "ITERATION",
            "event": "start",
            "message": "Starting iteration 1",
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 5,
                    "learning_rate": 0.0001,
                    "batch_size": 64,
                    "epochs": 10,
                    "custom_param": "test_value"
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Verify step_config is stored correctly
        iterations = await TrainingIteration.filter(training_job=job).all()
        assert len(iterations) == 1
        iteration = iterations[0]
        
        print(f"\n📊 Step config: {iteration.step_config}")
        
        assert iteration.step_config is not None
        assert iteration.step_config.get("current_iteration") == 1
        assert iteration.step_config.get("total_iterations") == 5
        assert iteration.step_config.get("learning_rate") == 0.0001
        assert iteration.step_config.get("batch_size") == 64
        assert iteration.step_config.get("epochs") == 10
        assert iteration.step_config.get("custom_param") == "test_value"
        
        print("\n✅ PASSED - step_config stored correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  TEST 11: ITERATION EVENT PROCESSING".center(70))
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
    results["group_iteration_config"] = await test_1_group_iteration_updates_training_config()
    results["iteration_start_creates"] = await test_2_iteration_start_creates_record()
    results["iteration_end_completed_at"] = await test_3_iteration_end_updates_completed_at()
    results["iteration_end_time_taken"] = await test_4_iteration_end_updates_time_taken()
    results["multiple_iterations"] = await test_5_multiple_iterations_handled()
    results["iteration_timestamp"] = await test_6_iteration_start_timestamp()
    results["iteration_config_stored"] = await test_7_iteration_config_stored()
    
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
    db_path = os.path.join(project_root, "test_event_iteration.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())