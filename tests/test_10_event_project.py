"""
Test 10: PROJECT Event Processing - Phase 4.1

Tests database updates from PROJECT phase events:
- PROJECT start: Updates job created_at with event timestamp
- PROJECT end: Marks job completed, sets completed_at, calculates time_taken
- Timestamp parsing (IST format)

Run: python tests/test_10_event_project.py
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
    db_path = os.path.join(project_root, "test_event_project.db")
    
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


async def test_1_project_start_updates_created_at():
    """Test 1: PROJECT start event updates job created_at"""
    print("=" * 70)
    print("TEST 1: PROJECT Start Updates created_at")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Create job with default created_at
        job = await TrainingJob.create(
            project_id="test-project-1",
            training_run_id="test-run-1",
            status=TrainingJobStatus.PENDING,
        )
        
        initial_created_at = job.created_at
        print(f"\n📊 Initial created_at: {initial_created_at}")
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process PROJECT start event with specific timestamp
        project_start_event = {
            "timestamp": "2025-11-06 10:00:00",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "start",
            "message": "Project started",
            "data": {"config": {"project_name": "test_project"}}
        }
        
        await monitor._process_log_line(json.dumps(project_start_event))
        
        # Verify created_at was updated
        job = await TrainingJob.get(uuid=job.uuid)
        print(f"📊 Updated created_at: {job.created_at}")
        
        # Check that created_at matches the event timestamp
        assert job.created_at is not None
        assert job.created_at.year == 2025
        assert job.created_at.month == 11
        assert job.created_at.day == 6
        assert job.created_at.hour == 10
        assert job.created_at.minute == 0
        
        print("\n✅ PASSED - created_at updated from PROJECT start event")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_project_end_marks_completed():
    """Test 2: PROJECT end event marks job as completed"""
    print("\n" + "=" * 70)
    print("TEST 2: PROJECT End Marks Job Completed")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        job = await TrainingJob.create(
            project_id="test-project-2",
            training_run_id="test-run-2",
            status=TrainingJobStatus.RUNNING,
        )
        
        print(f"\n📊 Initial status: {job.status.value}")
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process PROJECT end event
        project_end_event = {
            "timestamp": "2025-11-06 10:15:30",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "end",
            "message": "Project finished",
            "data": {"duration": 15.5}  # 15.5 minutes
        }
        
        await monitor._process_log_line(json.dumps(project_end_event))
        
        # Verify job is marked as completed
        job = await TrainingJob.get(uuid=job.uuid)
        print(f"📊 Updated status: {job.status.value}")
        
        assert job.status == TrainingJobStatus.COMPLETED
        assert job.completed_at is not None
        
        print(f"📊 Completed at: {job.completed_at}")
        
        print("\n✅ PASSED - Job marked as COMPLETED")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_project_end_sets_completed_at():
    """Test 3: PROJECT end event sets completed_at timestamp"""
    print("\n" + "=" * 70)
    print("TEST 3: PROJECT End Sets completed_at")
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
        
        # Process PROJECT end event with specific timestamp
        project_end_event = {
            "timestamp": "2025-11-06 10:20:45",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "end",
            "message": "Project finished",
            "data": {"duration": 20.75}
        }
        
        await monitor._process_log_line(json.dumps(project_end_event))
        
        # Verify completed_at is set
        job = await TrainingJob.get(uuid=job.uuid)
        
        assert job.completed_at is not None
        print(f"\n📊 Completed at: {job.completed_at}")
        
        # Verify timestamp matches event
        assert job.completed_at.year == 2025
        assert job.completed_at.month == 11
        assert job.completed_at.day == 6
        assert job.completed_at.hour == 10
        assert job.completed_at.minute == 20
        
        print("\n✅ PASSED - completed_at set correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_project_end_calculates_time_taken():
    """Test 4: PROJECT end event calculates time_taken"""
    print("\n" + "=" * 70)
    print("TEST 4: PROJECT End Calculates time_taken")
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
        
        # Process PROJECT end event with duration
        # Duration in event is in minutes
        project_end_event = {
            "timestamp": "2025-11-06 10:25:00",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "end",
            "message": "Project finished",
            "data": {"duration": 12.5}  # 12.5 minutes = 750 seconds
        }
        
        await monitor._process_log_line(json.dumps(project_end_event))
        
        # Verify time_taken is calculated
        job = await TrainingJob.get(uuid=job.uuid)
        
        assert job.time_taken is not None
        print(f"\n📊 Time taken: {job.time_taken} seconds")
        print(f"📊 Expected: 750 seconds (12.5 minutes)")
        
        # Duration from event is in minutes, converted to seconds
        expected_seconds = int(12.5 * 60)
        assert job.time_taken == expected_seconds
        
        print("\n✅ PASSED - time_taken calculated correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_timestamp_parsing_ist():
    """Test 5: Timestamps are parsed correctly (IST format)"""
    print("\n" + "=" * 70)
    print("TEST 5: Timestamp Parsing (IST)")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        job = await TrainingJob.create(
            project_id="test-project-5",
            training_run_id="test-run-5",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Test various timestamp formats
        test_timestamps = [
            "2025-11-06 10:00:00",
            "2025-11-06 14:30:45",
            "2025-12-31 23:59:59",
        ]
        
        for ts_str in test_timestamps:
            project_start_event = {
                "timestamp": ts_str,
                "level": "INFO",
                "phase": "PROJECT",
                "event": "start",
                "message": "Project started",
                "data": {"config": {}}
            }
            
            await monitor._process_log_line(json.dumps(project_start_event))
            
            job = await TrainingJob.get(uuid=job.uuid)
            print(f"\n📊 Input: {ts_str}")
            print(f"   Parsed: {job.created_at}")
            
            # Verify parsing
            assert job.created_at is not None
            
        print("\n✅ PASSED - Timestamps parsed correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_6_full_project_lifecycle():
    """Test 6: Full PROJECT lifecycle (start to end)"""
    print("\n" + "=" * 70)
    print("TEST 6: Full PROJECT Lifecycle")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        job = await TrainingJob.create(
            project_id="test-project-6",
            training_run_id="test-run-6",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process PROJECT start
        project_start_event = {
            "timestamp": "2025-11-06 10:00:00",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "start",
            "message": "Project started",
            "data": {"config": {"project_name": "full_test"}}
        }
        
        await monitor._process_log_line(json.dumps(project_start_event))
        
        job = await TrainingJob.get(uuid=job.uuid)
        start_time = job.created_at
        print(f"\n📊 Project started at: {start_time}")
        
        # Process PROJECT end
        project_end_event = {
            "timestamp": "2025-11-06 10:18:30",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "end",
            "message": "Project finished",
            "data": {"duration": 18.5}  # 18.5 minutes
        }
        
        await monitor._process_log_line(json.dumps(project_end_event))
        
        job = await TrainingJob.get(uuid=job.uuid)
        
        print(f"📊 Project completed at: {job.completed_at}")
        print(f"📊 Status: {job.status.value}")
        print(f"📊 Time taken: {job.time_taken} seconds")
        
        # Verify all fields
        assert job.created_at is not None
        assert job.completed_at is not None
        assert job.status == TrainingJobStatus.COMPLETED
        assert job.time_taken == int(18.5 * 60)  # 1110 seconds
        assert monitor.should_stop == True  # Monitor should stop after PROJECT end
        
        print("\n✅ PASSED - Full lifecycle works correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_7_time_taken_fallback_calculation():
    """Test 7: time_taken fallback when duration not in event"""
    print("\n" + "=" * 70)
    print("TEST 7: time_taken Fallback Calculation")
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
        
        # Set project start time manually
        project_start_event = {
            "timestamp": "2025-11-06 10:00:00",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "start",
            "message": "Project started",
            "data": {"config": {}}
        }
        
        await monitor._process_log_line(json.dumps(project_start_event))
        
        # Process PROJECT end WITHOUT duration in data
        project_end_event = {
            "timestamp": "2025-11-06 10:10:00",
            "level": "INFO",
            "phase": "PROJECT",
            "event": "end",
            "message": "Project finished",
            "data": {}  # No duration field
        }
        
        await monitor._process_log_line(json.dumps(project_end_event))
        
        # Verify time_taken is calculated from timestamps
        job = await TrainingJob.get(uuid=job.uuid)
        
        assert job.time_taken is not None
        print(f"\n📊 Time taken (calculated): {job.time_taken} seconds")
        print(f"📊 Expected: 600 seconds (10 minutes)")
        
        # Should be 10 minutes = 600 seconds
        assert job.time_taken == 600
        
        print("\n✅ PASSED - Fallback calculation works")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  TEST 10: PROJECT EVENT PROCESSING".center(70))
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
    results["project_start_created_at"] = await test_1_project_start_updates_created_at()
    results["project_end_completed"] = await test_2_project_end_marks_completed()
    results["project_end_completed_at"] = await test_3_project_end_sets_completed_at()
    results["project_end_time_taken"] = await test_4_project_end_calculates_time_taken()
    results["timestamp_parsing"] = await test_5_timestamp_parsing_ist()
    results["full_lifecycle"] = await test_6_full_project_lifecycle()
    results["time_taken_fallback"] = await test_7_time_taken_fallback_calculation()
    
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
    db_path = os.path.join(project_root, "test_event_project.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())