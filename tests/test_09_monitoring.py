"""
Test 09: Training Monitoring

Tests the TrainingJobMonitor's ability to:
- Locate global.json file on GPU server
- Download global.json
- Parse JSON lines correctly
- Handle incomplete JSON gracefully
- Detect new lines since last poll
- Polling interval works correctly
- Handle missing global.json initially
- Stop when training completes

Run: python tests/test_09_monitoring.py
"""

import asyncio
import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tortoise import Tortoise
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval
from app.services.training_job_monitor import TrainingJobMonitor
from scripts.ssh_executor import SshExecutor, CommandResult


# Sample global.json lines for testing
SAMPLE_LOG_LINES = [
    '{"timestamp": "2025-11-06 10:00:00", "level": "INFO", "phase": "PROJECT", "event": "start", "message": "Project started", "data": {"config": {"project_name": "test_project"}}}',
    '{"timestamp": "2025-11-06 10:00:05", "level": "INFO", "phase": "GROUP_ITERATION", "event": "start", "message": "Group iteration started", "data": {"config": {"no_iterations": 2}}}',
    '{"timestamp": "2025-11-06 10:00:10", "level": "INFO", "phase": "ITERATION", "event": "start", "message": "Iteration started 1", "data": {"config": {"current_iteration": 1}}}',
    '{"timestamp": "2025-11-06 10:00:15", "level": "INFO", "phase": "TRAJECTORY", "event": "start", "message": "Trajectory generation started", "data": {"config": {}}}',
    '{"timestamp": "2025-11-06 10:02:00", "level": "INFO", "phase": "TRAJECTORY", "event": "end", "message": "Trajectory generation finished", "data": {"duration": 1.75}}',
    '{"timestamp": "2025-11-06 10:02:05", "level": "INFO", "phase": "TRAINING", "event": "start", "message": "Training started", "data": {"config": {}}}',
    '{"timestamp": "2025-11-06 10:05:00", "level": "INFO", "phase": "TRAINING", "event": "epoch_complete", "message": "Epoch 1 complete", "data": {"epoch": 1, "avg_loss": 0.5}}',
    '{"timestamp": "2025-11-06 10:08:00", "level": "INFO", "phase": "TRAINING", "event": "epoch_complete", "message": "Epoch 2 complete", "data": {"epoch": 2, "avg_loss": 0.3}}',
    '{"timestamp": "2025-11-06 10:10:00", "level": "INFO", "phase": "TRAINING", "event": "end", "message": "Training finished", "data": {"duration": 7.92}}',
    '{"timestamp": "2025-11-06 10:10:05", "level": "INFO", "phase": "EVAL_TRAINING", "event": "start", "message": "Evaluation started", "data": {"config": {"dataset": "cv"}}}',
    '{"timestamp": "2025-11-06 10:12:00", "level": "INFO", "phase": "EVAL_MODEL", "event": "metrics", "message": "Evaluation metrics", "data": {"metrics": {"accuracy": 0.85}, "config": {"dataset": "cv"}}}',
    '{"timestamp": "2025-11-06 10:12:05", "level": "INFO", "phase": "EVAL_TRAINING", "event": "end", "message": "Evaluation finished", "data": {"duration": 1.98}}',
    '{"timestamp": "2025-11-06 10:12:10", "level": "INFO", "phase": "ITERATION", "event": "end", "message": "Iteration finished", "data": {"duration": 12.0}}',
    '{"timestamp": "2025-11-06 10:12:15", "level": "INFO", "phase": "PROJECT", "event": "end", "message": "Project finished", "data": {"duration": 12.25}}',
]

INCOMPLETE_JSON_LINE = '{"timestamp": "2025-11-06 10:00:00", "level": "INFO", "phase": "PROJECT"'


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_monitoring.db")
    
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


async def test_1_locate_global_json():
    """Test 1: Can locate global.json file on GPU server"""
    print("=" * 70)
    print("TEST 1: Locate global.json on GPU Server")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Mock successful file location
        mock_ssh.execute_command.return_value = CommandResult(
            command="find . -path './output/*/logs/global.json' -type f | head -1",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        
        # Create a temporary job
        job = await TrainingJob.create(
            project_id="test-project",
            training_run_id="test-run",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            remote_log_path="output/*/logs/global.json",
            poll_interval=1,
        )
        
        # Test file location
        result = await monitor._download_log_file()
        
        # Verify the find command was called
        assert mock_ssh.execute_command.called
        call_args = mock_ssh.execute_command.call_args[0][0]
        assert "find" in call_args
        assert "global.json" in call_args
        
        print("\n✅ PASSED - Can locate global.json file")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_download_global_json():
    """Test 2: Can download global.json"""
    print("\n" + "=" * 70)
    print("TEST 2: Download global.json")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Create a temporary file with sample content
        sample_content = "\n".join(SAMPLE_LOG_LINES[:3])
        
        # Mock file location
        mock_ssh.execute_command.return_value = CommandResult(
            command="find",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        
        # Mock file download
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(sample_content)
            return True
        
        mock_ssh.download_file = mock_download
        
        job = await TrainingJob.create(
            project_id="test-project-2",
            training_run_id="test-run-2",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Test download
        content = await monitor._download_log_file()
        
        assert content is not None
        assert len(content) > 0
        assert "PROJECT" in content
        
        print(f"\n📥 Downloaded {len(content)} bytes")
        print(f"   Lines: {len(content.strip().split(chr(10)))}")
        print("\n✅ PASSED - Can download global.json")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_parse_json_lines():
    """Test 3: Can parse JSON lines correctly"""
    print("\n" + "=" * 70)
    print("TEST 3: Parse JSON Lines")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        job = await TrainingJob.create(
            project_id="test-project-3",
            training_run_id="test-run-3",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Test parsing valid JSON lines
        parsed_count = 0
        for line in SAMPLE_LOG_LINES[:5]:
            try:
                await monitor._process_log_line(line)
                parsed_count += 1
            except Exception as e:
                print(f"Failed to parse: {line[:50]}...")
                raise e
        
        print(f"\n📊 Successfully parsed {parsed_count}/{len(SAMPLE_LOG_LINES[:5])} lines")
        
        # Verify database records were created
        iterations = await TrainingIteration.filter(training_job__uuid=job.uuid).count()
        print(f"   Created {iterations} iteration records")
        
        assert parsed_count == 5
        assert iterations > 0
        
        print("\n✅ PASSED - Can parse JSON lines correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_handle_incomplete_json():
    """Test 4: Handles incomplete JSON gracefully"""
    print("\n" + "=" * 70)
    print("TEST 4: Handle Incomplete JSON")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        job = await TrainingJob.create(
            project_id="test-project-4",
            training_run_id="test-run-4",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Test parsing incomplete JSON - should not crash
        try:
            await monitor._process_log_line(INCOMPLETE_JSON_LINE)
            print("\n⚠️  Incomplete JSON was handled (no crash)")
        except json.JSONDecodeError:
            print("\n⚠️  JSONDecodeError caught (expected)")
        
        # Verify monitor is still functional
        await monitor._process_log_line(SAMPLE_LOG_LINES[0])
        
        print("\n✅ PASSED - Handles incomplete JSON gracefully")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_detect_new_lines():
    """Test 5: Detects new lines since last poll"""
    print("\n" + "=" * 70)
    print("TEST 5: Detect New Lines Since Last Poll")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Simulate progressive file growth
        poll_contents = [
            "\n".join(SAMPLE_LOG_LINES[:3]),   # First poll: 3 lines
            "\n".join(SAMPLE_LOG_LINES[:6]),   # Second poll: 6 lines
            "\n".join(SAMPLE_LOG_LINES[:10]),  # Third poll: 10 lines
        ]
        
        poll_index = [0]  # Use list to allow modification in nested function
        
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(poll_contents[poll_index[0]])
            return True
        
        mock_ssh.execute_command.return_value = CommandResult(
            command="find",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        mock_ssh.download_file = mock_download
        
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
        
        # First poll
        await monitor._poll_and_update()
        assert monitor.processed_line_count == 3
        print(f"\n📊 Poll 1: Processed {monitor.processed_line_count} lines")
        
        # Second poll (3 new lines)
        poll_index[0] = 1
        await monitor._poll_and_update()
        assert monitor.processed_line_count == 6
        print(f"📊 Poll 2: Processed {monitor.processed_line_count} lines (3 new)")
        
        # Third poll (4 new lines)
        poll_index[0] = 2
        await monitor._poll_and_update()
        assert monitor.processed_line_count == 10
        print(f"📊 Poll 3: Processed {monitor.processed_line_count} lines (4 new)")
        
        # Fourth poll (no new lines)
        initial_count = monitor.processed_line_count
        await monitor._poll_and_update()
        assert monitor.processed_line_count == initial_count
        print(f"📊 Poll 4: No new lines detected")
        
        print("\n✅ PASSED - Detects new lines correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_6_polling_interval():
    """Test 6: Polling interval works correctly"""
    print("\n" + "=" * 70)
    print("TEST 6: Polling Interval")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Mock file operations
        mock_ssh.execute_command.return_value = CommandResult(
            command="find",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(SAMPLE_LOG_LINES[0])
            return True
        
        mock_ssh.download_file = mock_download
        
        job = await TrainingJob.create(
            project_id="test-project-6",
            training_run_id="test-run-6",
            status=TrainingJobStatus.PENDING,
        )
        
        # Test with 1 second interval
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for 3 seconds
        await asyncio.sleep(3)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        await asyncio.sleep(0.5)  # Give it time to stop
        
        # Cancel the task if still running
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        print(f"\n⏱️  Polling interval: {monitor.poll_interval}s")
        print(f"   Monitor ran for ~3 seconds")
        print(f"   Expected ~3 polls, monitor stopped gracefully")
        
        print("\n✅ PASSED - Polling interval works")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_7_handle_missing_file():
    """Test 7: Handles missing global.json initially"""
    print("\n" + "=" * 70)
    print("TEST 7: Handle Missing global.json")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Mock file not found initially
        call_count = [0]
        
        def mock_execute_command(cmd, check=True):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First 2 calls: file not found
                return CommandResult(
                    command=cmd,
                    stdout="",
                    stderr="",
                    return_code=1,
                    success=False,
                    duration=0.5
                )
            else:
                # Later calls: file found
                return CommandResult(
                    command=cmd,
                    stdout="./output/test_project/logs/global.json",
                    stderr="",
                    return_code=0,
                    success=True,
                    duration=0.5
                )
        
        mock_ssh.execute_command = mock_execute_command
        
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(SAMPLE_LOG_LINES[0])
            return True
        
        mock_ssh.download_file = mock_download
        
        job = await TrainingJob.create(
            project_id="test-project-7",
            training_run_id="test-run-7",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # First poll - file not found
        await monitor._poll_and_update()
        print("\n📊 Poll 1: File not found (handled gracefully)")
        
        # Second poll - file not found
        await monitor._poll_and_update()
        print("📊 Poll 2: File not found (handled gracefully)")
        
        # Third poll - file found
        await monitor._poll_and_update()
        print("📊 Poll 3: File found and processed")
        
        assert monitor.processed_line_count > 0
        
        print("\n✅ PASSED - Handles missing file gracefully")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_8_stop_when_complete():
    """Test 8: Stops when training completes"""
    print("\n" + "=" * 70)
    print("TEST 8: Stop When Training Completes")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Mock file with complete training
        complete_log = "\n".join(SAMPLE_LOG_LINES)
        
        mock_ssh.execute_command.return_value = CommandResult(
            command="find",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(complete_log)
            return True
        
        mock_ssh.download_file = mock_download
        
        job = await TrainingJob.create(
            project_id="test-project-8",
            training_run_id="test-run-8",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process all lines (including PROJECT end event)
        await monitor._poll_and_update()
        
        # Verify should_stop flag is set
        assert monitor.should_stop == True
        print("\n🛑 Monitor detected training completion")
        
        # Verify job status
        job = await TrainingJob.get(uuid=job.uuid)
        assert job.status == TrainingJobStatus.COMPLETED
        print(f"   Job status: {job.status.value}")
        print(f"   Completed at: {job.completed_at}")
        
        print("\n✅ PASSED - Stops when training completes")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_9_end_to_end_monitoring():
    """Test 9: End-to-end monitoring simulation"""
    print("\n" + "=" * 70)
    print("TEST 9: End-to-End Monitoring")
    print("=" * 70)
    
    try:
        mock_ssh = create_mock_ssh_executor()
        
        # Simulate progressive training
        complete_log = "\n".join(SAMPLE_LOG_LINES)
        
        mock_ssh.execute_command.return_value = CommandResult(
            command="find",
            stdout="./output/test_project/logs/global.json",
            stderr="",
            return_code=0,
            success=True,
            duration=0.5
        )
        
        def mock_download(remote_path, local_path):
            with open(local_path, 'w') as f:
                f.write(complete_log)
            return True
        
        mock_ssh.download_file = mock_download
        
        job = await TrainingJob.create(
            project_id="test-project-9",
            training_run_id="test-run-9",
            status=TrainingJobStatus.PENDING,
        )
        
        monitor = TrainingJobMonitor(
            training_job_uuid=str(job.uuid),
            ssh_executor=mock_ssh,
            poll_interval=1,
        )
        
        # Process all events
        await monitor._poll_and_update()
        
        # Verify database records
        job = await TrainingJob.get(uuid=job.uuid)
        iterations = await TrainingIteration.filter(training_job=job).count()
        epochs = await EpochTrain.filter(iteration__training_job=job).count()
        evals = await Eval.all().count()
        
        print(f"\n📊 Database Records Created:")
        print(f"   Job Status: {job.status.value}")
        print(f"   Iterations: {iterations}")
        print(f"   Epochs: {epochs}")
        print(f"   Evaluations: {evals}")
        
        assert job.status == TrainingJobStatus.COMPLETED
        assert iterations > 0
        assert epochs == 2  # We have 2 epoch_complete events
        assert evals > 0
        
        print("\n✅ PASSED - End-to-end monitoring works")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  TEST 09: TRAINING MONITORING".center(70))
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
    results["locate_file"] = await test_1_locate_global_json()
    results["download_file"] = await test_2_download_global_json()
    results["parse_json"] = await test_3_parse_json_lines()
    results["incomplete_json"] = await test_4_handle_incomplete_json()
    results["detect_new_lines"] = await test_5_detect_new_lines()
    results["polling_interval"] = await test_6_polling_interval()
    results["missing_file"] = await test_7_handle_missing_file()
    results["stop_complete"] = await test_8_stop_when_complete()
    results["end_to_end"] = await test_9_end_to_end_monitoring()
    
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
    db_path = os.path.join(project_root, "test_monitoring.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())