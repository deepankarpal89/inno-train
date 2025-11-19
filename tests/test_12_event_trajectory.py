"""
Test 12: TRAJECTORY Event Processing - Phase 4.3

Tests database updates from TRAJECTORY phase events:
- TRAJECTORY start: Creates traj_gen step
- TRAJECTORY end: Records duration
- Links to correct iteration

Run: python tests/test_12_event_trajectory.py
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
    db_path = os.path.join(project_root, "test_event_trajectory.db")
    
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


async def test_1_trajectory_start_creates_traj_gen_step():
    """Test 1: TRAJECTORY start event creates traj_gen step"""
    print("=" * 70)
    print("TEST 1: TRAJECTORY Start Creates traj_gen Step")
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
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 3
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process TRAJECTORY start event
        trajectory_start_event = {
            "timestamp": "2025-11-06 14:05:00",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "start",
            "message": "Starting trajectory generation",
            "data": {
                "config": {
                    "num_trajectories": 100,
                    "max_length": 512,
                    "temperature": 0.8
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_start_event))
        
        # Verify traj_gen step was created
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).all()
        
        print(f"\n📊 Number of traj_gen steps created: {len(traj_steps)}")
        
        assert len(traj_steps) == 1
        traj_step = traj_steps[0]
        
        print(f"📊 Step type: {traj_step.step_type}")
        print(f"📊 Iteration number: {traj_step.iteration_number}")
        print(f"📊 Created at: {traj_step.created_at}")
        print(f"📊 Step config: {traj_step.step_config}")
        
        assert traj_step.step_type == StepType.TRAJ_GEN
        assert traj_step.iteration_number == 1
        assert traj_step.step_config is not None
        assert traj_step.step_config.get("num_trajectories") == 100
        assert traj_step.step_config.get("max_length") == 512
        assert traj_step.step_config.get("temperature") == 0.8
        
        print("\n✅ PASSED - traj_gen step created successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_trajectory_end_records_duration():
    """Test 2: TRAJECTORY end event records duration"""
    print("\n" + "=" * 70)
    print("TEST 2: TRAJECTORY End Records Duration")
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
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 3
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process TRAJECTORY start
        trajectory_start_event = {
            "timestamp": "2025-11-06 14:15:00",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "start",
            "message": "Starting trajectory generation",
            "data": {
                "config": {
                    "num_trajectories": 50
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_start_event))
        
        # Process TRAJECTORY end with duration
        trajectory_end_event = {
            "timestamp": "2025-11-06 14:30:00",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "end",
            "message": "Trajectory generation completed",
            "data": {
                "duration": 15.5  # 15.5 minutes
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_end_event))
        
        # Verify duration was recorded
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).all()
        
        assert len(traj_steps) == 1
        traj_step = traj_steps[0]
        
        print(f"\n📊 Step time: {traj_step.step_time} minutes")
        print(f"📊 Completed at: {traj_step.completed_at}")
        
        assert traj_step.step_time is not None
        assert traj_step.step_time == 15.5
        assert traj_step.completed_at is not None
        assert traj_step.completed_at.year == 2025
        assert traj_step.completed_at.month == 11
        assert traj_step.completed_at.day == 6
        assert traj_step.completed_at.hour == 14
        assert traj_step.completed_at.minute == 30
        
        print("\n✅ PASSED - Duration recorded correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_trajectory_links_to_correct_iteration():
    """Test 3: TRAJECTORY step links to correct iteration"""
    print("\n" + "=" * 70)
    print("TEST 3: TRAJECTORY Links to Correct Iteration")
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
        
        # Create multiple iterations with trajectory steps
        for i in range(1, 4):
            # Start iteration
            iteration_start_event = {
                "timestamp": f"2025-11-06 15:{i*10:02d}:00",
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
            
            # Start trajectory
            trajectory_start_event = {
                "timestamp": f"2025-11-06 15:{i*10 + 1:02d}:00",
                "level": "INFO",
                "phase": "TRAJECTORY",
                "event": "start",
                "message": f"Starting trajectory generation for iteration {i}",
                "data": {
                    "config": {
                        "iteration": i,
                        "num_trajectories": 50 * i
                    }
                }
            }
            
            await monitor._process_log_line(json.dumps(trajectory_start_event))
            
            # End trajectory
            trajectory_end_event = {
                "timestamp": f"2025-11-06 15:{i*10 + 5:02d}:00",
                "level": "INFO",
                "phase": "TRAJECTORY",
                "event": "end",
                "message": f"Trajectory generation completed for iteration {i}",
                "data": {
                    "duration": 4.0 + i
                }
            }
            
            await monitor._process_log_line(json.dumps(trajectory_end_event))
        
        # Verify all trajectory steps were created and linked correctly
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).order_by("iteration_number").all()
        
        print(f"\n📊 Total trajectory steps created: {len(traj_steps)}")
        
        assert len(traj_steps) == 3
        
        for i, traj_step in enumerate(traj_steps, 1):
            print(f"\n📊 Trajectory Step {i}:")
            print(f"   Iteration number: {traj_step.iteration_number}")
            print(f"   Step time: {traj_step.step_time} minutes")
            print(f"   Num trajectories: {traj_step.step_config.get('num_trajectories')}")
            
            assert traj_step.iteration_number == i
            assert traj_step.step_time == 4.0 + i
            assert traj_step.step_config.get("num_trajectories") == 50 * i
            assert traj_step.completed_at is not None
        
        print("\n✅ PASSED - Trajectory steps linked to correct iterations")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_trajectory_timestamp_accuracy():
    """Test 4: TRAJECTORY start uses event timestamp for created_at"""
    print("\n" + "=" * 70)
    print("TEST 4: TRAJECTORY Start Uses Event Timestamp")
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
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 1
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process TRAJECTORY start with specific timestamp
        trajectory_start_event = {
            "timestamp": "2025-11-06 16:15:30",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "start",
            "message": "Starting trajectory generation",
            "data": {
                "config": {
                    "num_trajectories": 100
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_start_event))
        
        # Verify created_at matches event timestamp
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).all()
        
        assert len(traj_steps) == 1
        traj_step = traj_steps[0]
        
        print(f"\n📊 Created at: {traj_step.created_at}")
        print(f"📊 Expected: 2025-11-06 16:15:30")
        
        assert traj_step.created_at.year == 2025
        assert traj_step.created_at.month == 11
        assert traj_step.created_at.day == 6
        assert traj_step.created_at.hour == 16
        assert traj_step.created_at.minute == 15
        assert traj_step.created_at.second == 30
        
        print("\n✅ PASSED - created_at uses event timestamp")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_trajectory_config_stored():
    """Test 5: TRAJECTORY config is stored in step_config"""
    print("\n" + "=" * 70)
    print("TEST 5: TRAJECTORY Config Stored in step_config")
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
            "data": {
                "config": {
                    "current_iteration": 1,
                    "total_iterations": 1
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(iteration_start_event))
        
        # Process TRAJECTORY start with detailed config
        trajectory_start_event = {
            "timestamp": "2025-11-06 17:05:00",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "start",
            "message": "Starting trajectory generation",
            "data": {
                "config": {
                    "num_trajectories": 200,
                    "max_length": 1024,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 50,
                    "model_name": "gpt-3.5-turbo",
                    "custom_param": "test_value"
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_start_event))
        
        # Verify step_config is stored correctly
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).all()
        
        assert len(traj_steps) == 1
        traj_step = traj_steps[0]
        
        print(f"\n📊 Step config: {traj_step.step_config}")
        
        assert traj_step.step_config is not None
        assert traj_step.step_config.get("num_trajectories") == 200
        assert traj_step.step_config.get("max_length") == 1024
        assert traj_step.step_config.get("temperature") == 0.9
        assert traj_step.step_config.get("top_p") == 0.95
        assert traj_step.step_config.get("top_k") == 50
        assert traj_step.step_config.get("model_name") == "gpt-3.5-turbo"
        assert traj_step.step_config.get("custom_param") == "test_value"
        
        print("\n✅ PASSED - step_config stored correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_6_trajectory_without_iteration_context():
    """Test 6: TRAJECTORY without iteration context is handled gracefully"""
    print("\n" + "=" * 70)
    print("TEST 6: TRAJECTORY Without Iteration Context")
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
        
        # Process TRAJECTORY start WITHOUT iteration context
        trajectory_start_event = {
            "timestamp": "2025-11-06 18:00:00",
            "level": "INFO",
            "phase": "TRAJECTORY",
            "event": "start",
            "message": "Starting trajectory generation",
            "data": {
                "config": {
                    "num_trajectories": 100
                }
            }
        }
        
        await monitor._process_log_line(json.dumps(trajectory_start_event))
        
        # Verify no traj_gen step was created (no iteration context)
        traj_steps = await TrainingIteration.filter(
            training_job=job,
            step_type=StepType.TRAJ_GEN
        ).all()
        
        print(f"\n📊 Number of traj_gen steps created: {len(traj_steps)}")
        print(f"📊 Expected: 0 (no iteration context)")
        
        assert len(traj_steps) == 0
        
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
    print("#  TEST 12: TRAJECTORY EVENT PROCESSING".center(70))
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
    results["trajectory_start_creates_step"] = await test_1_trajectory_start_creates_traj_gen_step()
    results["trajectory_end_records_duration"] = await test_2_trajectory_end_records_duration()
    results["trajectory_links_to_iteration"] = await test_3_trajectory_links_to_correct_iteration()
    results["trajectory_timestamp"] = await test_4_trajectory_timestamp_accuracy()
    results["trajectory_config_stored"] = await test_5_trajectory_config_stored()
    results["trajectory_without_context"] = await test_6_trajectory_without_iteration_context()
    
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
    db_path = os.path.join(project_root, "test_event_trajectory.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("\n🧹 Cleaned up test database")


if __name__ == "__main__":
    asyncio.run(run_all_tests())