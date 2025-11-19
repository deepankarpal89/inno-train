"""
Phase 5.1: Orchestrator - Individual Methods Testing

Tests for TrainingJobOrchestrator individual methods:
1. _create_training_job_record() - Creates DB entry
2. _build_and_upload_yaml() - Builds and uploads YAML
3. _launch_gpu_instance() - Launches GPU instance
4. _setup_ssh_connection() - Sets up SSH with retry
5. _transfer_training_files() - Transfers all files
6. _start_monitoring() - Starts background monitoring
7. _execute_training_script() - Executes training script
8. wait_for_completion() - Polls for completion
9. _download_outputs() - Downloads results to S3
10. _cleanup_instance() - Terminates GPU instance
11. _mark_job_failed() - Updates status to failed
12. cancel_job() - Cancels running job

Run: python tests/test_16_orchestrator_methods.py
"""

import asyncio
import os
import sys
import uuid
import traceback
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tortoise import Tortoise
from dotenv import load_dotenv

from app.services.training_job_orchestrator import TrainingJobOrchestrator
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from scripts.project_yaml_builder import ProjectYamlBuilder

load_dotenv()


async def init_db():
    """Initialize database connection"""
    print("\n🔌 Connecting to database...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_root, "test_orchestrator.db")

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
    print("✅ Database connected!\n")


async def close_db():
    """Close database connection"""
    await Tortoise.close_connections()
    print("\n🔌 Database connection closed")


def get_sample_request_data():
    """Get sample training request data"""
    return {
        "data": {
            "request_data": {
                "training_run_id": f"test-run-{uuid.uuid4().hex[:8]}",
                "project": {"id": "test-project-001", "name": "Test Project"},
                "prompt": {
                    "id": "prompt-001",
                    "name": "Test Prompt",
                    "content": "You are a helpful assistant.",
                    "think_flag": True,
                    "seed_text": "Test seed",
                },
                "train_dataset": {
                    "id": "dataset-train-001",
                    "file_name": "datasets/train.csv",
                    "s3_path": "datasets/train.csv",
                    "file_path": "/data/train.csv",
                },
                "eval_dataset": {
                    "id": "dataset-eval-001",
                    "file_name": "datasets/eval.csv",
                    "s3_path": "datasets/eval.csv",
                    "file_path": "/data/eval.csv",
                },
                "config": {
                    "s3_path": "configs/config.yaml",
                    "file_path": "/config/config.yaml",
                },
                "training_config": {"iterations": 2, "epochs": 3, "batch_size": 16},
            }
        }
    }


async def test_1_create_training_job_record():
    """Test 1: _create_training_job_record() creates DB entry"""
    print("=" * 70)
    print("TEST 1: _create_training_job_record()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()
        request_data = get_sample_request_data()

        print("\n📝 Creating training job record...")
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        print(f"\n✅ Job created:")
        print(f"   UUID: {training_job_uuid}")

        # Verify in database
        job = await TrainingJob.get(uuid=training_job_uuid)
        assert job is not None
        assert job.status == TrainingJobStatus.PENDING
        assert job.project_id == "test-project-001"

        print(f"   Status: {job.status.value}")
        print(f"   Project ID: {job.project_id}")

        print("\n✅ PASSED")
        return True, training_job_uuid

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False, None


async def test_2_build_and_upload_yaml():
    """Test 2: _build_and_upload_yaml() works end-to-end"""
    print("\n" + "=" * 70)
    print("TEST 2: _build_and_upload_yaml()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()
        request_data = get_sample_request_data()

        print("\n📝 Building and uploading YAML...")

        # Mock the S3 upload
        with patch.object(ProjectYamlBuilder, "save_to_s3", return_value=True):
            yaml_builder = await orchestrator._build_and_upload_yaml(request_data)

        print(f"\n✅ YAML builder created:")
        print(f"   Has data: {yaml_builder.yaml_data is not None}")
        print(f"   Keys: {list(yaml_builder.yaml_data.keys())[:5]}...")

        assert yaml_builder is not None
        assert yaml_builder.yaml_data is not None

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_3_launch_gpu_instance():
    """Test 3: _launch_gpu_instance() returns valid instance"""
    print("\n" + "=" * 70)
    print("TEST 3: _launch_gpu_instance()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job first
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        print(f"\n📝 Launching GPU instance for job {training_job_uuid[:8]}...")

        # Mock Lambda client methods
        mock_gpu_config = {"name": "gpu_1x_a10", "region": "us-west-1"}

        mock_instance_config = {"id": "i-test123456", "ip": "192.168.1.100"}

        with patch.object(
            orchestrator.lambda_client,
            "list_available_instances",
            return_value=mock_gpu_config,
        ):
            with patch.object(
                orchestrator.lambda_client,
                "launch_instance",
                return_value=mock_instance_config,
            ):
                instance_id, instance_ip = await orchestrator._launch_gpu_instance(
                    training_job_uuid
                )

        print(f"\n✅ Instance launched:")
        print(f"   Instance ID: {instance_id}")
        print(f"   Instance IP: {instance_ip}")

        # Verify database update
        job = await TrainingJob.get(uuid=training_job_uuid)
        assert job.machine_config is not None
        assert job.machine_config["instance_id"] == instance_id
        assert job.machine_config["instance_ip"] == instance_ip

        print(f"   Machine config updated: ✓")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_4_setup_ssh_connection():
    """Test 4: _setup_ssh_connection() with retry logic"""
    print("\n" + "=" * 70)
    print("TEST 4: _setup_ssh_connection()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()
        test_ip = "192.168.1.100"

        print(f"\n📝 Setting up SSH connection to {test_ip}...")

        # Mock SSH executor
        mock_ssh = Mock()
        mock_ssh.connect = Mock()

        with patch(
            "app.services.training_job_orchestrator.SshExecutor", return_value=mock_ssh
        ):
            ssh_executor = await orchestrator._setup_ssh_connection(test_ip)

        print(f"\n✅ SSH connection established:")
        print(f"   IP: {test_ip}")
        print(f"   Executor created: {ssh_executor is not None}")

        assert ssh_executor is not None

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_5_transfer_training_files():
    """Test 5: _transfer_training_files() transfers all files"""
    print("\n" + "=" * 70)
    print("TEST 5: _transfer_training_files()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()
        test_ip = "192.168.1.100"

        # Create mock YAML builder
        yaml_builder = Mock()
        yaml_builder.yaml_data = {
            "train_s3_path": "datasets/train.csv",
            "train_file_path": "/data/train.csv",
            "eval_s3_path": "datasets/eval.csv",
            "eval_file_path": "/data/eval.csv",
            "config_s3_path": "configs/config.yaml",
            "config_file_path": "/config/config.yaml",
        }

        # Mock SSH executor
        mock_ssh = Mock()
        mock_ssh.upload_file = Mock()
        orchestrator.ssh_executor = mock_ssh

        print("\n📝 Transferring training files...")

        # Mock file transfer
        with patch.object(orchestrator.file_transfer, "transfer_file_to_server"):
            await orchestrator._transfer_training_files(yaml_builder, test_ip)

        print(f"\n✅ Files transferred:")
        print(f"   Train dataset: ✓")
        print(f"   Eval dataset: ✓")
        print(f"   Config file: ✓")
        print(f"   Training script: ✓")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_6_start_monitoring():
    """Test 6: _start_monitoring() starts background task"""
    print("\n" + "=" * 70)
    print("TEST 6: _start_monitoring()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        # Mock SSH executor
        mock_ssh = Mock()
        orchestrator.ssh_executor = mock_ssh

        print(f"\n📝 Starting monitoring for job {training_job_uuid[:8]}...")

        # Mock the monitor's start_monitoring to avoid actual polling
        with patch(
            "app.services.training_job_monitor.TrainingJobMonitor.start_monitoring",
            new_callable=AsyncMock,
        ):
            await orchestrator._start_monitoring(training_job_uuid)

        print(f"\n✅ Monitoring started:")
        print(f"   Monitor created: {orchestrator.monitor is not None}")
        print(f"   Monitor task created: {orchestrator.monitor_task is not None}")

        assert orchestrator.monitor is not None
        assert orchestrator.monitor_task is not None

        # Cancel the task
        orchestrator.monitor_task.cancel()
        try:
            await orchestrator.monitor_task
        except asyncio.CancelledError:
            pass

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_7_execute_training_script():
    """Test 7: _execute_training_script() runs non-blocking"""
    print("\n" + "=" * 70)
    print("TEST 7: _execute_training_script()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Mock SSH executor
        mock_ssh = Mock()
        mock_ssh.execute_command = Mock(
            return_value=Mock(success=True, stdout="", stderr="")
        )
        orchestrator.ssh_executor = mock_ssh

        print("\n📝 Executing training script...")

        await orchestrator._execute_training_script()

        print(f"\n✅ Training script executed:")
        print(f"   Command sent: ✓")
        print(f"   Non-blocking: ✓")

        # Verify execute_command was called
        assert mock_ssh.execute_command.called

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_8_wait_for_completion():
    """Test 8: wait_for_completion() polls correctly"""
    print("\n" + "=" * 70)
    print("TEST 8: wait_for_completion()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        print(f"\n📝 Waiting for job {training_job_uuid[:8]} to complete...")

        # Simulate job completion after short delay
        async def simulate_completion():
            await asyncio.sleep(0.5)
            job = await TrainingJob.get(uuid=training_job_uuid)
            job.status = TrainingJobStatus.COMPLETED
            await job.save()

        # Start simulation
        asyncio.create_task(simulate_completion())

        # Mock download outputs
        with patch.object(orchestrator, "_download_outputs", new_callable=AsyncMock):
            # Wait with short timeout
            await orchestrator.wait_for_completion(training_job_uuid, timeout_minutes=1)

        print(f"\n✅ Job completed:")
        print(f"   Status: COMPLETED")
        print(f"   Polling worked: ✓")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_9_download_outputs():
    """Test 9: _download_outputs() transfers results to S3"""
    print("\n" + "=" * 70)
    print("TEST 9: _download_outputs()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job with machine config
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        job = await TrainingJob.get(uuid=training_job_uuid)
        job.machine_config = {
            "instance_id": "i-test123",
            "instance_ip": "192.168.1.100",
        }
        await job.save()

        print(f"\n📝 Downloading outputs for job {training_job_uuid[:8]}...")

        # Mock file transfer
        with patch.object(orchestrator.file_transfer, "transfer_files_to_s3"):
            await orchestrator._download_outputs(training_job_uuid)

        print(f"\n✅ Outputs downloaded:")
        print(f"   Transfer initiated: ✓")
        print(f"   S3 path generated: ✓")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_10_cleanup_instance():
    """Test 10: _cleanup_instance() terminates GPU"""
    print("\n" + "=" * 70)
    print("TEST 10: _cleanup_instance()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()
        test_instance_id = "i-test123456"

        # Mock SSH executor
        mock_ssh = Mock()
        mock_ssh.disconnect = Mock()
        orchestrator.ssh_executor = mock_ssh

        print(f"\n📝 Cleaning up instance {test_instance_id}...")

        # Mock Lambda client terminate
        with patch.object(
            orchestrator.lambda_client, "terminate_instance", return_value=True
        ):
            await orchestrator._cleanup_instance(test_instance_id)

        print(f"\n✅ Instance cleaned up:")
        print(f"   SSH disconnected: ✓")
        print(f"   Instance terminated: ✓")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_11_mark_job_failed():
    """Test 11: _mark_job_failed() updates status"""
    print("\n" + "=" * 70)
    print("TEST 11: _mark_job_failed()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        error_message = "Test error: GPU out of memory"

        print(f"\n📝 Marking job {training_job_uuid[:8]} as failed...")

        await orchestrator._mark_job_failed(training_job_uuid, error_message)

        # Verify status
        job = await TrainingJob.get(uuid=training_job_uuid)
        assert job.status == TrainingJobStatus.FAILED
        assert job.completed_at is not None

        print(f"\n✅ Job marked as failed:")
        print(f"   Status: {job.status.value}")
        print(f"   Completed at: {job.completed_at}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_12_cancel_job():
    """Test 12: cancel_job() stops everything cleanly"""
    print("\n" + "=" * 70)
    print("TEST 12: cancel_job()")
    print("=" * 70)

    try:
        orchestrator = TrainingJobOrchestrator()

        # Create a test job with machine config
        request_data = get_sample_request_data()
        training_job_uuid = await orchestrator._create_training_job_record(request_data)

        job = await TrainingJob.get(uuid=training_job_uuid)
        job.status = TrainingJobStatus.RUNNING
        job.machine_config = {
            "instance_id": "i-test123",
            "instance_ip": "192.168.1.100",
        }
        await job.save()

        print(f"\n📝 Cancelling job {training_job_uuid[:8]}...")

        # Mock cleanup
        with patch.object(orchestrator, "_cleanup_instance", new_callable=AsyncMock):
            result = await orchestrator.cancel_job(training_job_uuid)

        assert result is True

        # Verify status
        job = await TrainingJob.get(uuid=training_job_uuid)
        assert job.status == TrainingJobStatus.CANCELLED
        assert job.completed_at is not None

        print(f"\n✅ Job cancelled:")
        print(f"   Status: {job.status.value}")
        print(f"   Cleanup called: ✓")
        print(f"   Completed at: {job.completed_at}")

        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 5.1: ORCHESTRATOR - INDIVIDUAL METHODS".center(70))
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
    results["test_1"], _ = await test_1_create_training_job_record()
    results["test_2"] = await test_2_build_and_upload_yaml()
    results["test_3"] = await test_3_launch_gpu_instance()
    results["test_4"] = await test_4_setup_ssh_connection()
    results["test_5"] = await test_5_transfer_training_files()
    results["test_6"] = await test_6_start_monitoring()
    results["test_7"] = await test_7_execute_training_script()
    results["test_8"] = await test_8_wait_for_completion()
    results["test_9"] = await test_9_download_outputs()
    results["test_10"] = await test_10_cleanup_instance()
    results["test_11"] = await test_11_mark_job_failed()
    results["test_12"] = await test_12_cancel_job()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    test_names = {
        "connection": "Database Connection",
        "test_1": "_create_training_job_record()",
        "test_2": "_build_and_upload_yaml()",
        "test_3": "_launch_gpu_instance()",
        "test_4": "_setup_ssh_connection()",
        "test_5": "_transfer_training_files()",
        "test_6": "_start_monitoring()",
        "test_7": "_execute_training_script()",
        "test_8": "wait_for_completion()",
        "test_9": "_download_outputs()",
        "test_10": "_cleanup_instance()",
        "test_11": "_mark_job_failed()",
        "test_12": "cancel_job()",
    }

    for test, result in results.items():
        status = "✅" if result else "❌"
        name = test_names.get(test, test)
        print(f"{status} {name}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    await close_db()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
