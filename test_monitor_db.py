#!/usr/bin/env python3
"""Test script for TrainingJobMonitor to verify database updates"""

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import Mock
from pathlib import Path

from tortoise import Tortoise
from app.services.training_job_monitor import TrainingJobMonitor
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval


class MockSshExecutor:
    """Mock SSH executor that reads from local global.json file"""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def execute_command(self, command: str, check=True):
        """Mock execute_command to return the log file path"""
        result = Mock()
        result.success = True
        result.stdout = self.log_file_path
        return result

    def download_file(self, remote_path: str, local_path: str):
        """Mock download_file to copy from local global.json"""
        with open(self.log_file_path, "r") as src:
            content = src.read()
        with open(local_path, "w") as dst:
            dst.write(content)


async def setup_test_db():
    """Setup test database"""
    await Tortoise.init(
        db_url="sqlite://:memory:",
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


async def cleanup_test_db():
    """Cleanup test database"""
    await Tortoise.close_connections()


async def create_test_job() -> str:
    """Create a test training job"""
    job = await TrainingJob.create(
        uuid=uuid.uuid4(),
        project_id="classify_spam_local",
        training_run_id="test_run_001",
        status=TrainingJobStatus.PENDING,
        training_request={"test": "data"},
    )
    return str(job.uuid)


async def print_db_state(job_uuid: str):
    """Print the current state of the database"""
    print("\n" + "=" * 80)
    print("DATABASE STATE AFTER PROCESSING GLOBAL.JSON")
    print("=" * 80)

    # Get the job
    job = await TrainingJob.get(uuid=job_uuid)
    print(f"\n📋 TRAINING JOB:")
    print(f"   UUID: {job.uuid}")
    print(f"   Project ID: {job.project_id}")
    print(f"   Training Run ID: {job.training_run_id}")
    print(f"   Status: {job.status.value}")
    print(f"   Created At: {job.created_at}")
    print(f"   Completed At: {job.completed_at}")
    if job.time_taken:
        print(
            f"   Time Taken: {job.time_taken} seconds ({job.time_taken / 60:.2f} minutes)"
        )
    else:
        print("   Time Taken: None")

    if job.training_config:
        print(f"\n   Training Config:")
        for key, value in job.training_config.items():
            print(f"      {key}: {value}")

    # Get iterations
    iterations = await TrainingIteration.filter(training_job__uuid=job_uuid).order_by(
        "iteration_number", "created_at"
    )
    print(f"\n🔄 TRAINING ITERATIONS ({len(iterations)} total):")

    iteration_steps = {}
    for iteration in iterations:
        iter_num = iteration.iteration_number
        if iter_num not in iteration_steps:
            iteration_steps[iter_num] = []
        iteration_steps[iter_num].append(iteration)

    for iter_num in sorted(iteration_steps.keys()):
        print(f"\n   ━━━ Iteration {iter_num} ━━━")
        for iteration in iteration_steps[iter_num]:
            if iteration.step_type == StepType.ITERATION:
                print(f"       Type: {iteration.step_type.value} (Main)")
                print(f"       UUID: {iteration.uuid}")
                print(f"       Created: {iteration.created_at}")
                print(f"       Completed: {iteration.completed_at}")
                if iteration.time_taken:
                    print(f"       Time Taken: {iteration.time_taken} minutes")
                if iteration.step_config:
                    print(
                        f"       Step Config: {json.dumps(iteration.step_config, indent=10)[:200]}..."
                    )
            else:
                print(f"\n       ├── Step: {iteration.step_type.value}")
                print(f"           UUID: {iteration.uuid}")
                print(f"           Created: {iteration.created_at}")
                print(f"           Completed: {iteration.completed_at}")
                if iteration.step_time:
                    print(f"           Step Time: {iteration.step_time} minutes")
                if iteration.step_config:
                    print(
                        f"           Step Config: {json.dumps(iteration.step_config, indent=14)[:200]}..."
                    )

    # Get epochs
    epochs = await EpochTrain.all().order_by("iteration_number", "epoch_number")
    if epochs:
        print(f"\n📊 EPOCH TRAINING ({len(epochs)} total):")
        for epoch in epochs:
            print(f"   Iteration {epoch.iteration_number}, Epoch {epoch.epoch_number}:")
            print(f"       UUID: {epoch.uuid}")
            print(f"       Metrics: {epoch.metrics}")

    # Get evaluations
    evals = await Eval.all()
    if evals:
        print(f"\n📈 EVALUATIONS ({len(evals)} total):")
        for eval_record in evals:
            print(f"   Model: {eval_record.model_id}")
            print(f"       UUID: {eval_record.uuid}")
            print(f"       Dataset: {eval_record.dataset}")
            print(f"       Metrics:")
            for key, value in eval_record.metrics.items():
                print(f"           {key}: {value}")
            print(f"       Eval Data Path: {eval_record.eval_data_path}")

    print("\n" + "=" * 80 + "\n")


async def test_monitor():
    """Test the TrainingJobMonitor with the actual global.json file"""
    print("\n🚀 Starting TrainingJobMonitor Test\n")

    # Setup
    await setup_test_db()
    job_uuid = await create_test_job()
    print(f"✅ Created test job: {job_uuid}")

    # Path to the global.json file
    log_file_path = "models/global.json"
    if not Path(log_file_path).exists():
        print(f"❌ Error: {log_file_path} not found!")
        await cleanup_test_db()
        return

    print(f"✅ Found log file: {log_file_path}")

    # Count lines in log file
    with open(log_file_path, "r") as f:
        lines = [line for line in f if line.strip()]
        print(f"📝 Log file contains {len(lines)} events")

    # Create mock SSH executor
    mock_ssh = MockSshExecutor(log_file_path)

    # Create monitor
    monitor = TrainingJobMonitor(
        training_job_uuid=job_uuid,
        ssh_executor=mock_ssh,
        remote_log_path=log_file_path,
        poll_interval=1,  # Fast polling for testing
    )

    print(f"✅ Created TrainingJobMonitor\n")
    print("📝 Processing log file...\n")

    # Process the log file once (simulate one poll cycle)
    await monitor._poll_and_update()

    print("\n✅ Finished processing log file")

    # Print database state
    await print_db_state(job_uuid)

    # Summary
    job = await TrainingJob.get(uuid=job_uuid)
    iterations_count = await TrainingIteration.filter(
        training_job__uuid=job_uuid
    ).count()
    epochs_count = await EpochTrain.all().count()
    evals_count = await Eval.all().count()

    print("📊 SUMMARY:")
    print(f"   Job Status: {job.status.value}")
    print(f"   Total Iterations/Steps: {iterations_count}")
    print(f"   Total Epochs: {epochs_count}")
    print(f"   Total Evaluations: {evals_count}")
    print(
        f"   Job Duration: {job.time_taken} seconds"
        if job.time_taken
        else "   Job Duration: Not set"
    )

    # Cleanup
    await cleanup_test_db()
    print("\n✅ Test completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(test_monitor())
