#!/usr/bin/env python3
"""
Test script for TrainingJobMonitor without SSH connection.
Reads from local global.json file and processes events into database.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tortoise import Tortoise
from scripts.ssh_executor import CommandResult
from app.services.training_job_monitor import TrainingJobMonitor
from models.training_job import TrainingJob, TrainingJobStatus
from scripts.utils import ist_now, parse_timestamp


class MockSshExecutor:
    """
    Mock SSH executor that reads from local global.json file instead of SSH.
    """

    def __init__(self, local_json_path: str):
        self.local_json_path = local_json_path
        self.ip = "localhost"
        self.username = "test"
        self.timeout = 120
        self.client = None

    def connect(self):
        """Mock connect - does nothing."""
        pass

    def disconnect(self):
        """Mock disconnect - does nothing."""
        pass

    def execute_command(self, command: str, check: bool = True) -> CommandResult:
        """
        Mock execute_command that simulates finding the global.json file.
        """
        if "find . -path" in command and "global.json" in command:
            # Return the local path as if found on remote server
            return CommandResult(
                command=command,
                stdout=self.local_json_path,
                stderr="",
                return_code=0,
                success=True,
                duration=0.1,
            )
        else:
            # For other commands, return empty success
            return CommandResult(
                command=command,
                stdout="",
                stderr="",
                return_code=0,
                success=True,
                duration=0.1,
            )

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Mock download_file that copies from local global.json to temp file.
        """
        try:
            # Copy content from our local global.json to the temp file
            with open(self.local_json_path, "r") as src:
                content = src.read()

            with open(local_path, "w") as dst:
                dst.write(content)

            return True
        except Exception as e:
            raise Exception(
                f"Failed to copy file {self.local_json_path} to {local_path}: {str(e)}"
            )


async def setup_database():
    """
    Initialize database for testing.
    """
    # Use in-memory SQLite for testing
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

    # Generate schema
    await Tortoise.generate_schemas()
    print("✅ Database initialized")


async def create_test_job() -> str:
    """
    Create a test training job in database.
    """
    job_uuid = str(uuid.uuid4())

    job = await TrainingJob.create(
        uuid=job_uuid,
        project_id="spam_local_test",
        training_run_id="test_run_1",
        status=TrainingJobStatus.PENDING,
        project_config={
            "project_name": "spam local",
            "no_iterations": 1,
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        },
    )

    print(f"✅ Created test job: {job_uuid}")
    return job_uuid


async def test_monitor_with_local_file(global_json_path: str):
    """
    Test the TrainingJobMonitor with local global.json file.
    """
    print("=" * 70)
    print("🧪 TESTING TRAINING JOB MONITOR")
    print("=" * 70)

    try:
        # Setup database
        await setup_database()

        # Create test job
        job_uuid = await create_test_job()

        # Create mock SSH executor
        mock_ssh = MockSshExecutor(global_json_path)

        # Setup logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("TestMonitor")

        # Completion callback
        completion_called = False

        def on_completion():
            nonlocal completion_called
            completion_called = True
            print("🎉 Completion callback triggered!")

        # Create monitor with fast polling for testing
        monitor = TrainingJobMonitor(
            training_job_uuid=job_uuid,
            ssh_executor=mock_ssh,
            remote_log_path=global_json_path,  # Use local path directly
            poll_interval=1,  # Fast polling for testing
            logger=logger,
            on_completion_callback=on_completion,
        )

        print(f"📊 Starting monitor for job: {job_uuid}")
        print(f"📁 Reading from: {global_json_path}")

        # Start monitoring (will process all events and complete)
        await monitor.start_monitoring()

        print("\n" + "=" * 70)
        print("📈 FINAL RESULTS")
        print("=" * 70)

        # Check final job status
        final_job = await TrainingJob.get(uuid=job_uuid)
        print(f"📋 Job Status: {final_job.status}")
        print(f"⏱️  Time Taken: {final_job.time_taken} seconds")
        print(f"📅 Created At: {final_job.created_at}")
        print(f"✅ Completed At: {final_job.completed_at}")
        print(f"🔧 Project Config: {final_job.project_config}")

        # Check iterations
        from models.training_iteration import TrainingIteration

        iterations = await TrainingIteration.filter(training_job__uuid=job_uuid).all()
        print(f"\n🔄 Total Iterations Created: {len(iterations)}")

        for iteration in iterations:
            print(
                f"  - {iteration.step_type}: Iteration {iteration.iteration_number} ({iteration.time_taken}s)"
            )

        # Check epochs
        from models.epoch_train import EpochTrain

        epochs = await EpochTrain.filter(iteration__training_job__uuid=job_uuid).all()
        print(f"\n📊 Total Epochs: {len(epochs)}")

        for epoch in epochs:
            print(f"  - Epoch {epoch.epoch_number}: {epoch.metrics}")

        # Check evaluations
        from models.eval import Eval

        evals = await Eval.filter(iteration__training_job__uuid=job_uuid).all()
        print(f"\n📈 Total Evaluations: {len(evals)}")

        for eval_record in evals:
            print(f"  - {eval_record.model_id}: {eval_record.metrics}")

        # Check completion callback
        print(f"\n🎯 Completion Callback Called: {completion_called}")

        # Full database dump for verification
        print("\n" + "=" * 70)
        print("📦 FULL DATABASE DUMP")
        print("=" * 70)

        jobs = await TrainingJob.all().values()
        print(f"🧰 TrainingJob count: {len(jobs)}")
        for job in jobs:
            print(job)

            # Iterations for this job (primary key is 'uuid')
            iters = await TrainingIteration.filter(
                training_job__uuid=job["uuid"]
            ).values()
            print(f"  iterations: {len(iters)}")
            for it in iters:
                print(f"  ITER ROW: {it}")

                # Epochs/Evals FK column is 'iteration', so .values() exposes 'iteration_id'
                epochs = await EpochTrain.filter(iteration_id=it["uuid"]).values()
                print(f"    epochs: {len(epochs)}")
                for ep in epochs:
                    print(f"    EPOCH ROW: {ep}")

                evals = await Eval.filter(iteration_id=it["uuid"]).values()
                print(f"    evals: {len(evals)}")
                for ev in evals:
                    print(f"    EVAL ROW: {ev}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Close database connections
        await Tortoise.close_connections()


async def main():
    """
    Main test function.
    """
    # Get global.json path
    global_json_path = (
        "/Users/deepankarpal/Projects/innotone/innotone-training/global.json"
    )

    if not os.path.exists(global_json_path):
        print(f"❌ global.json file not found at: {global_json_path}")
        print("Please provide the correct path to global.json file")
        return

    print(f"📁 Using global.json from: {global_json_path}")

    # Run the test
    await test_monitor_with_local_file(global_json_path)


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())
