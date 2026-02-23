# test/scripts/test_training_job_monitor_parsing.py
"""
Test script to verify TrainingJobMonitor can correctly parse global.json log file.
Tests the actual _process_log_line method and event handlers from TrainingJobMonitor.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.training_job_monitor import TrainingJobMonitor
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval
from app.database import init_db, async_session_maker
from sqlalchemy import select


class TestTrainingJobMonitorParsing:
    """Test harness for TrainingJobMonitor parsing functionality."""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.test_job_uuid = str(uuid.uuid4())
        self.results = {
            "total_lines": 0,
            "parsed_lines": 0,
            "failed_lines": 0,
            "events_by_phase": {},
            "parsing_errors": [],
            "db_records_created": {
                "training_iterations": 0,
                "epoch_trains": 0,
                "evals": 0,
            },
        }

    async def setup_test_database(self):
        """Initialize test database and create test training job."""
        print("\n🔧 Setting up test database...")

        # Initialize database
        await init_db()

        # Create test training job
        async with async_session_maker() as session:
            test_job = TrainingJob(
                uuid=self.test_job_uuid,
                project_id="test_project",
                training_run_id="test_run_1",
                status=TrainingJobStatus.RUNNING,
                created_at=datetime.now().isoformat(),
            )
            session.add(test_job)
            await session.commit()

        print(f"✅ Created test training job: {self.test_job_uuid}")

    async def test_parsing(self):
        """Test parsing of global.json file using actual TrainingJobMonitor."""
        print(f"\n📁 Testing log file: {self.log_file_path}")

        # Check if file exists
        if not Path(self.log_file_path).exists():
            print(f"\n❌ FAILED: File not found: {self.log_file_path}")
            return False

        # Create a mock SSH executor (not needed for parsing test)
        class MockSshExecutor:
            def execute_command(self, cmd, check=True):
                class Result:
                    success = True
                    stdout = ""

                return Result()

        # Create TrainingJobMonitor instance
        monitor = TrainingJobMonitor(
            training_job_uuid=self.test_job_uuid,
            ssh_executor=MockSshExecutor(),
            poll_interval=1,
        )

        # Load the training job
        async with async_session_maker() as session:
            monitor.training_job = await monitor._load_training_job(session)

        print("\n🔍 Processing log file line by line...")

        # Read and process each line
        with open(self.log_file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                self.results["total_lines"] += 1

                # Skip empty lines
                if not line.strip():
                    continue

                try:
                    # Parse JSON
                    event = json.loads(line)
                    phase = event.get("phase", "UNKNOWN")
                    event_type = event.get("event", "UNKNOWN")

                    # Track event
                    if phase not in self.results["events_by_phase"]:
                        self.results["events_by_phase"][phase] = []
                    self.results["events_by_phase"][phase].append(event_type)

                    # Process using actual TrainingJobMonitor method
                    await monitor._process_log_line(line)

                    self.results["parsed_lines"] += 1
                    print(f"  ✓ Line {line_num}: {phase}:{event_type}")

                except json.JSONDecodeError as e:
                    self.results["failed_lines"] += 1
                    self.results["parsing_errors"].append(
                        {
                            "line": line_num,
                            "error": f"JSON decode error: {str(e)}",
                            "content": line[:100],
                        }
                    )
                    print(f"  ✗ Line {line_num}: JSON decode error")

                except Exception as e:
                    self.results["failed_lines"] += 1
                    self.results["parsing_errors"].append(
                        {
                            "line": line_num,
                            "error": f"Processing error: {str(e)}",
                            "content": line[:100],
                        }
                    )
                    print(f"  ✗ Line {line_num}: {str(e)}")

        # Flush any remaining batched updates
        await monitor.update_batcher.flush()

        return True

    async def verify_database_records(self):
        """Verify that database records were created correctly."""
        print("\n🔍 Verifying database records...")

        async with async_session_maker() as session:
            # Count TrainingIteration records
            stmt = select(TrainingIteration).where(
                TrainingIteration.training_job_uuid == self.test_job_uuid
            )
            result = await session.execute(stmt)
            iterations = result.scalars().all()
            self.results["db_records_created"]["training_iterations"] = len(iterations)

            print(f"\n📊 TrainingIteration records: {len(iterations)}")
            for iteration in iterations:
                print(
                    f"  - {iteration.step_type.value}: iteration {iteration.iteration_number}"
                )

            # Count EpochTrain records
            stmt = select(EpochTrain)
            result = await session.execute(stmt)
            epochs = result.scalars().all()
            self.results["db_records_created"]["epoch_trains"] = len(epochs)

            print(f"\n📊 EpochTrain records: {len(epochs)}")
            for epoch in epochs:
                print(
                    f"  - Iteration {epoch.iteration_number}, Epoch {epoch.epoch_number}"
                )
                if epoch.metrics:
                    print(f"    Metrics: {epoch.metrics}")

            # Count Eval records
            stmt = select(Eval)
            result = await session.execute(stmt)
            evals = result.scalars().all()
            self.results["db_records_created"]["evals"] = len(evals)

            print(f"\n📊 Eval records: {len(evals)}")
            for eval_record in evals:
                print(f"  - {eval_record.model_id} on {eval_record.dataset}")
                if eval_record.metrics:
                    print(f"    Metrics: {eval_record.metrics}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)

        print(f"\n📝 Parsing Results:")
        print(f"  Total lines: {self.results['total_lines']}")
        print(f"  Successfully parsed: {self.results['parsed_lines']}")
        print(f"  Failed to parse: {self.results['failed_lines']}")

        print(f"\n📊 Events by Phase:")
        for phase, events in sorted(self.results["events_by_phase"].items()):
            print(f"  {phase}: {len(events)} events")

        print(f"\n💾 Database Records Created:")
        print(
            f"  TrainingIterations: {self.results['db_records_created']['training_iterations']}"
        )
        print(f"  EpochTrains: {self.results['db_records_created']['epoch_trains']}")
        print(f"  Evals: {self.results['db_records_created']['evals']}")

        if self.results["parsing_errors"]:
            print(f"\n❌ Parsing Errors ({len(self.results['parsing_errors'])}):")
            for error in self.results["parsing_errors"][:5]:  # Show first 5
                print(f"  Line {error['line']}: {error['error']}")
            if len(self.results["parsing_errors"]) > 5:
                print(
                    f"  ... and {len(self.results['parsing_errors']) - 5} more errors"
                )

        # Determine success
        success = (
            self.results["failed_lines"] == 0
            and self.results["parsed_lines"] > 0
            and self.results["db_records_created"]["training_iterations"] > 0
        )

        print("\n" + "=" * 70)
        if success:
            print("✅ PASSED: TrainingJobMonitor successfully parsed global.json!")
            print(f"   Processed {self.results['parsed_lines']} events")
            print(
                f"   Created {self.results['db_records_created']['training_iterations']} training iteration records"
            )
        else:
            print("❌ FAILED: Issues detected during parsing")
            if self.results["failed_lines"] > 0:
                print(f"   Failed lines: {self.results['failed_lines']}")
            if self.results["db_records_created"]["training_iterations"] == 0:
                print("   No database records created")
        print("=" * 70)

        return success


async def main():
    """Main test function."""
    # Default log file path
    default_log_path = "data/global.json"

    # Allow custom path from command line
    log_path = sys.argv[1] if len(sys.argv) > 1 else default_log_path

    print("=" * 70)
    print("  TrainingJobMonitor Parsing Test")
    print("=" * 70)

    # Create test instance
    test = TestTrainingJobMonitorParsing(log_path)

    try:
        # Setup test database
        await test.setup_test_database()

        # Run parsing test
        await test.test_parsing()

        # Verify database records
        await test.verify_database_records()

        # Print summary
        success = test.print_summary()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
