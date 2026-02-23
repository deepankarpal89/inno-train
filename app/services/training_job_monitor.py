"""
TrainingJobMonitor - Monitors global.json file on GPU server and updates database
"""

import asyncio
import json
import logging
import sys
import tempfile
import os
import time
import uuid

from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session_maker
from scripts.utils import ist_now, parse_timestamp
from scripts.ssh_executor import SshExecutor
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval


class UpdateBatcher:
    """Batches database updates to reduce write operations using SQLAlchemy."""

    def __init__(self, batch_size=5, max_delay=2.0):
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.batch = []
        self.last_flush = time.time()

    async def add_update(self, obj):
        """Add an object to the batch queue."""
        self.batch.append(obj)
        if (
            len(self.batch) >= self.batch_size
            or (time.time() - self.last_flush) > self.max_delay
        ):
            await self.flush()

    async def flush(self):
        """Flush all pending updates to the database."""
        if not self.batch:
            return

        # Use a single session for all updates
        async with async_session_maker() as session:
            # Add all objects to the session
            for obj in self.batch:
                session.add(obj)

            # Commit all changes at once
            await session.commit()

            # Clear the batch
            self.batch = []
            self.last_flush = time.time()


class TrainingJobMonitor:
    """
    Monitors the global.json file on GPU server and updates database in real-time.
    Parses training events and updates TrainingJob, TrainingIteration, EpochTrain, and Eval tables.
    """

    def __init__(
        self,
        training_job_uuid: str,
        ssh_executor: SshExecutor,
        remote_log_path: str = "output/*/logs/global.json",
        poll_interval: int = 5,
        logger: logging.Logger = None,
        completion_callback=None,
    ):
        """
        Initialize the monitor.

        Args:
            training_job_uuid: UUID of the TrainingJob record
            ssh_executor: SSH executor for file operations
            remote_log_path: Path to global.json on remote server (supports wildcards)
            poll_interval: Polling interval in seconds
            logger: Logger instance
            completion_callback: Async callback function(success: bool, error_message: Optional[str]) to invoke when training completes
        """
        self.training_job_uuid = training_job_uuid
        self.training_job = None  # Will be loaded async in start_monitoring
        self.ssh_executor = ssh_executor
        self.remote_log_path = remote_log_path
        self.poll_interval = poll_interval
        self.logger = logger or self._setup_logger()
        self.update_batcher = UpdateBatcher()
        self.completion_callback = completion_callback

        # Track processed lines to avoid duplicates
        self.processed_line_count = 0

        # Track current iteration and epoch for database updates
        self.current_project: Optional[TrainingIteration] = None
        self.current_group_iteration: Optional[TrainingIteration] = None
        self.current_iteration: Optional[TrainingIteration] = None
        self.current_iteration_number: Optional[int] = None
        self.iteration_start_time: Optional[datetime] = None

        self.current_trajectory: Optional[TrainingIteration] = None
        self.current_training: Optional[TrainingIteration] = None
        self.current_evaluation: Optional[TrainingIteration] = None

        self.current_epoch_train: Optional[EpochTrain] = None
        self.current_eval_model: Optional[Eval] = None

        # Track phases

        # Flag to stop monitoring
        self.should_stop = False
        self.training_completed_successfully = False
        self.training_error_message = None

    async def _safe_db_operation(self, operation_name: str, operation_func):
        """Safely execute database operations with error handling."""
        try:
            # Create a new session for the operation
            async with async_session_maker() as session:
                # If operation_func is callable, call it with the session
                if callable(operation_func):
                    return await operation_func(session)
                # Otherwise, assume it's already an awaitable
                else:
                    return await operation_func
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"Database operation '{operation_name}' failed: {error_msg}"
            )
            return None

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the monitor."""
        logger = logging.getLogger(f"TrainingJobMonitor-{self.training_job_uuid[:8]}")
        logger.setLevel(logging.INFO)

        # Only add handler if none exists to avoid duplicate logs
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = (
                False  # Prevent duplicate logs in case parent logger is configured
            )

        return logger

    def _get_timestamp(self, event: Dict[str, Any]) -> str:
        event_timestamp = parse_timestamp(event.get("timestamp"))
        resp = event_timestamp.isoformat() if event_timestamp else ist_now().isoformat()

        return resp

    def validate_duration(
        self, duration_value, start_time, end_time, context=""
    ) -> float:
        """Validate and calculate duration, falling back to timestamp calculation if invalid.

        Args:
            duration_value: Duration value from event data
            start_time: Start timestamp (datetime or ISO string)
            end_time: End timestamp (datetime or ISO string)
            context: Optional context for logging (e.g., 'eval', 'training')

        Returns:
            float: Valid duration in minutes
        """
        try:
            # Try to convert duration to float and validate
            duration_float = (
                float(duration_value) if duration_value is not None else None
            )

            # Check if duration is unreasonable (> 24 hours or negative)
            if duration_float is None or duration_float > 24 * 60 or duration_float < 0:
                context_msg = f" {context}" if context else ""
                self.logger.warning(
                    f"Invalid{context_msg} duration detected: {duration_float} min, calculating from timestamps"
                )
                duration_float = self.get_duration(start_time, end_time)
        except (ValueError, TypeError):
            context_msg = f" {context}" if context else ""
            self.logger.warning(
                f"Invalid{context_msg} duration format: {duration_value}, calculating from timestamps"
            )
            duration_float = self.get_duration(start_time, end_time)

        return duration_float

    def get_duration(self, start_time, end_time):
        """Calculate duration in minutes between two timestamps.

        Args:
            start_time: datetime object or ISO format string
            end_time: datetime object or ISO format string

        Returns:
            float: Duration in minutes
        """
        try:
            # Convert start_time to datetime if it's a string
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            elif not isinstance(start_time, datetime):
                self.logger.error(f"Invalid start_time type: {type(start_time)}")
                return 0.0

            # Convert end_time to datetime if it's a string
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            elif not isinstance(end_time, datetime):
                self.logger.error(f"Invalid end_time type: {type(end_time)}")
                return 0.0

            # Calculate duration in minutes
            duration = (end_time - start_time).total_seconds() / 60
            return max(0.0, duration)  # Ensure non-negative duration

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error calculating duration: {e}")
            return 0.0

    async def _load_training_job(self, session: AsyncSession):
        """Load a training job by UUID using SQLAlchemy."""
        stmt = select(TrainingJob).where(TrainingJob.uuid == self.training_job_uuid)
        result = await session.execute(stmt)
        return result.scalars().first()

    async def _run_db_operation(self, operation_func):
        """Run a database operation with a new session."""
        async with async_session_maker() as session:
            if callable(operation_func):
                return await operation_func(session)
            else:
                return await operation_func

    async def start_monitoring(self):
        """Start monitoring the global.json file."""
        self.logger.info(f"🔍 Starting monitoring for job {self.training_job_uuid}")

        try:
            # Load training job using SQLAlchemy
            async with async_session_maker() as session:
                self.training_job = await self._load_training_job(session)
                if not self.training_job:
                    self.logger.error(
                        f"❌ Training job {self.training_job_uuid} not found"
                    )
                    return

                self.logger.info(
                    f"✅ Loaded training job: {self.training_job.project_id}"
                )
        except Exception as e:
            self.logger.error(
                f"❌ Failed to load training job {self.training_job_uuid}: {e}"
            )
            return

        try:
            while not self.should_stop:
                try:
                    await self._poll_and_update()
                    await asyncio.sleep(self.poll_interval)
                except asyncio.CancelledError:
                    self.logger.info(
                        f"Monitoring cancelled for job {self.training_job_uuid}"
                    )
                    raise  # Re-raise to properly handle cancellation
                except Exception as e:
                    self.logger.error(f"Error during polling: {str(e)}")
                    try:
                        await asyncio.sleep(self.poll_interval)
                    except asyncio.CancelledError:
                        self.logger.info(
                            f"Monitoring cancelled during error recovery for job {self.training_job_uuid}"
                        )
                        raise

        except asyncio.CancelledError:
            self.logger.info(f"Monitoring cancelled for job {self.training_job_uuid}")
            raise
        except Exception as e:
            self.logger.error(f"Fatal error in monitoring: {str(e)}")
            self.training_error_message = str(e)
            try:
                await self._mark_job_failed(str(e))
            except Exception as db_error:
                self.logger.error(f"Failed to mark job as failed: {db_error}")
            raise
        finally:
            # Invoke completion callback if provided
            if self.completion_callback:
                try:
                    await self.completion_callback(
                        success=self.training_completed_successfully,
                        error_message=self.training_error_message,
                    )
                except Exception as callback_error:
                    self.logger.error(f"Error in completion callback: {callback_error}")

    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.logger.info("🛑 Stopping monitoring")
        self.should_stop = True

    async def _check_script_running(self) -> bool:
        """Check if the training script is still running on the GPU server.

        Returns:
            bool: True if script is running, False otherwise
        """
        try:
            # Check if bash process running run_docker_job.sh exists
            check_cmd = "pgrep -f 'bash.*run_docker_job.sh' || echo 'not_running'"
            result = await asyncio.to_thread(
                self.ssh_executor.execute_command, check_cmd, check=False
            )

            if result.success:
                output = result.stdout.strip()
                # If we get 'not_running' or empty output, script is not running
                if output == "not_running" or not output:
                    return False
                # If we get a PID, script is still running
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check script status: {e}")
            return True  # Assume running if we can't check

    async def _poll_and_update(self):
        """Poll the global.json file and update database."""
        try:
            # Download global.json from remote server
            log_content = await self._download_log_file()

            if not log_content:
                # Check if script is still running
                script_running = await self._check_script_running()
                if not script_running:
                    self.logger.warning(
                        "Script has stopped but no log file found - possible failure"
                    )
                    self.training_error_message = (
                        "Training script stopped without producing logs"
                    )
                    await self.stop_monitoring()
                    return

                # If we couldn't get the log file, increase the polling interval temporarily
                # to avoid hammering the server
                await asyncio.sleep(
                    min(self.poll_interval * 2, 30)
                )  # Cap at 30 seconds
                return

            # Parse new lines
            lines = log_content.strip().split("\n")
            new_lines = lines[self.processed_line_count :]

            if not new_lines:
                # Check if script completed but no new logs
                script_running = await self._check_script_running()
                if not script_running and not self.training_completed_successfully:
                    self.logger.warning(
                        "Script has stopped but training not marked complete - checking for errors"
                    )
                    # Check the script log file for errors
                    await self._check_script_log_for_errors()
                return

            # Process each new line
            for line in new_lines:
                if line.strip():
                    await self._process_log_line(line)

            # Update processed count
            self.processed_line_count = len(lines)

        except Exception as e:
            self.logger.warning(f"Failed to poll log file: {str(e)}")

    async def _download_log_file(self) -> Optional[str]:
        """Download the global.json file from remote server."""
        tmp_path = None
        try:
            # First, ensure SSH connection is active
            if not await self._ensure_ssh_connection():
                self.logger.warning(
                    "Cannot download log file: SSH connection unavailable"
                )
                return None

            # Find the actual path (resolve wildcards)
            result = self.ssh_executor.execute_command(
                f"find . -path './{self.remote_log_path}' -type f | head -1",
                check=False,
            )

            if not result.success or not result.stdout.strip():
                return None

            actual_path = result.stdout.strip()

            # Create temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as tmp:
                tmp_path = tmp.name

            # Download file with proper error handling
            try:
                self.ssh_executor.download_file(actual_path, tmp_path)

                # Read content with proper file handling
                with open(tmp_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    self.logger.debug("Downloaded log file is empty")
                    return None

                return content

            except Exception as download_error:
                self.logger.debug(
                    f"Failed to download or read log file: {download_error}"
                )
                return None

        except Exception as e:
            self.logger.debug(f"Could not download log file: {str(e)}")
            return None

        finally:
            # Ensure cleanup in all code paths
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as cleanup_error:
                    self.logger.warning(
                        f"Failed to cleanup temporary file {tmp_path}: {cleanup_error}"
                    )

    async def _run_transaction(self, operation_func):
        """Run a database operation within a transaction."""
        async with async_session_maker() as session:
            async with session.begin():
                if callable(operation_func):
                    return await operation_func(session)
                else:
                    return await operation_func

    async def _check_script_log_for_errors(self) -> None:
        """Check the script log file for error messages."""
        try:
            result = await asyncio.to_thread(
                self.ssh_executor.execute_command,
                "tail -n 50 run_docker_job.sh.log 2>/dev/null || echo 'no_log'",
                check=False,
            )

            if result.success and result.stdout.strip() != "no_log":
                log_content = result.stdout.strip()
                self.logger.error(f"Script log (last 50 lines):\n{log_content}")

                # Look for common error patterns
                if "Error:" in log_content or "failed" in log_content.lower():
                    self.training_error_message = (
                        "Training script failed - check logs for details"
                    )
                else:
                    self.training_error_message = "Training script stopped unexpectedly"
            else:
                self.training_error_message = (
                    "Training script stopped and no log file found"
                )

            await self.stop_monitoring()
        except Exception as e:
            self.logger.error(f"Failed to check script log: {e}")
            self.training_error_message = f"Failed to check script status: {e}"
            await self.stop_monitoring()

    async def _ensure_ssh_connection(self):
        """Ensure SSH connection is active, attempt to reconnect if not."""
        try:
            # Simple test command to check connection
            result = self.ssh_executor.execute_command(
                "echo 'connection_test'", check=False
            )
            if not result.success:
                self.logger.warning(
                    "SSH connection appears to be down, attempting to reconnect"
                )
                reconnected = self.ssh_executor.reconnect()
                # Verify reconnection
                if reconnected:
                    verify = self.ssh_executor.execute_command(
                        "echo 'connection_test'", check=False
                    )
                    if verify.success:
                        self.logger.info("✅ SSH connection re-established")
                        return True
                    else:
                        self.logger.error(
                            "Failed to verify SSH connection after reconnect"
                        )
                        return False
                else:
                    self.logger.error("Failed to re-establish SSH connection")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking SSH connection: {str(e)}")
            return False

    async def _process_event_in_transaction(self, event):
        """Process an event within a transaction."""
        phase = event.get("phase")

        # Use SQLAlchemy transaction
        async with async_session_maker() as session:
            async with session.begin():
                # Pass the session to each handler method
                if phase == "PROJECT":
                    await self._handle_project_event(event, session)
                elif phase == "GROUP_ITERATION":
                    await self._handle_group_iteration_event(event, session)
                elif phase == "ITERATION":
                    await self._handle_iteration_event(event, session)
                elif phase == "TRAJECTORY":
                    await self._handle_trajectory_event(event, session)
                elif phase == "TRAINING":
                    await self._handle_training_event(event, session)
                elif phase == "EPOCH_TRAIN":
                    await self._handle_epoch_train_event(event, session)
                elif phase == "EVAL_TRAINING":
                    await self._handle_eval_training_event(event, session)
                elif phase == "EVAL_MODEL":
                    await self._handle_eval_model_event(event, session)

                # Flush all batched updates at the end of transaction
                # No need to call flush explicitly as session.commit() will be called automatically

    async def _process_log_line(self, line: str):
        """Process a single log line and update database."""
        try:
            event = json.loads(line)
            phase = event.get("phase")

            self.logger.debug(f"Processing: {phase}")

            # Process the event with SQLAlchemy transaction
            await self._process_event_in_transaction(event)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse log line: {line[:100]}... Error: {e}")
        except Exception as e:
            self.logger.error(f"Error processing log line: {str(e)}.. Line: {line}")

    async def _handle_project_event(self, event: Dict[str, Any], session: AsyncSession):
        """Handle PROJECT phase events."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            event_timestamp = self._get_timestamp(event)

            # Create new TrainingIteration using SQLAlchemy
            self.current_project = TrainingIteration(
                uuid=str(uuid.uuid4()),
                training_job_uuid=self.training_job.uuid,
                iteration_number=1,
                step_type=StepType.PROJECT,
                step_config=data.get("config", {}),
                created_at=event_timestamp,
            )

            # Add to session
            session.add(self.current_project)
            # No need to call commit as it's handled by the transaction

            self.logger.info(f"📋 Project started at: {event_timestamp}")

        elif event_type == "end":
            if not self.current_project:
                self.logger.error("No current project to complete")
                return

            self.current_project.completed_at = self._get_timestamp(event)
            data = event.get("data", {})
            duration = data.get("duration", None)
            if not duration:
                duration = self.get_duration(
                    self.current_project.created_at, self.current_project.completed_at
                )
            self.current_project.time_taken = duration

            # Add to session
            session.add(self.current_project)
            # No need to call commit as it's handled by the transaction

            self.logger.info("✅ Project completed")

            # Mark training as successfully completed
            self.training_completed_successfully = True

            # Stop monitoring - training is complete
            await self.stop_monitoring()

    async def _handle_group_iteration_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle GROUP_ITERATION phase events."""
        event_type = event.get("event")
        if event_type == "start":
            event_timestamp = self._get_timestamp(event)
            data = event.get("data", {})
            config = data.get("config", {})

            # Create new TrainingIteration using SQLAlchemy
            self.current_group_iteration = TrainingIteration(
                uuid=str(uuid.uuid4()),
                training_job_uuid=self.training_job.uuid,
                iteration_number=config.get("no_iterations", None),
                step_type=StepType.GROUP_ITERATION,
                step_config=config,
                created_at=event_timestamp,
            )

            # Add to session
            session.add(self.current_group_iteration)
            # No need to call commit as it's handled by the transaction

            self.logger.info(
                f"🔄 Group iteration started with {config.get('no_iterations')} iterations"
            )
        elif event_type == "end":
            if not self.current_group_iteration:
                self.logger.error("No current group iteration to complete")
                return

            self.current_group_iteration.completed_at = self._get_timestamp(event)
            duration = event.get("data", {}).get("duration", None)
            if not duration:
                duration = self.get_duration(
                    self.current_group_iteration.created_at,
                    self.current_group_iteration.completed_at,
                )
            self.current_group_iteration.time_taken = duration

            # Add to session
            session.add(self.current_group_iteration)
            # No need to call commit as it's handled by the transaction

            self.logger.info("✅ Group iteration completed")

    async def _handle_iteration_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle ITERATION phase events."""

        event_type = event.get("event")
        data = event.get("data", {})
        config = data.get("config", {})

        if event_type == "start":
            # Extract iteration number from nested runtime config
            runtime_config = config.get("runtime", {})
            iteration_number = runtime_config.get("current_iteration")
            self.current_iteration_number = iteration_number
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)

            # Create TrainingIteration record with SQLAlchemy
            self.current_iteration = TrainingIteration(
                uuid=str(uuid.uuid4()),
                training_job_uuid=self.training_job.uuid,
                iteration_number=iteration_number,
                step_type=StepType.ITERATION,
                step_config=config,
                created_at=event_timestamp,
            )

            # Add to session
            session.add(self.current_iteration)
            # No need to call commit as it's handled by the transaction

            self.logger.info(f"🔢 Iteration {iteration_number} started")

        elif event_type == "end":
            if not self.current_iteration:
                return

            # Set completion timestamp and save
            self.current_iteration.completed_at = self._get_timestamp(event)

            # Calculate duration from event data or timestamps
            duration = data.get("duration")
            if duration:
                self.current_iteration.time_taken = float(duration)
            else:
                self.current_iteration.time_taken = self.get_duration(
                    self.current_iteration.created_at,
                    self.current_iteration.completed_at,
                )

            # Add to session
            session.add(self.current_iteration)
            # No need to call commit as it's handled by the transaction

            iteration_number = self.current_iteration_number  # Store before clearing
            self.current_iteration = None
            self.logger.info(f"✅ Iteration {iteration_number} completed")

    async def _handle_trajectory_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle TRAJECTORY phase events."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create a trajectory generation step with SQLAlchemy
            if self.current_iteration and self.current_iteration_number is not None:
                self.current_trajectory = TrainingIteration(
                    uuid=str(uuid.uuid4()),
                    training_job_uuid=self.training_job.uuid,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.TRAJECTORY,
                    step_config=config,
                    created_at=event_timestamp,
                )

                # Add to session
                session.add(self.current_trajectory)
                # No need to call commit as it's handled by the transaction

                self.logger.info(
                    f"🎯 Trajectory generation started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning(
                    f"No current iteration for trajectory generation. "
                    f"current_iteration={self.current_iteration}, "
                    f"current_iteration_number={self.current_iteration_number}"
                )

        elif event_type == "end":
            # Critical null check - ensure current_trajectory exists
            if not self.current_trajectory:
                self.logger.error("No current trajectory to complete")
                return

            self.current_trajectory.completed_at = self._get_timestamp(event)
            duration = data.get("duration")
            if not duration:
                duration = self.get_duration(
                    self.current_trajectory.created_at,
                    self.current_trajectory.completed_at,
                )
            self.current_trajectory.time_taken = duration

            # Add to session
            session.add(self.current_trajectory)
            # No need to call commit as it's handled by the transaction

            self.current_trajectory = None
            self.logger.info(f"✅ Trajectory generation completed in {duration} min")

    async def _handle_training_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle TRAINING phase events for overall training iteration (start/end)."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create a training step with SQLAlchemy
            if self.current_iteration and self.current_iteration_number is not None:
                self.current_training = TrainingIteration(
                    uuid=str(uuid.uuid4()),
                    training_job_uuid=self.training_job.uuid,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.TRAINING,
                    step_config=config,
                    created_at=event_timestamp,
                )

                # Add to session
                session.add(self.current_training)
                # No need to call commit as it's handled by the transaction

                self.logger.info(
                    f"🏋️ Training started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning(
                    f"No current iteration for training. "
                    f"current_iteration={self.current_iteration}, "
                    f"current_iteration_number={self.current_iteration_number}"
                )

        elif event_type == "end":
            # Critical null check - ensure current_training exists
            if not self.current_training:
                self.logger.error("No current training to complete")
                return

            self.current_training.completed_at = self._get_timestamp(event)
            duration = data.get("duration")

            # Validate and get duration
            duration_float = self.validate_duration(
                duration,
                self.current_training.created_at,
                self.current_training.completed_at,
                context="training",
            )

            self.current_training.time_taken = duration_float

            # Add to session
            session.add(self.current_training)
            # No need to call commit as it's handled by the transaction

            self.current_training = None

            self.logger.info(f"✅ Training completed in {duration_float:.2f} min")

    async def _handle_epoch_train_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle TRAINING phase epoch_complete events for individual epochs."""
        event_type = event.get("event")

        if event_type == "start":
            # Critical null check - ensure current_training exists
            if not self.current_training:
                self.logger.error("No current training for epoch event")
                return

            created_at = self._get_timestamp(event)
            data = event.get("data", {})
            epoch = data.get("epoch")

            # Validate epoch number
            try:
                epoch = int(epoch)
                if epoch < 1:
                    self.logger.error(f"Invalid epoch number: {epoch}")
                    return
            except (ValueError, TypeError):
                self.logger.error(f"Invalid epoch format: {epoch}")
                return

            # Validate iteration_number exists
            if self.current_iteration_number is None:
                self.logger.error(
                    f"Cannot create epoch train: iteration_number is None. "
                    f"current_iteration={self.current_iteration}, "
                    f"current_training={self.current_training}"
                )
                return

            # Create EpochTrain with SQLAlchemy
            self.current_epoch_train = EpochTrain(
                uuid=str(uuid.uuid4()),
                iteration_uuid=self.current_training.uuid,
                iteration_number=self.current_iteration_number,
                epoch_number=epoch,
                created_at=created_at,
            )

            # Add to session
            session.add(self.current_epoch_train)
            # No need to call commit as it's handled by the transaction

        elif event_type == "end":
            # Critical null check - ensure current_epoch_train exists
            if not self.current_epoch_train:
                self.logger.error("No current epoch train to complete")
                return

            self.current_epoch_train.completed_at = self._get_timestamp(event)
            data = event.get("data", {})
            epoch = data.get("epoch", "unknown")  # Store epoch for logging

            duration = data.get("duration")

            # Validate and get duration
            duration_float = self.validate_duration(
                duration,
                self.current_epoch_train.created_at,
                self.current_epoch_train.completed_at,
                context="epoch training",
            )

            self.current_epoch_train.time_taken = duration_float

            self.current_epoch_train.metrics = data.get("metrics", {})
            self.current_epoch_train.model_path = data.get("model_path", None)
            self.current_epoch_train.optimizer_path = data.get("optimizer_path", None)

            # Add to session
            session.add(self.current_epoch_train)
            # No need to call commit as it's handled by the transaction

            self.current_epoch_train = None

            self.logger.info(f"📊 Epoch {epoch} completed in {duration_float:.2f} min")

    async def _handle_eval_training_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle EVAL_TRAINING phase events for overall evaluation iteration."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create an evaluation step with SQLAlchemy
            if self.current_iteration and self.current_iteration_number is not None:
                self.current_evaluation = TrainingIteration(
                    uuid=str(uuid.uuid4()),
                    training_job_uuid=self.training_job.uuid,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.EVALUATION,
                    step_config=config,
                    created_at=event_timestamp,
                )

                # Add to session
                session.add(self.current_evaluation)
                # No need to call commit as it's handled by the transaction

                self.logger.info(
                    f"📈 Evaluation started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning(
                    f"No current iteration for evaluation. "
                    f"current_iteration={self.current_iteration}, "
                    f"current_iteration_number={self.current_iteration_number}"
                )

        elif event_type == "end":
            # Critical null check - ensure current_evaluation exists
            if not self.current_evaluation:
                self.logger.error("No current evaluation to complete")
                return

            self.current_evaluation.completed_at = self._get_timestamp(event)
            duration = data.get("duration")
            if not duration:
                duration = self.get_duration(
                    self.current_evaluation.created_at,
                    self.current_evaluation.completed_at,
                )
            self.current_evaluation.time_taken = float(duration)

            # Add to session
            session.add(self.current_evaluation)
            # No need to call commit as it's handled by the transaction

            self.logger.info(f"✅ Evaluation completed in {float(duration):.2f} min")

        elif event_type == "metrics":
            # Critical null check - ensure current_evaluation exists
            if not self.current_evaluation:
                self.logger.error("No current evaluation for metrics update")
                return

            self.current_evaluation.metrics = data.get("metrics", {})

            # Add to session
            session.add(self.current_evaluation)
            # No need to call commit as it's handled by the transaction

            self.logger.info(f"📊 Evaluation metrics updated")

    async def _handle_eval_model_event(
        self, event: Dict[str, Any], session: AsyncSession
    ):
        """Handle EVAL_MODEL phase events for specific epoch model evaluation."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Ensure we have a current evaluation to link to
            if not self.current_evaluation:
                self.logger.error("No current evaluation for model evaluation")
                return

            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)

            model_path = data.get("model_path", "")
            config = data.get("config", {})
            model_epoch = config.get("model_epoch", "")
            dataset = config.get("dataset", "eval")
            training_name = config.get("training_name", "")

            # Create Eval record with SQLAlchemy
            self.current_eval_model = Eval(
                uuid=str(uuid.uuid4()),
                model_id=f"iteration_{self.current_iteration_number}_epoch_{model_epoch}",
                dataset=dataset,
                config=config,
                created_at=event_timestamp,
                eval_metadata={
                    "model_path": model_path,
                    "training_name": training_name,
                    "model_epoch": model_epoch,
                },
                iteration_uuid=self.current_evaluation.uuid,
            )

            # Add to session
            session.add(self.current_eval_model)
            # No need to call commit as it's handled by the transaction

            self.logger.info(
                f"🔍 Model evaluation started for {dataset}_epoch_{model_epoch}"
            )

        elif event_type == "metrics":
            metrics = data.get("metrics", {})
            metrics_json_path = data.get("metrics_json_path", "")

            # Update the current Eval record with metrics
            if self.current_eval_model:
                self.current_eval_model.metrics = metrics
                self.current_eval_model.eval_data_path = metrics_json_path

                # Add to session
                session.add(self.current_eval_model)
                # No need to call commit as it's handled by the transaction

                self.logger.info(f"📊 Evaluation metrics recorded: {metrics}")
            else:
                self.logger.warning(
                    "Received metrics event without a current eval record"
                )

        elif event_type == "end":
            # Update the current Eval record with completion time
            if self.current_eval_model:
                self.current_eval_model.completed_at = self._get_timestamp(event)
                duration = data.get("duration", None)

                # Validate and get duration
                duration_float = self.validate_duration(
                    duration,
                    self.current_eval_model.created_at,
                    self.current_eval_model.completed_at,
                    context="eval model",
                )

                self.current_eval_model.time_taken = duration_float

                # Add to session
                session.add(self.current_eval_model)
                # No need to call commit as it's handled by the transaction

                self.logger.info(f"✅ Model evaluation completed")

                # Reset current eval
                self.current_eval_model = None
            else:
                self.logger.warning("Received end event without a current eval record")

    async def _mark_job_failed(self, error_message: str):
        """Mark the job as failed."""

        async def mark_failed(session: AsyncSession):
            # Get job with SQLAlchemy
            stmt = select(TrainingJob).where(TrainingJob.uuid == self.training_job_uuid)
            result = await session.execute(stmt)
            job = result.scalars().first()

            if job:
                job.status = TrainingJobStatus.FAILED
                job.completed_at = ist_now().isoformat()
                # Store error in job_metadata
                if job.job_metadata is None:
                    job.job_metadata = {}
                job.job_metadata["error"] = error_message

                # Add to session
                session.add(job)
                # Commit is handled by the transaction
                return job
            return None

        result = await self._safe_db_operation("mark_job_failed", mark_failed)
        if result:
            self.logger.error(f"❌ Job marked as failed: {error_message}")
        else:
            self.logger.error(
                f"Failed to mark job as failed in database: {error_message}"
            )
