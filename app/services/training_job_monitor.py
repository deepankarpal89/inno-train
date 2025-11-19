"""
TrainingJobMonitor - Monitors global.json file on GPU server and updates database
"""

import asyncio
import json
import logging
import tempfile
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from scripts.utils import ist_now, parse_timestamp
from scripts.ssh_executor import SshExecutor
from models.training_job import TrainingJob, TrainingJobStatus
from models.training_iteration import TrainingIteration, StepType
from models.epoch_train import EpochTrain
from models.eval import Eval
from test import event_timestamp


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
    ):
        """
        Initialize the monitor.

        Args:
            training_job_uuid: UUID of the TrainingJob record
            ssh_executor: SSH executor for file operations
            remote_log_path: Path to global.json on remote server (supports wildcards)
            poll_interval: Polling interval in seconds
            logger: Logger instance
        """
        self.training_job_uuid = training_job_uuid
        self.training_job = None  # Will be loaded async in start_monitoring
        self.ssh_executor = ssh_executor
        self.remote_log_path = remote_log_path
        self.poll_interval = poll_interval
        self.logger = logger or self._setup_logger()

        # Track processed lines to avoid duplicates
        self.processed_line_count = 0

        # Track current iteration and epoch for database updates
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

    async def _safe_db_operation(self, operation_name: str, operation_func):
        """Safely execute database operations with error handling."""
        try:
            return await operation_func()
        except Exception as e:
            error_msg = str(e)
            if "Event loop is closed" in error_msg or "different loop" in error_msg:
                self.logger.warning(
                    f"Database operation '{operation_name}' failed due to event loop issues: {error_msg}"
                )
            else:
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

    def _get_timestamp(self, event: Dict[str, Any]) -> datetime:
        event_timestamp = parse_timestamp(event.get("timestamp"))
        return event_timestamp if event_timestamp else ist_now()

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

    async def start_monitoring(self):
        """Start monitoring the global.json file."""
        self.logger.info(f"🔍 Starting monitoring for job {self.training_job_uuid}")

        try:
            # Update job status to RUNNING
            async def update_job_status():
                self.training_job = await TrainingJob.get(uuid=self.training_job_uuid)
                self.training_job.status = TrainingJobStatus.RUNNING
                await self.training_job.save()

            await self._safe_db_operation(
                "update_job_status_running", update_job_status
            )

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
            try:
                await self._mark_job_failed(str(e))
            except Exception as db_error:
                self.logger.error(f"Failed to mark job as failed: {db_error}")

    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.logger.info("🛑 Stopping monitoring")
        self.should_stop = True

    async def _poll_and_update(self):
        """Poll the global.json file and update database."""
        try:
            # Download global.json from remote server
            log_content = await self._download_log_file()

            if not log_content:
                return

            # Parse new lines
            lines = log_content.strip().split("\n")
            new_lines = lines[self.processed_line_count :]

            if not new_lines:
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
            # First, find the actual path (resolve wildcards)
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

    async def _process_log_line(self, line: str):
        """Process a single log line and update database."""
        try:
            event = json.loads(line)
            phase = event.get("phase")

            self.logger.debug(f"Processing: {phase}")

            # Route to appropriate handler
            if phase == "PROJECT":
                await self._handle_project_event(event)
            elif phase == "GROUP_ITERATION":
                await self._handle_group_iteration_event(event)
            elif phase == "ITERATION":
                await self._handle_iteration_event(event)
            elif phase == "TRAJECTORY":
                await self._handle_trajectory_event(event)
            elif phase == "TRAINING":
                await self._handle_training_event(event)
            elif phase == "EPOCH_TRAIN":
                await self._handle_epoch_train_event(event)
            elif phase == "EVAL_TRAINING":
                await self._handle_eval_training_event(event)
            elif phase == "EVAL_MODEL":
                await self._handle_eval_model_event(event)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse log line: {line[:100]}... Error: {e}")
        except Exception as e:
            self.logger.error(f"Error processing log line: {str(e)}.. Line: {line}")

    async def _handle_project_event(self, event: Dict[str, Any]):
        """Handle PROJECT phase events."""
        # Critical null check - ensure training_job is loaded
        if not self.training_job:
            self.logger.error("Training job not loaded, cannot process PROJECT event")
            return

        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            event_timestamp = self._get_timestamp(event)
            self.training_job.created_at = event_timestamp
            self.training_job.project_config = data.get("config", {})
            await self.training_job.save()
            self.logger.info(f"📋 Project started at: {event_timestamp}")

        elif event_type == "end":
            self.training_job.completed_at = self._get_timestamp(event)
            data = event.get("data", {})
            duration = data.get("duration", None)
            if not duration:
                duration = self.get_duration(
                    self.training_job.created_at, self.training_job.completed_at
                )
            self.training_job.time_taken = duration
            # Mark job as completed
            self.training_job.status = TrainingJobStatus.COMPLETED
            await self.training_job.save()
            self.logger.info("✅ Project completed")

            # Stop monitoring - training is complete
            self.should_stop = True

    async def _handle_group_iteration_event(self, event: Dict[str, Any]):
        """Handle GROUP_ITERATION phase events."""
        event_type = event.get("event")
        if event_type == "start":
            event_timestamp = self._get_timestamp(event)
            data = event.get("data", {})
            config = data.get("config", {})
            self.current_group_iteration = await TrainingIteration.create(
                training_job=self.training_job,
                iteration_number=config.get("no_iterations", None),
                step_type=StepType.GROUP_ITERATION,
                step_config=config,
                created_at=event_timestamp,
            )
            await self.current_group_iteration.save()
            self.logger.info(
                f"🔄 Group iteration started with {config.get('no_iterations')} iterations"
            )
        elif event_type == "end":

            self.current_group_iteration.completed_at = self._get_timestamp(event)
            duration = event.get("data", {}).get("duration", None)
            if not duration:
                duration = self.get_duration(
                    self.current_group_iteration.created_at,
                    self.current_group_iteration.completed_at,
                )
            self.current_group_iteration.time_taken = duration
            await self.current_group_iteration.save()
            self.logger.info("✅ Group iteration completed")

    async def _handle_iteration_event(self, event: Dict[str, Any]):
        """Handle ITERATION phase events."""
        if not self.training_job:
            self.logger.error("Training job not loaded, cannot process ITERATION event")
            return

        event_type = event.get("event")
        data = event.get("data", {})
        config = data.get("config", {})

        if event_type == "start":
            iteration_number = config.get("current_iteration")
            self.current_iteration_number = iteration_number
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)

            # Create TrainingIteration record
            self.current_iteration = await TrainingIteration.create(
                training_job=self.training_job,
                iteration_number=iteration_number,
                step_type=StepType.ITERATION,
                step_config=config,
                created_at=event_timestamp,
            )

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

            await self.current_iteration.save()

            iteration_number = self.current_iteration_number  # Store before clearing
            self.current_iteration = None
            self.logger.info(f"✅ Iteration {iteration_number} completed")

    async def _handle_trajectory_event(self, event: Dict[str, Any]):
        """Handle TRAJECTORY phase events."""
        if not self.training_job:
            self.logger.error(
                "Training job not loaded, cannot process TRAJECTORY event"
            )
            return

        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create a trajectory generation step
            if self.current_iteration:
                self.current_trajectory = await TrainingIteration.create(
                    training_job=self.training_job,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.TRAJECTORY_GENERATION,
                    step_config=config,
                    created_at=event_timestamp,
                )

                self.logger.info(
                    f"🎯 Trajectory generation started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning("No current iteration for trajectory generation")

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
            await self.current_trajectory.save()

            self.current_trajectory = None
            self.logger.info(f"✅ Trajectory generation completed in {duration} min")

    async def _handle_training_event(self, event: Dict[str, Any]):
        """Handle TRAINING phase events for overall training iteration (start/end)."""
        if not self.training_job:
            self.logger.error("Training job not loaded, cannot process TRAINING event")
            return

        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create a training step
            if self.current_iteration:
                self.current_training = await TrainingIteration.create(
                    training_job=self.training_job,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.TRAINING,
                    step_config=config,
                    created_at=event_timestamp,
                )

                self.logger.info(
                    f"🏋️ Training started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning("No current iteration for training")

        elif event_type == "end":
            # Critical null check - ensure current_training exists
            if not self.current_training:
                self.logger.error("No current training to complete")
                return

            self.current_training.completed_at = self._get_timestamp(event)
            duration = data.get("duration")
            if not duration:
                duration = self.get_duration(
                    self.current_training.created_at, self.current_training.completed_at
                )
            self.current_training.time_taken = duration
            await self.current_training.save()

            self.current_training = None

            self.logger.info(f"✅ Training completed in {duration} min")

    async def _handle_epoch_train_event(self, event: Dict[str, Any]):
        """Handle TRAINING phase epoch_complete events for individual epochs."""
        if not self.training_job:
            self.logger.error(
                "Training job not loaded, cannot process EPOCH_TRAIN event"
            )
            return

        event_type = event.get("event")

        if event_type == "start":
            # Critical null check - ensure current_training exists
            if not self.current_training:
                self.logger.error("No current training for epoch event")
                return

            created_at = self._get_timestamp(event)
            data = event.get("data", {})
            epoch = data.get("epoch")

            self.current_epoch_train = await EpochTrain.create(
                iteration_number=self.current_iteration_number,
                iteration=self.current_training,
                epoch_number=epoch,
                created_at=created_at,
            )

        elif event_type == "end":
            # Critical null check - ensure current_epoch_train exists
            if not self.current_epoch_train:
                self.logger.error("No current epoch train to complete")
                return

            self.current_epoch_train.completed_at = self._get_timestamp(event)
            data = event.get("data", {})
            epoch = data.get("epoch", "unknown")  # Store epoch for logging

            duration = data.get("duration")
            if not duration:
                duration = self.get_duration(
                    self.current_epoch_train.created_at,
                    self.current_epoch_train.completed_at,
                )
            self.current_epoch_train.time_taken = duration

            self.current_epoch_train.metrics = data.get("metrics", {})
            self.current_epoch_train.model_path = data.get("model_path", None)
            self.current_epoch_train.optimizer_path = data.get("optimizer_path", None)
            await self.current_epoch_train.save()

            self.current_epoch_train = None

            self.logger.info(f"📊 Epoch {epoch} completed")

    async def _handle_eval_training_event(self, event: Dict[str, Any]):
        """Handle EVAL_TRAINING phase events for overall evaluation iteration."""
        if not self.training_job:
            self.logger.error(
                "Training job not loaded, cannot process EVAL_TRAINING event"
            )
            return

        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            event_timestamp = self._get_timestamp(event)
            config = data.get("config", {})

            # Create an evaluation step
            if self.current_iteration:
                self.current_evaluation = await TrainingIteration.create(
                    training_job=self.training_job,
                    iteration_number=self.current_iteration_number,
                    step_type=StepType.EVALUATION,
                    step_config=config,
                    created_at=event_timestamp,
                )

                self.logger.info(
                    f"📈 Evaluation started for iteration {self.current_iteration_number}"
                )
            else:
                self.logger.warning("No current iteration for evaluation")

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
            await self.current_evaluation.save()

            self.current_evaluation = None

            self.logger.info(f"✅ Evaluation completed in {duration} min")

        elif event_type == "metrics":
            # Critical null check - ensure current_evaluation exists
            if not self.current_evaluation:
                self.logger.error("No current evaluation for metrics update")
                return

            self.current_evaluation.metrics = data.get("metrics", {})
            await self.current_evaluation.save()

            self.logger.info(f"📊 Evaluation metrics updated")

    async def _handle_eval_model_event(self, event: Dict[str, Any]):
        """Handle EVAL_MODEL phase events for specific epoch model evaluation."""
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "start":
            # Parse timestamp from event
            event_timestamp = self._get_timestamp(event)

            model_path = data.get("model_path", "")
            config = data.get("config", {})
            model_epoch = config.get("model_epoch", "")
            dataset = config.get("dataset", "cv")
            training_name = config.get("training_name", "")

            # Create Eval record with initial data
            self.current_eval_model = await Eval.create(
                model_id=f"iteration_{self.current_iteration_number}_epoch_{model_epoch}",
                dataset=dataset,
                config=config,
                created_at=event_timestamp,
                metadata={
                    "model_path": model_path,
                    "training_name": training_name,
                    "model_epoch": model_epoch,
                },
                iteration=self.current_evaluation,
            )

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
                await self.current_eval_model.save()

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
                if not duration:
                    duration = self.get_duration(
                        self.current_eval_model.created_at,
                        self.current_eval_model.completed_at,
                    )
                self.current_eval_model.time_taken = float(duration)

                await self.current_eval_model.save()

                self.logger.info(f"✅ Model evaluation completed")

                # Reset current eval
                self.current_eval_model = None
            else:
                self.logger.warning("Received end event without a current eval record")

    async def _mark_job_failed(self, error_message: str):
        """Mark the job as failed."""

        async def mark_failed():
            job = await TrainingJob.get(uuid=self.training_job_uuid)
            job.status = TrainingJobStatus.FAILED
            job.completed_at = ist_now()
            # Store error in metadata
            if job.metadata is None:
                job.metadata = {}
            job.metadata["error"] = error_message
            await job.save()
            return job

        result = await self._safe_db_operation("mark_job_failed", mark_failed)
        if result:
            self.logger.error(f"❌ Job marked as failed: {error_message}")
        else:
            self.logger.error(
                f"Failed to mark job as failed in database: {error_message}"
            )
