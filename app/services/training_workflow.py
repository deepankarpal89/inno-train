"""TrainingWorkflow - Clean, refactored training job orchestration."""

import asyncio
import logging
import os
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from app.database import async_session_maker
from scripts.lambda_client import LambdaClient
from scripts.project_yaml_builder import ProjectYamlBuilder
from scripts.s3_to_server_transfer import S3ToServerTransfer
from scripts.utils import ist_now, calculate_duration
from scripts.ssh_executor import SshExecutor
from app.services.training_job_monitor import TrainingJobMonitor
from models.training_job import TrainingJob, TrainingJobStatus
from concurrent.futures import ThreadPoolExecutor
from scripts.file_logger import get_file_logger
from app.services.db_service import db_service


class WorkflowError(Exception):
    """Base exception for workflow errors."""

    def __init__(self, message: str, job_uuid: str = None, cleanup_needed: bool = True):
        super().__init__(message)
        self.job_uuid = job_uuid
        self.cleanup_needed = cleanup_needed


class InfrastructureError(WorkflowError):
    """Raised when infrastructure operations fail."""

    pass


class FileTransferError(WorkflowError):
    """Raised when file transfer operations fail."""

    pass


class TrainingExecutionError(WorkflowError):
    """Raised when training execution fails."""

    pass


class JobNotFoundError(WorkflowError):
    """Raised when a job cannot be found."""

    pass


class WorkflowConstants:
    """Constants for training workflow configuration."""

    # SSH Configuration
    SSH_RETRY_ATTEMPTS = 5
    SSH_RETRY_BASE_DELAY = 2  # seconds
    SSH_USERNAME = "ubuntu"

    # Timing Configuration
    DEFAULT_TIMEOUT_MINUTES = 120
    POLL_INTERVAL_SECONDS = 10
    MONITORING_POLL_INTERVAL = 5

    # File Paths
    TRAINING_SCRIPT_NAME = "run_docker_job.sh"
    LOG_FILE_NAME = "training_workflow.log"
    REMOTE_LOG_PATH = "output/*/logs/global.json"
    REMOTE_OUTPUT_PATH = "output/"

    # Transfer Configuration
    TRANSFER_CONFIGS = [
        ("train_s3_path", "train_file_path", "train dataset"),
        ("eval_s3_path", "eval_file_path", "eval dataset"),
        ("config_s3_path", "config_file_path", "config"),
    ]

    # Instance Naming
    INSTANCE_NAME_PREFIX = "innotrain"

    # Logging Configuration
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    TRANSER_THREAD_COUNT = 5


@dataclass
class WorkflowState:
    """Holds the state of a training workflow execution."""

    job_uuid: str
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    yaml_builder: Optional["ProjectYamlBuilder"] = None


class TrainingWorkflow:
    """Orchestrates training jobs with clean separation of concerns."""

    def __init__(
        self,
        job_uuid: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        job: Optional[TrainingJob] = None,
    ):
        """Private constructor. Use create() or create_with_job() factory methods instead."""
        load_dotenv()
        # self.logger = logger or self._create_logger()
        if job_uuid:
            self.logger = get_file_logger(f"workflow_{job_uuid}")
        else:
            self.logger = get_file_logger(f"workflow_no_job_uuid")

        # Core services (always available)
        self.gpu_client = LambdaClient()
        self.file_transfer = S3ToServerTransfer(logger=self.logger)
        self.executor = ThreadPoolExecutor(
            max_workers=WorkflowConstants.TRANSER_THREAD_COUNT
        )

        # Job-specific state (reset for each job)
        self._reset_job_state()

        # Set job data
        self.job_uuid = job_uuid
        self.job = job
        if job_uuid and job:
            self.state = WorkflowState(job_uuid=job_uuid)
            if job.machine_config:
                self.state.instance_id = job.machine_config.get("instance_id", None)
                self.state.instance_ip = job.machine_config.get("instance_ip", None)
            if job.training_request:
                self.data = self._parse_request_data(job.training_request)

    @staticmethod
    async def get_job(job_uuid, session=None):
        """Get a job by UUID using SQLAlchemy."""
        from sqlalchemy import select
        from app.database import async_session_maker

        if session is None:
            async with async_session_maker() as session:
                stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
                result = await session.execute(stmt)
                return result.scalars().first()
        else:
            stmt = select(TrainingJob).where(TrainingJob.uuid == job_uuid)
            result = await session.execute(stmt)
            return result.scalars().first()

    @classmethod
    async def for_existing_job(
        cls, job_uuid: str, logger: Optional[logging.Logger] = None
    ):
        """Create workflow from an existing job in the database."""
        job = await cls.get_job(job_uuid)
        if not job:
            raise JobNotFoundError(f"Job not found: {job_uuid}")
        return cls(job_uuid=job_uuid, logger=logger, job=job)

    @classmethod
    def for_new_job(cls, logger: Optional[logging.Logger] = None):
        """Create workflow for a new job (will create job record later)."""
        return cls(job_uuid=None, logger=logger, job=None)

    def __post_init__(self):
        required_vars = ["BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def _reset_job_state(self) -> None:
        """Reset all job-specific state for a new training job."""
        self.ssh_executor: Optional[SshExecutor] = None
        self.monitor: Optional[TrainingJobMonitor] = None
        self.state: Optional[WorkflowState] = None
        self.job: Optional[TrainingJob] = None
        self.request_data: Optional[Dict[str, Any]] = None
        self.data: Optional[Dict[str, Any]] = None
        self.job_uuid: Optional[str] = None

    # ==================== Public API ====================

    async def run_complete_training(self, request_data: Dict[str, Any]) -> str:
        """Run complete training workflow from start to finish.

        This method orchestrates the complete training workflow:
        1. Initialize job record in database
        2. Build and upload configuration
        3. Provision GPU instance
        4. Establish SSH connection
        5. Transfer files (datasets, config, training script)
        6. Execute training with monitoring
        7. Download outputs and cleanup

        Args:
            request_data: Training request containing project info, datasets, and parameters

        Returns:
            str: UUID of the completed training job

        Raises:
            WorkflowError: If any step in the training workflow fails
        """
        try:
            # Initialize job and state
            if not self.job:
                self.job_uuid = await self._initialize_job(request_data)
            else:
                self.job_uuid = str(self.job.uuid)

            self.state = WorkflowState(job_uuid=self.job_uuid)
            self.logger.info(f"🚀 Starting training job {self.job_uuid}")

            # Build and upload configuration
            self.state.yaml_builder = await self._prepare_configuration()

            # Launch GPU and setup connection
            await self._provision_gpu_instance()
            await self._establish_ssh_connection()

            # Transfer files
            await self._transfer_all_files()

            # Execute training with monitoring (blocks until completion)
            await self._execute_training_with_monitoring()

            self.job.status = TrainingJobStatus.COMPLETED

            # Update with SQLAlchemy
            async with async_session_maker() as session:
                session.add(self.job)
                await session.commit()

            self.logger.info(f"✅ Training job {self.job_uuid} completed successfully")
            return self.job_uuid

        except Exception as e:
            self.logger.error(f"❌ Training job failed: {str(e)}")
            await self._handle_failure(str(e))
            raise
        finally:
            try:
                # Always cleanup resources
                await self._cleanup_resources()
                self.job.completed_at = ist_now()

                # Calculate time_taken using the helper method
                self._update_job_time_taken()

                # Update with SQLAlchemy
                async with async_session_maker() as session:
                    session.add(self.job)
                    await session.commit()
            except Exception as cleanup_error:
                self.logger.error(f"Failed to cleanup resources: {str(cleanup_error)}")
                if "e" in locals():
                    raise e from cleanup_error
                raise

    async def cancel_training(self) -> dict:
        """Cancel a running training job.

        Returns:
            dict: A dictionary with keys:
                - success: bool
                - message: str
                - status: str
        """
        try:
            # Check if job is already in a terminal state
            if self.job.status in [
                TrainingJobStatus.COMPLETED,
                TrainingJobStatus.FAILED,
                TrainingJobStatus.CANCELLED,
            ]:
                return {
                    "success": True,
                    "message": f"Job is already in terminal state: {self.job.status.value}",
                    "status": self.job.status.value,
                }

            # Check if instance exists and in what state
            instance_id = None
            if self.job.machine_config:
                instance_id = self.job.machine_config.get("instance_id")

            # If no instance, just update status
            if not instance_id:
                self.logger.info(
                    f"No instance found for job {self.job_uuid}, skipping cleanup"
                )

                # Update status
                self.job.status = TrainingJobStatus.CANCELLED
                self.job.completed_at = ist_now()

                # Calculate time_taken
                self._update_job_time_taken()

                # Update with SQLAlchemy
                async with async_session_maker() as session:
                    session.add(self.job)
                    await session.commit()

                return {
                    "success": True,
                    "message": f"Job {self.job_uuid} cancelled (no instance to clean up)",
                    "status": self.job.status.value,
                }

            # Just terminate the instance if SSH isn't established yet
            if (
                not self.ssh_executor
                or not hasattr(self.ssh_executor, "is_connected")
                or not self.ssh_executor.is_connected()
            ):
                self.logger.info(
                    f"SSH not connected for job {self.job_uuid}, terminating instance directly"
                )
                await self._terminate_instance()

                # Update status
                self.job.status = TrainingJobStatus.CANCELLED
                self.job.completed_at = ist_now()

                # Calculate time_taken
                self._update_job_time_taken()

                # Update with SQLAlchemy
                async with async_session_maker() as session:
                    session.add(self.job)
                    await session.commit()

                return {
                    "success": True,
                    "message": f"Job {self.job_uuid} cancelled (instance terminated)",
                    "status": self.job.status.value,
                }
            else:
                # Full cleanup for running instances with SSH connection
                self.logger.info(f"Performing full cleanup for job {self.job_uuid}")
                await self._cleanup_resources()

                # Update status
                self.job.status = TrainingJobStatus.CANCELLED
                self.job.completed_at = ist_now()

                # Calculate time_taken
                self._update_job_time_taken()

                # Update with SQLAlchemy
                async with async_session_maker() as session:
                    session.add(self.job)
                    await session.commit()

                self.logger.info(f"🛑 Cancelled job {self.job_uuid}")
                return {
                    "success": True,
                    "message": f"Training job {self.job_uuid} cancelled successfully",
                    "status": self.job.status.value,
                }

        except Exception as e:
            self.logger.error(f"Failed to cancel job: {str(e)}")
            return {
                "success": False,
                "message": f"An error occurred while cancelling the job: {str(e)}",
                "status": self.job.status.value if self.job else "unknown",
            }

    # ==================== Job Initialization ====================

    async def _initialize_job(self, request_data: Dict[str, Any]) -> str:
        """Create database record for new training job.

        Args:
            request_data: Training request data containing project info and parameters

        Returns:
            str: UUID of the created training job

        Raises:
            Exception: If job creation fails
        """
        from app.database import async_session_maker

        self.request_data = request_data
        self.data = self._parse_request_data(self.request_data)

        job_uuid = str(uuid.uuid4())

        async with async_session_maker() as session:
            # Create new job with SQLAlchemy
            self.job = TrainingJob(
                uuid=job_uuid,
                project_id=self.data.get("project", {}).get("id"),
                training_run_id=self.data.get("training_run_id"),
                training_request=self.request_data,
                status=TrainingJobStatus.PENDING,
                created_at=ist_now(),
            )

            session.add(self.job)
            await session.commit()
            await session.refresh(self.job)

        self.job_uuid = job_uuid

        self.logger.info(f"📝 Created job record: {self.job_uuid}")
        return self.job_uuid

    async def _prepare_configuration(self) -> ProjectYamlBuilder:
        """Build and upload project YAML configuration.

        Args:
            request_data: Training request data containing configuration parameters

        Returns:
            ProjectYamlBuilder: Configured YAML builder instance

        Raises:
            Exception: If YAML upload to S3 fails
        """

        yaml_builder = ProjectYamlBuilder()
        yaml_builder._add_data(self.data)
        self.job.project_yaml_config = yaml_builder.get_yaml_dict()

        # Update with SQLAlchemy
        async with async_session_maker() as session:
            session.add(self.job)
            await session.commit()

        if not yaml_builder.save_to_s3():
            raise Exception("Failed to upload YAML to S3")

        self.logger.info("✅ Configuration uploaded to S3")
        return yaml_builder

    # ==================== Infrastructure Setup ====================

    async def _provision_gpu_instance(self) -> None:
        """Launch GPU instance and update job record."""
        try:
            gpu_config = self.gpu_client.list_available_instances()
            if not gpu_config:
                raise InfrastructureError("No available GPU instances")

            instance_config = self.gpu_client.launch_instance(
                gpu_config["name"],
                gpu_config["region"],
                name=f"{WorkflowConstants.INSTANCE_NAME_PREFIX}-{self.state.job_uuid[:8]}",
            )

            if not instance_config:
                raise InfrastructureError("Failed to launch GPU instance")

            self.state.instance_id = instance_config["id"]
            self.state.instance_ip = instance_config["ip"]

            # Update job record
            self.job.machine_config = {
                "instance_id": self.state.instance_id,
                "instance_ip": self.state.instance_ip,
                "instance_type": gpu_config["name"],
                "region": gpu_config["region"],
            }

            # Update with SQLAlchemy
            async with async_session_maker() as session:
                session.add(self.job)
                await session.commit()

            self.logger.info(f"✅ GPU instance launched: {self.state.instance_id}")
        except Exception as e:
            if isinstance(e, InfrastructureError):
                raise
            raise InfrastructureError(f"GPU provisioning failed: {str(e)}")

    async def _establish_ssh_connection(self) -> None:
        """Setup SSH connection with retry logic using async method."""
        self.ssh_executor = SshExecutor(
            ip=self.state.instance_ip, username=WorkflowConstants.SSH_USERNAME
        )

        for attempt in range(WorkflowConstants.SSH_RETRY_ATTEMPTS):
            try:
                # Use async connection method
                await self.ssh_executor.connect_async()
                self.logger.info("✅ SSH connection established")
                return
            except Exception as e:
                if attempt == WorkflowConstants.SSH_RETRY_ATTEMPTS - 1:
                    raise InfrastructureError(
                        f"SSH connection failed after {WorkflowConstants.SSH_RETRY_ATTEMPTS} attempts: {str(e)}"
                    )

                wait_time = WorkflowConstants.SSH_RETRY_BASE_DELAY**attempt
                self.logger.info(
                    f"SSH retry {attempt + 1}/{WorkflowConstants.SSH_RETRY_ATTEMPTS} in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    # ==================== File Transfer ====================

    async def _transfer_all_files(self) -> None:
        """Transfer all required files to GPU server."""
        s3_bucket = self.job_s3_bucket

        # Create transfer tasks for parallel execution
        transfer_tasks = [
            self._transfer_file(
                s3_bucket,
                self.state.yaml_builder.yaml_data[s3_key],
                self.state.yaml_builder.yaml_data[path_key],
                description,
            )
            for s3_key, path_key, description in WorkflowConstants.TRANSFER_CONFIGS
        ]

        # Execute transfers in parallel
        await asyncio.gather(*transfer_tasks)

        # Upload training script
        await self._upload_training_script()

    async def _transfer_file(
        self, s3_bucket: str, s3_prefix: str, server_path: str, description: str
    ) -> None:
        """Transfer a single file from S3 to server."""
        try:
            await asyncio.to_thread(
                self.file_transfer.transfer_file_to_server,
                s3_bucket,
                s3_prefix,
                self.state.instance_ip,
                server_path,
            )
            self.logger.info(f"📤 Transferred {description}")
        except Exception as e:
            raise FileTransferError(f"Failed to transfer {description}: {str(e)}")

    async def _upload_training_script(self) -> None:
        """Upload the training script to the GPU server."""
        try:
            script_path = WorkflowConstants.TRAINING_SCRIPT_NAME
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Training script not found: {script_path}")

            await asyncio.to_thread(
                self.ssh_executor.upload_file, script_path, script_path
            )
            self.logger.info(f"📤 Uploaded training script: {script_path}")
        except Exception as e:
            raise FileTransferError(f"Failed to upload training script: {str(e)}")

    async def _stop_monitoring(self) -> None:
        """Stop monitoring the training job."""
        if self.monitor:
            await self.monitor.stop_monitoring()
            self.logger.info("🛑 Stopped monitoring")
        else:
            self.logger.info("No monitor to stop")

    # ==================== Training Execution ====================

    async def _execute_training_with_monitoring(self) -> None:
        """Execute training with monitoring until completion."""
        try:
            # Start monitoring
            self.monitor = TrainingJobMonitor(
                training_job_uuid=self.state.job_uuid,
                ssh_executor=self.ssh_executor,
                remote_log_path=WorkflowConstants.REMOTE_LOG_PATH,
                poll_interval=WorkflowConstants.MONITORING_POLL_INTERVAL,
                logger=self.logger,
            )

            # Start monitoring and training concurrently
            monitor_task = asyncio.create_task(self.monitor.start_monitoring())
            training_task = asyncio.create_task(
                asyncio.to_thread(
                    lambda: self.ssh_executor.execute_script(
                        WorkflowConstants.TRAINING_SCRIPT_NAME, check=True
                    )
                )
            )

            self.job.status = TrainingJobStatus.RUNNING
            self.job.started_at = ist_now()

            # Update with SQLAlchemy
            async with async_session_maker() as session:
                session.add(self.job)
                await session.commit()
            self.logger.info(
                f"🚀 Training job {self.job_uuid} script execution started on GPU"
            )

            # Wait for both tasks to complete
            await asyncio.gather(monitor_task, training_task)

            self.logger.info("✅ Training and monitoring completed")

        except Exception as e:
            # Stop monitoring if it's still running
            await self._cleanup_resources()
            raise TrainingExecutionError(f"Training execution failed: {str(e)}")

    async def _download_outputs(self) -> bool:
        """Download training outputs from GPU server to S3."""
        self.logger.info(f"📊 Starting output download for job {self.state.job_uuid}")

        try:
            # Get output paths
            s3_bucket = self.job_s3_bucket
            s3_path = self.job_s3_path
            # Transfer to S3
            self.logger.info("🚀 Starting file transfer to S3...")

            await asyncio.to_thread(
                self.file_transfer.transfer_files_to_s3,
                self.state.instance_ip,
                WorkflowConstants.REMOTE_OUTPUT_PATH,
                s3_bucket,
                s3_path,
                recursive=True,
            )

            self.logger.info(f"✅ Outputs transferred to S3: {s3_path}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to download outputs: {str(e)}", exc_info=True)
            raise FileTransferError(f"Output download failed: {str(e)}")

    # ==================== Cleanup & Error Handling ====================

    async def _handle_failure(self, error_message: str) -> None:
        """Handle job failure with cleanup."""
        if not self.state:
            return

        # Mark job as failed
        try:
            await self._cleanup_resources()
            self.job.status = TrainingJobStatus.FAILED
            self.job.completed_at = ist_now()

            # Calculate time_taken
            self._update_job_time_taken()

            # Update with SQLAlchemy
            async with async_session_maker() as session:
                session.add(self.job)
                await session.commit()

        except Exception as e:
            self.logger.error(f"Failed to mark job as failed: {str(e)}")

    async def _cleanup_resources(self) -> None:
        """Cleanup all resources including SSH connections and GPU instances."""
        self.logger.info("🧹 Starting resource cleanup")

        # Stop monitoring
        await self._stop_monitoring()

        # Only attempt to download outputs if SSH connection exists and is healthy
        if self.ssh_executor:
            try:
                # Test SSH connection with a simple command
                test_result = await asyncio.to_thread(
                    self.ssh_executor.execute_command, "echo 'test'", check=False
                )

                if test_result.success:
                    try:
                        await self._download_outputs()
                    except Exception as e:
                        self.logger.error(
                            f"Failed to download outputs during cleanup: {e}"
                        )
                else:
                    self.logger.info(
                        "Skipping output download - SSH connection unhealthy"
                    )
            except Exception as e:
                self.logger.info(f"Skipping output download - SSH error: {str(e)}")
        else:
            self.logger.info("Skipping output download - No SSH executor")

        # Cleanup SSH connection
        await self.close_ssh_connection()

        # Cleanup GPU instance
        await self._terminate_instance()
        self.logger.info("✅ Resource cleanup completed")

    @property
    def job_s3_bucket(self):
        s3_bucket = os.getenv("BUCKET_NAME")
        if not s3_bucket:
            raise ValueError("No S3 bucket configured")
        return s3_bucket

    @property
    def job_s3_path(self):
        if not self.job.project_id and self.job.training_run_id:
            raise ValueError("No project_id or training_run_id configured")
        return f"media/projects/{self.job.project_id}/{self.job.training_run_id}"

    @property
    def log_file_path(self):
        return f"logs/workflow_{self.job_uuid}.log"

    async def close_ssh_connection(self) -> None:
        if self.ssh_executor:
            try:
                # Set a timeout for the disconnect operation
                disconnect_task = asyncio.create_task(
                    asyncio.to_thread(self.ssh_executor.disconnect)
                )
                try:
                    await asyncio.wait_for(disconnect_task, timeout=5.0)
                    self.logger.info("✅ SSH connection closed")
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "SSH disconnect timed out after 5s, forcing cleanup"
                    )
                finally:
                    # Ensure we clear the reference even if disconnect fails
                    self.ssh_executor = None
            except Exception as e:
                self.logger.error(f"Error closing SSH connection: {e}")
                # Still clear the reference
                self.ssh_executor = None

    async def _terminate_instance(self) -> None:
        """Terminate the GPU instance if it exists."""
        if not self.state or not self.state.instance_id:
            return

        try:
            if await asyncio.to_thread(
                self.gpu_client.terminate_instance, self.state.instance_id
            ):
                self.logger.info(f"✅ Instance {self.state.instance_id} terminated")
            else:
                self.logger.warning(
                    f"⚠️ Failed to terminate instance {self.state.instance_id}"
                )
        except Exception as e:
            self.logger.error(f"Error terminating instance: {e}")
            raise

    # ==================== Utilities ====================

    def transfer_progress_callback(self, transferred, total):
        percent = (transferred / total) * 100
        self.logger.info(f"Transfer progress: {percent:.1f}%")

    def _parse_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse nested request data structure."""
        return request_data.get("data", {}).get("request_data", request_data)

    def _update_job_time_taken(self) -> None:
        """Calculate and update time_taken field based on created_at and completed_at."""
        if self.job and self.job.created_at and self.job.completed_at:
            self.job.time_taken = calculate_duration(
                self.job.created_at, self.job.completed_at
            )
            if self.job.time_taken is not None:
                self.logger.info(f"Job duration: {self.job.time_taken:.2f} minutes")

    def _create_logger(self) -> logging.Logger:
        """Create logger with console and file handlers."""
        logger = logging.getLogger("TrainingWorkflow")

        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(WorkflowConstants.LOG_FORMAT)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(WorkflowConstants.LOG_FILE_NAME)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Configure root logger for library logs
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            root_file_handler = logging.FileHandler(WorkflowConstants.LOG_FILE_NAME)
            root_file_handler.setFormatter(formatter)
            root_file_handler.setLevel(logging.INFO)
            root_logger.addHandler(root_file_handler)
            root_logger.setLevel(logging.INFO)

        # Configure library loggers
        for logger_name in ["paramiko", "tortoise", "scripts.s3_to_server_transfer"]:
            lib_logger = logging.getLogger(logger_name)
            lib_logger.setLevel(logging.INFO)
            lib_logger.propagate = True

        return logger
