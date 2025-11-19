"""TrainingWorkflow - Clean, refactored training job orchestration."""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass

from dotenv import load_dotenv

from scripts.lambda_client import LambdaClient
from scripts.project_yaml_builder import ProjectYamlBuilder
from scripts.s3_to_server_transfer import S3ToServerTransfer
from scripts.ssh_executor import SshExecutor
from app.services.training_job_monitor import TrainingJobMonitor
from models.training_job import TrainingJob, TrainingJobStatus


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


@dataclass
class WorkflowState:
    """Holds the state of a training workflow execution."""

    job_uuid: str
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    yaml_builder: Optional["ProjectYamlBuilder"] = None


class TrainingWorkflow:
    """Orchestrates training jobs with clean separation of concerns."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize workflow with required clients."""
        load_dotenv()
        self.logger = logger or self._create_logger()

        # Core services (always available)
        self.lambda_client = LambdaClient()
        self.file_transfer = S3ToServerTransfer(logger=self.logger)

        # Job-specific state (reset for each job)
        self._reset_job_state()

    def _reset_job_state(self) -> None:
        """Reset all job-specific state for a new training job."""
        self.ssh_executor: Optional[SshExecutor] = None
        self.monitor: Optional[TrainingJobMonitor] = None
        self.state: Optional[WorkflowState] = None
        self.job: Optional[TrainingJob] = None

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
                job_uuid = await self._initialize_job(request_data)
            else:
                job_uuid = str(self.job.uuid)

            self.state = WorkflowState(job_uuid=job_uuid)
            self.logger.info(f"🚀 Starting training job {job_uuid}")

            # Build and upload configuration
            self.state.yaml_builder = await self._prepare_configuration(request_data)

            # Launch GPU and setup connection
            await self._provision_gpu_instance()
            await self._establish_ssh_connection()

            # Transfer files
            await self._transfer_all_files()

            # Execute training with monitoring (blocks until completion)
            await self._execute_training_with_monitoring()

            # Download outputs
            await self._download_outputs()

            self.logger.info(f"✅ Training job {job_uuid} completed successfully")
            return job_uuid

        except Exception as e:
            self.logger.error(f"❌ Training job failed: {str(e)}")
            await self._handle_failure(str(e))
            raise
        finally:
            # Always cleanup resources
            await self._cleanup_resources()

    async def get_job_status(self, job_uuid: str) -> TrainingJobStatus:
        """Get the current status of a training job."""
        try:
            job = await TrainingJob.get(uuid=job_uuid)
            return job.status
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            raise

    async def cancel_training(self, job_uuid: str) -> bool:
        """Cancel a running training job."""
        try:
            job = await TrainingJob.get(uuid=job_uuid)

            if job.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED]:
                return False

            # Stop monitoring
            if self.monitor:
                await self.monitor.stop_monitoring()

            # Terminate instance
            machine_config = job.machine_config or {}
            if instance_id := machine_config.get("instance_id"):
                await self._cleanup_instance(instance_id)

            # Update status
            job.status = TrainingJobStatus.CANCELLED
            job.completed_at = datetime.now()
            await job.save()

            self.logger.info(f"🛑 Cancelled job {job_uuid}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel job: {str(e)}")
            return False

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
        data = self._parse_request_data(request_data)

        self.job = await TrainingJob.create(
            uuid=uuid.uuid4(),
            project_id=data.get("project", {}).get("id"),
            training_run_id=data.get("training_run_id"),
            training_request=request_data,
            status=TrainingJobStatus.PENDING,
        )

        self.logger.info(f"📝 Created job record: {self.job.uuid}")
        return str(self.job.uuid)

    async def _prepare_configuration(
        self, request_data: Dict[str, Any]
    ) -> ProjectYamlBuilder:
        """Build and upload project YAML configuration.

        Args:
            request_data: Training request data containing configuration parameters

        Returns:
            ProjectYamlBuilder: Configured YAML builder instance

        Raises:
            Exception: If YAML upload to S3 fails
        """
        data = self._parse_request_data(request_data)

        yaml_builder = ProjectYamlBuilder()
        yaml_builder._add_data(data)
        self.job.project_yaml_config = yaml_builder.get_yaml_dict()
        await self.job.save()

        if not yaml_builder.save_to_s3():
            raise Exception("Failed to upload YAML to S3")

        self.logger.info("✅ Configuration uploaded to S3")
        return yaml_builder

    # ==================== Infrastructure Setup ====================

    async def _provision_gpu_instance(self) -> None:
        """Launch GPU instance and update job record."""
        try:
            gpu_config = self.lambda_client.list_available_instances()
            if not gpu_config:
                raise InfrastructureError(
                    "No available GPU instances", job_uuid=self.state.job_uuid
                )

            instance_config = self.lambda_client.launch_instance(
                gpu_config["name"],
                gpu_config["region"],
                name=f"{WorkflowConstants.INSTANCE_NAME_PREFIX}-{self.state.job_uuid[:8]}",
            )

            if not instance_config:
                raise InfrastructureError(
                    "Failed to launch GPU instance", job_uuid=self.state.job_uuid
                )

            self.state.instance_id = instance_config["id"]
            self.state.instance_ip = instance_config["ip"]

            # Update job record
            job = await TrainingJob.get(uuid=self.state.job_uuid)
            job.machine_config = {
                "instance_id": self.state.instance_id,
                "instance_ip": self.state.instance_ip,
                "instance_type": gpu_config["name"],
                "region": gpu_config["region"],
            }
            await job.save()

            self.logger.info(f"✅ GPU instance launched: {self.state.instance_id}")
        except Exception as e:
            if isinstance(e, InfrastructureError):
                raise
            raise InfrastructureError(
                f"GPU provisioning failed: {str(e)}", job_uuid=self.state.job_uuid
            )

    async def _establish_ssh_connection(self) -> None:
        """Setup SSH connection with retry logic."""
        self.ssh_executor = SshExecutor(
            ip=self.state.instance_ip, username=WorkflowConstants.SSH_USERNAME
        )

        for attempt in range(WorkflowConstants.SSH_RETRY_ATTEMPTS):
            try:
                await asyncio.to_thread(self.ssh_executor.connect)
                self.logger.info("✅ SSH connection established")
                return
            except Exception as e:
                if attempt == WorkflowConstants.SSH_RETRY_ATTEMPTS - 1:
                    raise InfrastructureError(
                        f"SSH connection failed after {WorkflowConstants.SSH_RETRY_ATTEMPTS} attempts: {str(e)}",
                        job_uuid=self.state.job_uuid,
                    )

                wait_time = WorkflowConstants.SSH_RETRY_BASE_DELAY**attempt
                self.logger.info(
                    f"SSH retry {attempt + 1}/{WorkflowConstants.SSH_RETRY_ATTEMPTS} in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    # ==================== File Transfer ====================

    async def _transfer_all_files(self) -> None:
        """Transfer all required files to GPU server."""
        s3_bucket = os.getenv("BUCKET_NAME")
        if not s3_bucket:
            raise FileTransferError(
                "S3 bucket not configured", job_uuid=self.state.job_uuid
            )

        # Transfer datasets and config using constants
        for s3_key, path_key, description in WorkflowConstants.TRANSFER_CONFIGS:
            await self._transfer_file(
                s3_bucket,
                self.state.yaml_builder.yaml_data[s3_key],
                self.state.yaml_builder.yaml_data[path_key],
                description,
            )

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
            raise FileTransferError(
                f"Failed to transfer {description}: {str(e)}",
                job_uuid=self.state.job_uuid,
            )

    async def _upload_training_script(self) -> None:
        """Upload the training script to the GPU server."""
        try:
            script_path = WorkflowConstants.TRAINING_SCRIPT_NAME
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Training script not found: {script_path}")

            await asyncio.to_thread(
                self.ssh_executor.upload_file,
                script_path,
                script_path,  # Upload to same name on remote
            )
            self.logger.info(f"📤 Uploaded training script: {script_path}")
        except Exception as e:
            raise FileTransferError(
                f"Failed to upload training script: {str(e)}",
                job_uuid=self.state.job_uuid,
            )

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

            # Wait for both tasks to complete
            await asyncio.gather(monitor_task, training_task)

            self.logger.info("✅ Training and monitoring completed")

        except Exception as e:
            # Stop monitoring if it's still running
            if self.monitor:
                await self.monitor.stop_monitoring()
            raise TrainingExecutionError(
                f"Training execution failed: {str(e)}", job_uuid=self.state.job_uuid
            )

    async def _download_outputs(self) -> bool:
        """Download training outputs from GPU server to S3."""
        self.logger.info(f"📊 Starting output download for job {self.state.job_uuid}")

        try:
            job = await TrainingJob.get(uuid=self.state.job_uuid)
            machine_config = job.machine_config or {}
            self.logger.info(f"Machine config: {machine_config}")

            if not (instance_ip := machine_config.get("instance_ip")):
                self.logger.warning("No instance IP, skipping output download")
                return False

            # Get output paths
            request_data = job.training_request or {}
            data = self._parse_request_data(request_data)
            project_id = data.get("project", {}).get("id")
            training_run_id = data.get("training_run_id")

            if not project_id or not training_run_id:
                self.logger.warning(
                    "Missing project_id or training_run_id, skipping output download"
                )
                return False

            s3_bucket = os.getenv("BUCKET_NAME")
            if not s3_bucket:
                self.logger.warning("No S3 bucket configured, skipping output download")
                return False

            s3_path = f"media/projects/{project_id}/{training_run_id}"
            self.logger.info(
                f"📎 Transfer params: {instance_ip}:output/ -> s3://{s3_bucket}/{s3_path}"
            )

            # Transfer to S3
            self.logger.info("🚀 Starting file transfer to S3...")
            await asyncio.to_thread(
                self.file_transfer.transfer_files_to_s3,
                instance_ip,
                WorkflowConstants.REMOTE_OUTPUT_PATH,
                s3_bucket,
                s3_path,
                recursive=True,
            )
            self.logger.info(f"✅ Outputs transferred to S3: {s3_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download outputs: {str(e)}", exc_info=True)
            raise FileTransferError(
                f"Output download failed: {str(e)}", job_uuid=self.state.job_uuid
            )

    # ==================== Cleanup & Error Handling ====================

    async def _handle_failure(self, error_message: str) -> None:
        """Handle job failure with cleanup."""
        if not self.state:
            return

        # Mark job as failed
        try:
            job = await TrainingJob.get(uuid=self.state.job_uuid)
            job.status = TrainingJobStatus.FAILED
            job.completed_at = datetime.now()
            await job.save()

            if self.monitor:
                await self.monitor.stop_monitoring()
        except Exception as e:
            self.logger.error(f"Failed to mark job as failed: {str(e)}")

        # Cleanup instance
        if self.state.instance_id:
            await self._cleanup_instance(self.state.instance_id)

    async def _cleanup_resources(self) -> None:
        """Cleanup all resources including SSH connections and GPU instances."""
        self.logger.info("🧹 Starting resource cleanup")

        # Stop monitoring
        if self.monitor:
            try:
                self.logger.info("🛑 Stopping monitoring")
                await self.monitor.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping monitor: {e}")

        # Cleanup SSH connection
        if self.ssh_executor:
            try:
                await asyncio.to_thread(self.ssh_executor.disconnect)
                self.logger.info("✅ SSH connection closed")
            except Exception as e:
                self.logger.error(f"Error closing SSH connection: {e}")

        # Cleanup GPU instance
        if self.state and self.state.instance_id:
            try:
                if self.lambda_client.terminate_instance(self.state.instance_id):
                    self.logger.info(f"✅ Instance {self.state.instance_id} terminated")
                else:
                    self.logger.warning(
                        f"⚠️ Failed to terminate instance {self.state.instance_id}"
                    )
            except Exception as e:
                self.logger.error(f"Error terminating instance: {e}")

        self.logger.info("✅ Resource cleanup completed")

    # ==================== Utilities ====================

    def _parse_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse nested request data structure."""
        return request_data.get("data", {}).get("request_data", request_data)

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
