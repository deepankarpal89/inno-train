import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import tempfile
import os

from scripts.lambda_client import LambdaClient
from scripts.ssh_executor import SshExecutor, CommandResult


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    LAUNCHING_INSTANCE = "launching_instance"
    CONNECTING = "connecting"
    TRANSFERRING_FILES = "transferring_files"
    EXECUTING = "executing"
    DOWNLOADING_OUTPUTS = "downloading_outputs"
    COMPLETED = "completed"
    FAILED = "failed"
    CLEANING_UP = "cleaning_up"


@dataclass
class JobConfig:
    """Configuration for a training job."""

    job_id: str
    docker_image: str
    docker_username: str = None
    docker_password: str = None
    files_to_transfer: List[Dict[str, str]] = (
        None  # [{"local": "path", "remote": "path"}]
    )
    environment_vars: Dict[str, str] = None
    volume_mappings: List[Dict[str, str]] = (
        None  # [{"host": "path", "container": "path"}]
    )
    output_files: List[str] = None  # Remote paths to download after execution
    timeout_minutes: int = 60
    instance_type: str = None  # If None, will auto-select cheapest available
    region: str = None  # If None, will auto-select


@dataclass
class JobResult:
    """Result of a training job execution."""

    job_id: str
    status: JobStatus
    instance_id: str = None
    instance_ip: str = None
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = None
    execution_result: CommandResult = None
    error_message: str = None
    cost_estimate: float = None
    logs: List[str] = None


class TrainingOrchestrator:
    """Orchestrates the complete training workflow: GPU provisioning -> File transfer -> Docker execution."""

    def __init__(
        self, lambda_client: LambdaClient = None, logger: logging.Logger = None
    ):
        """Initialize the orchestrator."""
        self.lambda_client = lambda_client or LambdaClient()
        self.logger = logger or self._setup_logger()
        self.ssh_executor: Optional[SshExecutor] = None
        self.current_instance = None
        self.jobs: Dict[str, JobResult] = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the orchestrator."""
        logger = logging.getLogger("TrainingOrchestrator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def run_training_job(self, job_config: JobConfig) -> JobResult:
        """Execute a complete training job workflow."""
        job_result = JobResult(
            job_id=job_config.job_id,
            status=JobStatus.PENDING,
            start_time=datetime.now(),
            logs=[],
        )

        self.jobs[job_config.job_id] = job_result

        try:
            self.logger.info(f"🚀 Starting training job {job_config.job_id}")

            # Step 1: Launch GPU instance
            await self._launch_instance(job_config, job_result)

            # Step 2: Setup SSH connection
            await self._setup_ssh_connection(job_result)

            # Step 3: Transfer files
            await self._transfer_files(job_config, job_result)

            # Step 4: Execute training
            await self._execute_training(job_config, job_result)

            # Step 5: Download outputs
            await self._download_outputs(job_config, job_result)

            # Mark as completed
            job_result.status = JobStatus.COMPLETED
            job_result.end_time = datetime.now()
            job_result.duration_seconds = (
                job_result.end_time - job_result.start_time
            ).total_seconds()

            self.logger.info(
                f"✅ Job {job_config.job_id} completed successfully in {job_result.duration_seconds:.1f}s"
            )

        except Exception as e:
            await self._handle_failure(job_config, job_result, e)
        finally:
            await self._cleanup(job_result)

        return job_result

    async def _launch_instance(self, job_config: JobConfig, job_result: JobResult):
        """Launch a GPU instance."""
        job_result.status = JobStatus.LAUNCHING_INSTANCE
        self.logger.info(f"🔧 Launching GPU instance for job {job_config.job_id}")

        try:
            # Auto-select instance type and region if not specified
            if not job_config.instance_type or not job_config.region:
                available = self.lambda_client.list_available_instances()
                instance_type = job_config.instance_type or available["name"]
                region = job_config.region or available["region"]
            else:
                instance_type = job_config.instance_type
                region = job_config.region

            # Launch instance (this blocks until active)
            self.current_instance = self.lambda_client.launch_instance(
                instance_type_name=instance_type,
                region_name=region,
                name=f"innotrain-{job_config.job_id}",
            )

            if not self.current_instance:
                raise Exception("Failed to launch instance")

            job_result.instance_id = self.current_instance["id"]
            job_result.instance_ip = self.current_instance["ip"]

            self.logger.info(
                f"✅ Instance {job_result.instance_id} launched at {job_result.instance_ip}"
            )
            job_result.logs.append(
                f"Instance launched: {job_result.instance_id} at {job_result.instance_ip}"
            )

        except Exception as e:
            raise Exception(f"Failed to launch instance: {str(e)}")

    async def _setup_ssh_connection(self, job_result: JobResult):
        """Setup SSH connection to the instance."""
        job_result.status = JobStatus.CONNECTING
        self.logger.info(f"🔗 Setting up SSH connection to {job_result.instance_ip}")

        try:
            self.ssh_executor = SshExecutor(job_result.instance_ip)

            # Retry connection with backoff
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.ssh_executor.connect()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = 2**attempt
                    self.logger.info(
                        f"SSH connection attempt {attempt + 1} failed, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

            self.logger.info("✅ SSH connection established")
            job_result.logs.append("SSH connection established")

        except Exception as e:
            raise Exception(f"Failed to setup SSH connection: {str(e)}")

    async def _transfer_files(self, job_config: JobConfig, job_result: JobResult):
        """Transfer required files to the instance."""
        job_result.status = JobStatus.TRANSFERRING_FILES

        files_to_transfer = job_config.files_to_transfer or []

        # Always transfer the Docker execution script
        docker_script = self._generate_docker_script(job_config)
        files_to_transfer.append(
            {"local": docker_script, "remote": "run_docker_job.sh"}
        )

        if not files_to_transfer:
            self.logger.info("📁 No files to transfer")
            return

        self.logger.info(f"📁 Transferring {len(files_to_transfer)} files")

        try:
            for file_mapping in files_to_transfer:
                local_path = file_mapping["local"]
                remote_path = file_mapping["remote"]

                self.logger.info(f"📤 Uploading {local_path} -> {remote_path}")
                self.ssh_executor.upload_file(local_path, remote_path)
                job_result.logs.append(f"Uploaded: {local_path} -> {remote_path}")

            self.logger.info("✅ File transfer completed")

        except Exception as e:
            raise Exception(f"Failed to transfer files: {str(e)}")
        finally:
            # Clean up temporary docker script
            if docker_script and os.path.exists(docker_script):
                os.unlink(docker_script)

    async def _execute_training(self, job_config: JobConfig, job_result: JobResult):
        """Execute the training job."""
        job_result.status = JobStatus.EXECUTING
        self.logger.info(f"🏃 Executing training job {job_config.job_id}")

        try:
            # Set environment variables if provided
            if job_config.environment_vars:
                for key, value in job_config.environment_vars.items():
                    self.ssh_executor.execute_command(f'export {key}="{value}"')

            # Execute the Docker script with timeout
            timeout_seconds = job_config.timeout_minutes * 60

            # Run the script asynchronously with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.ssh_executor.execute_script, "run_docker_job.sh", check=True
                ),
                timeout=timeout_seconds,
            )

            job_result.execution_result = result
            self.logger.info("✅ Training execution completed successfully")
            job_result.logs.append("Training execution completed")

        except asyncio.TimeoutError:
            raise Exception(
                f"Training job timed out after {job_config.timeout_minutes} minutes"
            )
        except Exception as e:
            raise Exception(f"Training execution failed: {str(e)}")

    async def _download_outputs(self, job_config: JobConfig, job_result: JobResult):
        """Download output files from the instance."""
        job_result.status = JobStatus.DOWNLOADING_OUTPUTS

        output_files = job_config.output_files or ["output/execution.log"]

        if not output_files:
            self.logger.info("📥 No output files to download")
            return

        self.logger.info(f"📥 Downloading {len(output_files)} output files")

        try:
            # Create local output directory
            local_output_dir = f"outputs/{job_config.job_id}"
            os.makedirs(local_output_dir, exist_ok=True)

            for remote_file in output_files:
                try:
                    local_file = os.path.join(
                        local_output_dir, os.path.basename(remote_file)
                    )
                    self.ssh_executor.download_file(remote_file, local_file)
                    self.logger.info(f"📥 Downloaded {remote_file} -> {local_file}")
                    job_result.logs.append(f"Downloaded: {remote_file} -> {local_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {remote_file}: {str(e)}")

            self.logger.info("✅ Output download completed")

        except Exception as e:
            # Don't fail the job if output download fails
            self.logger.warning(f"Output download failed: {str(e)}")
            job_result.logs.append(f"Output download warning: {str(e)}")

    def _generate_docker_script(self, job_config: JobConfig) -> str:
        """Generate a dynamic Docker execution script based on job config."""
        script_content = f"""#!/bin/bash

# Auto-generated Docker execution script for job {job_config.job_id}
# Generated at {datetime.now().isoformat()}

set -e  # Exit on any error

IMAGE_NAME="{job_config.docker_image}"
OUTPUT_DIR="output"
LOG_FILE="${{OUTPUT_DIR}}/execution.log"
CONTAINER_NAME="innotrain-{job_config.job_id}"

# Create output directory
mkdir -p "${{OUTPUT_DIR}}"

echo "[$(date)] Starting Docker job {job_config.job_id}..." | tee -a "${{LOG_FILE}}"

# Docker login if credentials provided
"""

        if job_config.docker_username and job_config.docker_password:
            script_content += f"""
echo "[$(date)] Logging into Docker Hub..." | tee -a "${{LOG_FILE}}"
echo "{job_config.docker_password}" | sudo docker login -u "{job_config.docker_username}" --password-stdin 2>&1 | tee -a "${{LOG_FILE}}"
"""

        script_content += f"""
# Pull the Docker image
echo "[$(date)] Pulling Docker image: ${{IMAGE_NAME}}" | tee -a "${{LOG_FILE}}"
sudo docker pull "${{IMAGE_NAME}}" 2>&1 | tee -a "${{LOG_FILE}}"

# Prepare volume mappings
VOLUME_ARGS=""
"""

        # Add volume mappings
        default_volumes = [
            {"host": "$(pwd)/data", "container": "/app/data"},
            {"host": "$(pwd)/project_yaml", "container": "/app/project_yaml"},
            {"host": "$(pwd)/output", "container": "/app/output"},
        ]

        volumes = job_config.volume_mappings or default_volumes

        for volume in volumes:
            script_content += f'VOLUME_ARGS="$VOLUME_ARGS -v {volume["host"]}:{volume["container"]}"\n'

        # Add environment variables
        if job_config.environment_vars:
            script_content += '\n# Environment variables\nENV_ARGS=""\n'
            for key, value in job_config.environment_vars.items():
                script_content += f'ENV_ARGS="$ENV_ARGS -e {key}={value}"\n'
        else:
            script_content += 'ENV_ARGS=""\n'

        script_content += f"""
# Run the Docker container
echo "[$(date)] Starting container..." | tee -a "${{LOG_FILE}}"
sudo docker run --name "${{CONTAINER_NAME}}" \\
    --rm \\
    $VOLUME_ARGS \\
    $ENV_ARGS \\
    "${{IMAGE_NAME}}" 2>&1 | tee -a "${{LOG_FILE}}"

# Check execution result
if [ ${{PIPESTATUS[0]}} -eq 0 ]; then
    echo "[$(date)] Container executed successfully" | tee -a "${{LOG_FILE}}"
else
    echo "[$(date)] Error: Container execution failed" | tee -a "${{LOG_FILE}}"
    exit 1
fi

echo "[$(date)] Job {job_config.job_id} completed successfully!" | tee -a "${{LOG_FILE}}"
"""

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            return f.name

    async def _handle_failure(
        self, job_config: JobConfig, job_result: JobResult, error: Exception
    ):
        """Handle job failure."""
        job_result.status = JobStatus.FAILED
        job_result.error_message = str(error)
        job_result.end_time = datetime.now()

        if job_result.start_time:
            job_result.duration_seconds = (
                job_result.end_time - job_result.start_time
            ).total_seconds()

        self.logger.error(f"❌ Job {job_config.job_id} failed: {error}")
        job_result.logs.append(f"Job failed: {str(error)}")

    async def _cleanup(self, job_result: JobResult):
        """Clean up resources after job completion or failure."""
        job_result.status = JobStatus.CLEANING_UP
        self.logger.info(f"🧹 Cleaning up resources for job {job_result.job_id}")

        try:
            # Close SSH connection
            if self.ssh_executor:
                self.ssh_executor.disconnect()
                self.ssh_executor = None

            # Terminate instance
            if self.current_instance and job_result.instance_id:
                success = self.lambda_client.terminate_instance(job_result.instance_id)
                if success:
                    self.logger.info(f"✅ Instance {job_result.instance_id} terminated")
                    job_result.logs.append(
                        f"Instance {job_result.instance_id} terminated"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Failed to terminate instance {job_result.instance_id}"
                    )

            self.current_instance = None

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get the status of a specific job."""
        return self.jobs.get(job_id)

    def list_jobs(self) -> Dict[str, JobResult]:
        """List all jobs and their statuses."""
        return self.jobs.copy()

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job_result = self.jobs.get(job_id)
        if not job_result:
            return False

        if job_result.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False

        try:
            self.logger.info(f"🛑 Cancelling job {job_id}")
            await self._cleanup(job_result)
            job_result.status = JobStatus.FAILED
            job_result.error_message = "Job cancelled by user"
            job_result.end_time = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False
