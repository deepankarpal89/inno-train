#!/usr/bin/env python3
"""
Test TrainingWorkflow file transfer functionality using a YAML configuration.
This script uses the actual TrainingWorkflow class to test file transfers to a GPU instance.
"""

import asyncio
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.services.training_workflow import TrainingWorkflow, WorkflowState
from scripts.project_yaml_builder import ProjectYamlBuilder
from scripts.ssh_executor import SshExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkflowFileTransferTester:
    """Test TrainingWorkflow file transfer functionality."""

    def __init__(self, gpu_ip: str, yaml_file_path: str):
        """
        Initialize the workflow file transfer tester.

        Args:
            gpu_ip: IP address of the GPU instance
            yaml_file_path: Path to the YAML configuration file
        """
        self.gpu_ip = gpu_ip
        self.yaml_file_path = yaml_file_path
        self.workflow = None
        self.yaml_data = None

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        logger.info(f"📄 Loading YAML configuration from: {self.yaml_file_path}")

        if not os.path.exists(self.yaml_file_path):
            raise FileNotFoundError(f"YAML file not found: {self.yaml_file_path}")

        with open(self.yaml_file_path, "r") as f:
            self.yaml_data = yaml.safe_load(f)

        logger.info(f"✅ YAML loaded successfully")
        logger.info(f"   Project: {self.yaml_data.get('project_name')}")
        logger.info(f"   Task Type: {self.yaml_data.get('project_task_type')}")
        logger.info(f"   Training Run ID: {self.yaml_data.get('training_run_id')}")

        return self.yaml_data

    async def setup_workflow(self) -> None:
        """Setup TrainingWorkflow instance with test configuration."""
        logger.info("🔧 Setting up TrainingWorkflow instance...")

        # Ensure BUCKET_NAME is set (required by job_s3_bucket property)
        if not os.getenv("BUCKET_NAME"):
            os.environ["BUCKET_NAME"] = "innotone-training-data"
            logger.info("ℹ️  Set BUCKET_NAME to default: innotone-training-data")

        # Create workflow instance without a job (for testing)
        self.workflow = TrainingWorkflow.for_new_job(logger=logger)

        # Create a mock state with the GPU IP
        job_uuid = self.yaml_data.get("training_run_id", "test-transfer")
        self.workflow.state = WorkflowState(
            job_uuid=job_uuid,
            instance_ip=self.gpu_ip,
            instance_id=f"test-instance-{job_uuid[:8]}",
        )

        # Create a ProjectYamlBuilder and populate it with the loaded YAML data
        yaml_builder = ProjectYamlBuilder(logger=logger)
        yaml_builder.yaml_data = self.yaml_data
        self.workflow.state.yaml_builder = yaml_builder

        logger.info(f"✅ Workflow configured")
        logger.info(f"   GPU IP: {self.gpu_ip}")
        logger.info(f"   S3 Bucket: {self.workflow.job_s3_bucket}")
        logger.info(f"   Job UUID: {job_uuid}")

    async def establish_ssh_connection(self) -> None:
        """Establish SSH connection using workflow's method."""
        logger.info("🔌 Establishing SSH connection to GPU...")

        # Use the workflow's SSH connection method
        await self.workflow._establish_ssh_connection()

        logger.info("✅ SSH connection established via workflow")

    async def upload_yaml_as_config(self) -> None:
        """Upload the local YAML file to GPU instance as config.yaml."""
        logger.info("📤 Uploading local YAML file as config.yaml to GPU...")

        ssh_executor = self.workflow.ssh_executor

        # Create ddp_rlhf_classify/projects_yaml directory on GPU
        await asyncio.to_thread(
            ssh_executor.execute_command,
            "mkdir -p ddp_rlhf_classify/projects_yaml",
            check=True,
        )

        # Upload the YAML file as config.yaml
        await asyncio.to_thread(
            ssh_executor.upload_file,
            self.yaml_file_path,
            "ddp_rlhf_classify/projects_yaml/config.yaml",
        )

        logger.info(
            "✅ Uploaded YAML file as ddp_rlhf_classify/projects_yaml/config.yaml"
        )

    async def test_file_transfer(self) -> None:
        """Test the file transfer functionality - upload YAML and download datasets from S3."""
        logger.info("=" * 70)
        logger.info("🚀 Testing file transfer")
        logger.info("=" * 70)

        # Upload local YAML file as config.yaml
        await self.upload_yaml_as_config()

        # Setup AWS credentials on GPU before downloading from S3
        logger.info("🔧 Setting up AWS credentials on GPU...")
        await self.workflow._setup_aws_on_gpu()

        # Download datasets from S3 (skip config since we uploaded it manually)
        logger.info("📥 Downloading datasets from S3...")

        ssh_executor = self.workflow.ssh_executor
        s3_bucket = self.workflow.job_s3_bucket
        yaml_data = self.yaml_data

        # Create data directory inside ddp_rlhf_classify
        await asyncio.to_thread(
            ssh_executor.execute_command,
            "mkdir -p ddp_rlhf_classify/data",
            check=True,
        )

        # Download train and eval datasets
        download_tasks = []

        # Train dataset
        if "train_s3_path" in yaml_data and "train_file_path" in yaml_data:
            train_local_path = f"ddp_rlhf_classify/{yaml_data['train_file_path']}"
            logger.info(
                f"   Downloading train dataset from s3://{s3_bucket}/{yaml_data['train_s3_path']}"
            )
            logger.info(f"   To: {train_local_path}")
            download_tasks.append(
                asyncio.to_thread(
                    ssh_executor.download_from_s3,
                    s3_bucket,
                    yaml_data["train_s3_path"],
                    train_local_path,
                    False,
                )
            )

        # Eval dataset
        if "eval_s3_path" in yaml_data and "eval_file_path" in yaml_data:
            eval_local_path = f"ddp_rlhf_classify/{yaml_data['eval_file_path']}"
            logger.info(
                f"   Downloading eval dataset from s3://{s3_bucket}/{yaml_data['eval_s3_path']}"
            )
            logger.info(f"   To: {eval_local_path}")
            download_tasks.append(
                asyncio.to_thread(
                    ssh_executor.download_from_s3,
                    s3_bucket,
                    yaml_data["eval_s3_path"],
                    eval_local_path,
                    False,
                )
            )

        # Train image directory (for image classification tasks)
        if (
            "train_image_root_s3_path" in yaml_data
            and "train_image_root_path" in yaml_data
            and yaml_data["train_image_root_s3_path"]
            and yaml_data["train_image_root_path"]
        ):
            train_images_local_path = (
                f"ddp_rlhf_classify/{yaml_data['train_image_root_path']}"
            )
            logger.info(
                f"   Downloading train images from s3://{s3_bucket}/{yaml_data['train_image_root_s3_path']}"
            )
            logger.info(f"   To: {train_images_local_path}")
            download_tasks.append(
                asyncio.to_thread(
                    ssh_executor.download_from_s3,
                    s3_bucket,
                    yaml_data["train_image_root_s3_path"],
                    train_images_local_path,
                    True,  # recursive=True for directories
                )
            )

        # Eval image directory (for image classification tasks)
        if (
            "eval_image_root_s3_path" in yaml_data
            and "eval_image_root_path" in yaml_data
            and yaml_data["eval_image_root_s3_path"]
            and yaml_data["eval_image_root_path"]
        ):
            eval_images_local_path = (
                f"ddp_rlhf_classify/{yaml_data['eval_image_root_path']}"
            )
            logger.info(
                f"   Downloading eval images from s3://{s3_bucket}/{yaml_data['eval_image_root_s3_path']}"
            )
            logger.info(f"   To: {eval_images_local_path}")
            download_tasks.append(
                asyncio.to_thread(
                    ssh_executor.download_from_s3,
                    s3_bucket,
                    yaml_data["eval_image_root_s3_path"],
                    eval_images_local_path,
                    True,  # recursive=True for directories
                )
            )

        # Execute downloads in parallel
        if download_tasks:
            await asyncio.gather(*download_tasks)
            logger.info("✅ Datasets downloaded from S3")
        else:
            logger.warning("⚠️  No datasets to download")

        logger.info("=" * 70)
        logger.info("✅ File transfer completed")
        logger.info("=" * 70)

    async def verify_files(self) -> None:
        """Verify that files were transferred successfully."""
        logger.info("🔍 Verifying transferred files...")

        ssh_executor = self.workflow.ssh_executor

        # List files in ddp_rlhf_classify directory
        result = await asyncio.to_thread(
            ssh_executor.execute_command,
            "ls -lh ddp_rlhf_classify/",
            check=False,
        )

        if result.success:
            logger.info("📂 Files in ddp_rlhf_classify/ directory:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

        # List files in data directory
        result = await asyncio.to_thread(
            ssh_executor.execute_command,
            "ls -lh ddp_rlhf_classify/data/",
            check=False,
        )

        if result.success:
            logger.info("📂 Files in ddp_rlhf_classify/data/ directory:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

        # List files in projects_yaml directory
        result = await asyncio.to_thread(
            ssh_executor.execute_command,
            "ls -lh ddp_rlhf_classify/projects_yaml/",
            check=False,
        )

        if result.success:
            logger.info("📂 Files in ddp_rlhf_classify/projects_yaml/ directory:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

        # Check image directories if they exist in YAML
        if "train_image_root_path" in self.yaml_data:
            img_dir = self.yaml_data["train_image_root_path"]
            result = await asyncio.to_thread(
                ssh_executor.execute_command,
                f"ls -lh {img_dir}/ 2>/dev/null | head -20",
                check=False,
            )

            if result.success and result.stdout.strip():
                logger.info(f"📂 Sample files in {img_dir}/ directory:")
                for line in result.stdout.split("\n")[:10]:
                    if line.strip():
                        logger.info(f"   {line}")

        if "eval_image_root_path" in self.yaml_data:
            img_dir = self.yaml_data["eval_image_root_path"]
            result = await asyncio.to_thread(
                ssh_executor.execute_command,
                f"ls -lh {img_dir}/ 2>/dev/null | head -20",
                check=False,
            )

            if result.success and result.stdout.strip():
                logger.info(f"📂 Sample files in {img_dir}/ directory:")
                for line in result.stdout.split("\n")[:10]:
                    if line.strip():
                        logger.info(f"   {line}")

        # Check disk usage
        result = await asyncio.to_thread(
            ssh_executor.execute_command,
            "df -h .",
            check=False,
        )

        if result.success:
            logger.info("💾 Disk usage:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

    async def cleanup(self) -> None:
        """Cleanup SSH connection."""
        if self.workflow and self.workflow.ssh_executor:
            logger.info("🧹 Cleaning up SSH connection...")
            await asyncio.to_thread(self.workflow.ssh_executor.disconnect)
            logger.info("✅ Cleanup complete")

    async def run(self) -> None:
        """Run the complete file transfer test."""
        try:
            # Load YAML configuration
            self.load_yaml_config()

            # Setup workflow
            await self.setup_workflow()

            # Establish SSH connection
            await self.establish_ssh_connection()

            # Test file transfer using workflow's method
            await self.test_file_transfer()

            # Verify files
            await self.verify_files()

            logger.info("=" * 70)
            logger.info("✅ File transfer test completed successfully!")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"❌ Test failed: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise
        finally:
            pass
            # await self.cleanup()


async def main():
    """Main entry point for the test script."""
    import argparse

    GPU_IP = "129.80.50.39"

    parser = argparse.ArgumentParser(
        description="Test TrainingWorkflow file transfer functionality using YAML configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file (e.g., test_requests/image_classification_request_*.yaml)",
    )

    args = parser.parse_args()

    # Validate environment variables
    required_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "SSH_KEY_PATH"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"❌ Missing required environment variables: {', '.join(missing_vars)}"
        )
        sys.exit(1)

    # Create and run tester
    tester = WorkflowFileTransferTester(gpu_ip=GPU_IP, yaml_file_path=args.config)

    await tester.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
