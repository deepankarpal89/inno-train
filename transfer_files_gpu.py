#!/usr/bin/env python3
"""
Simulate file transfer operations to a GPU instance using a YAML configuration.
This script mimics the _transfer_all_files method from TrainingWorkflow.
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

from scripts.ssh_executor import SshExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileTransferSimulator:
    """Simulates file transfer operations to a GPU instance."""

    # Transfer configurations matching WorkflowConstants.TRANSFER_CONFIGS
    TRANSFER_CONFIGS = [
        ("train_s3_path", "train_file_path", "train dataset"),
        ("eval_s3_path", "eval_file_path", "eval dataset"),
        ("config_s3_path", "config_file_path", "config"),
    ]

    def __init__(self, gpu_ip: str, yaml_file_path: str, s3_bucket: str):
        """
        Initialize the file transfer simulator.

        Args:
            gpu_ip: IP address of the GPU instance
            yaml_file_path: Path to the YAML configuration file
            s3_bucket: S3 bucket name for file transfers
        """
        self.gpu_ip = gpu_ip
        self.yaml_file_path = yaml_file_path
        self.s3_bucket = s3_bucket
        self.ssh_executor = None
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

    async def setup_ssh_connection(self) -> None:
        """Establish SSH connection to the GPU instance."""
        logger.info(f"🔌 Establishing SSH connection to GPU: {self.gpu_ip}")

        self.ssh_executor = SshExecutor(ip=self.gpu_ip, username="ubuntu")

        # Connect asynchronously
        await self.ssh_executor.connect_async()
        logger.info("✅ SSH connection established")

    async def setup_aws_on_gpu(self) -> None:
        """Setup AWS CLI and credentials on GPU instance."""
        logger.info("🔧 Setting up AWS CLI on GPU instance...")

        try:
            # Check if AWS CLI is installed
            if not await asyncio.to_thread(self.ssh_executor.check_aws_cli_installed):
                logger.info("📦 Installing AWS CLI on GPU instance...")
                await asyncio.to_thread(self.ssh_executor.install_aws_cli)
                logger.info("✅ AWS CLI installed")
            else:
                logger.info("✅ AWS CLI already installed")

            # Configure AWS credentials
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

            if not aws_access_key or not aws_secret_key:
                raise ValueError(
                    "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY environment variables."
                )

            await asyncio.to_thread(
                self.ssh_executor.setup_aws_credentials,
                aws_access_key,
                aws_secret_key,
                aws_region,
            )
            logger.info("✅ AWS credentials configured on GPU")

            # Verify credentials
            logger.info("🔍 Verifying AWS credentials...")
            credentials_valid = await asyncio.to_thread(
                self.ssh_executor.verify_aws_credentials
            )

            if not credentials_valid:
                raise Exception(
                    "AWS credentials verification failed. Please check your credentials."
                )

            logger.info("✅ AWS credentials verified successfully")

        except Exception as e:
            raise Exception(f"Failed to setup AWS on GPU: {str(e)}")

    async def create_directories(self) -> None:
        """Create necessary directories on the GPU instance."""
        logger.info("📁 Creating directories on GPU instance...")

        # Create base directories including image directories
        await asyncio.to_thread(
            self.ssh_executor.execute_command,
            "mkdir -p data projects_yaml data/train_images data/eval_images",
            check=True,
        )

        logger.info("✅ Directories created successfully")

    async def download_file_from_s3(
        self, s3_bucket: str, s3_path: str, local_path: str, description: str
    ) -> None:
        """Download a single file from S3 to GPU server."""
        logger.info(f"📥 Downloading {description}...")
        logger.info(f"   S3: s3://{s3_bucket}/{s3_path}")
        logger.info(f"   Local: {local_path}")

        try:
            await asyncio.to_thread(
                self.ssh_executor.download_from_s3,
                s3_bucket,
                s3_path,
                local_path,
                False,  # recursive=False for single file
            )
            logger.info(f"✅ Downloaded {description}")
        except Exception as e:
            logger.error(f"❌ Failed to download {description}: {str(e)}")
            raise

    async def download_directory_from_s3(
        self, s3_bucket: str, s3_path: str, local_path: str, description: str
    ) -> None:
        """Download a directory recursively from S3 to GPU server."""
        logger.info(f"📥 Downloading {description} directory...")
        logger.info(f"   S3: s3://{s3_bucket}/{s3_path}/")
        logger.info(f"   Local: {local_path}/")

        try:
            await asyncio.to_thread(
                self.ssh_executor.download_from_s3,
                s3_bucket,
                s3_path,
                local_path,
                True,  # recursive=True for directory
            )
            logger.info(f"✅ Downloaded {description} directory")
        except Exception as e:
            logger.error(f"❌ Failed to download {description} directory: {str(e)}")
            raise

    async def transfer_all_files(self) -> None:
        """
        Download files from S3 directly to GPU instance using AWS CLI.
        This mimics the _transfer_all_files method from TrainingWorkflow.
        """
        logger.info("=" * 70)
        logger.info("🚀 Starting file transfer simulation")
        logger.info("=" * 70)

        # Setup AWS CLI on GPU
        await self.setup_aws_on_gpu()

        # Create directories
        await self.create_directories()

        # Prepare download tasks for CSV files and config
        download_tasks = []

        for s3_key, path_key, description in self.TRANSFER_CONFIGS:
            if s3_key in self.yaml_data and path_key in self.yaml_data:
                download_tasks.append(
                    self.download_file_from_s3(
                        self.s3_bucket,
                        self.yaml_data[s3_key],
                        self.yaml_data[path_key],
                        description,
                    )
                )
            else:
                logger.warning(f"⚠️  Skipping {description}: missing keys in YAML")

        # Download image directories if they exist in the YAML
        if (
            "train_image_root_s3_path" in self.yaml_data
            and "train_image_root_path" in self.yaml_data
        ):
            download_tasks.append(
                self.download_directory_from_s3(
                    self.s3_bucket,
                    self.yaml_data["train_image_root_s3_path"],
                    self.yaml_data["train_image_root_path"],
                    "training images",
                )
            )
        else:
            logger.info("ℹ️  No training images to download")

        if (
            "eval_image_root_s3_path" in self.yaml_data
            and "eval_image_root_path" in self.yaml_data
        ):
            download_tasks.append(
                self.download_directory_from_s3(
                    self.s3_bucket,
                    self.yaml_data["eval_image_root_s3_path"],
                    self.yaml_data["eval_image_root_path"],
                    "evaluation images",
                )
            )
        else:
            logger.info("ℹ️  No evaluation images to download")

        # Execute all downloads in parallel
        logger.info(f"📦 Executing {len(download_tasks)} download tasks in parallel...")
        await asyncio.gather(*download_tasks)

        logger.info("=" * 70)
        logger.info("✅ All files and images downloaded from S3 to GPU")
        logger.info("=" * 70)

    async def verify_files(self) -> None:
        """Verify that files were transferred successfully."""
        logger.info("🔍 Verifying transferred files...")

        # List files in data directory
        result = await asyncio.to_thread(
            self.ssh_executor.execute_command,
            "ls -lh data/",
            check=False,
        )

        if result.success:
            logger.info("📂 Files in data/ directory:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

        # List files in projects_yaml directory
        result = await asyncio.to_thread(
            self.ssh_executor.execute_command,
            "ls -lh projects_yaml/",
            check=False,
        )

        if result.success:
            logger.info("📂 Files in projects_yaml/ directory:")
            for line in result.stdout.split("\n"):
                if line.strip():
                    logger.info(f"   {line}")

        # Check image directories if they exist
        for img_dir in ["data/train_images", "data/eval_images"]:
            result = await asyncio.to_thread(
                self.ssh_executor.execute_command,
                f"ls -lh {img_dir}/ 2>/dev/null | head -20",
                check=False,
            )

            if result.success and result.stdout.strip():
                logger.info(f"📂 Sample files in {img_dir}/ directory:")
                for line in result.stdout.split("\n")[:10]:
                    if line.strip():
                        logger.info(f"   {line}")

    async def cleanup(self) -> None:
        """Cleanup SSH connection."""
        if self.ssh_executor:
            logger.info("🧹 Cleaning up SSH connection...")
            await asyncio.to_thread(self.ssh_executor.disconnect)
            logger.info("✅ Cleanup complete")

    async def run(self) -> None:
        """Run the complete file transfer simulation."""
        try:
            # Load YAML configuration
            self.load_yaml_config()

            # Setup SSH connection
            await self.setup_ssh_connection()

            # Transfer all files
            await self.transfer_all_files()

            # Verify files
            await self.verify_files()

        except Exception as e:
            logger.error(f"❌ Simulation failed: {str(e)}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main entry point for the simulation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate file transfers to a GPU instance using YAML configuration"
    )
    parser.add_argument(
        "--gpu-ip", required=True, help="IP address of the GPU instance"
    )
    parser.add_argument(
        "--yaml-file",
        required=True,
        help="Path to the YAML configuration file (e.g., test_requests/image_classification_request_*.yaml)",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("S3_BUCKET_NAME", "innotone-training-data"),
        help="S3 bucket name (default: from S3_BUCKET_NAME env var or 'innotone-training-data')",
    )

    args = parser.parse_args()

    # Create and run simulator
    simulator = FileTransferSimulator(
        gpu_ip=args.gpu_ip, yaml_file_path=args.yaml_file, s3_bucket=args.s3_bucket
    )

    await simulator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
