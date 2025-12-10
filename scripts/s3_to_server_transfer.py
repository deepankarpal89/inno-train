#!/usr/bin/env python3
"""
S3 to Server File Transfer Utility

This script transfers files from S3 to a remote server using StorageClient and SshExecutor.
It handles downloading files from S3 and uploading them to the target server.
"""

import os
import logging
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
import concurrent.futures

# Import local modules
from scripts.storage_client import StorageClient
from scripts.ssh_executor import SshExecutor

# Load environment variables
load_dotenv()


class S3ToServerTransfer:
    def __init__(self, storage_type: str = "minio", logger: Optional[logging.Logger] = None):
        """
        Initialize the transfer utility.

        Args:
            storage_type: Type of storage ('minio' or 'aws_s3')
            logger: Optional logger instance for logging transfer operations
        """
        self.storage = StorageClient(storage_type=storage_type)
        self.logger = logger or logging.getLogger(__name__)

    def get_ssh(self,server_ip: str,username: str = "ubuntu"):
        ssh = SshExecutor(ip=server_ip, username=username)
        ssh.connect()
        return ssh

    def _setup_temp_dir(self):
        """Create a temporary directory for file downloads."""
        temp_dir = tempfile.mkdtemp(prefix="s3_to_server_")
        self.logger.debug(f"Using temporary directory: {temp_dir}")
        return temp_dir

    def _cleanup_temp_dir(self,temp_dir):
        """Clean up temporary files."""
        if temp_dir is not None and os.path.exists(temp_dir):
            import shutil

            try:
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary directory: {e}")

    def transfer_file_to_server(self,s3_bucket: str,s3_file_path: str,server_ip: str,server_folder_path: str,username: str = "ubuntu",) -> bool:
        """
        Transfer a single file from S3 to a remote server.

        Args:
            s3_bucket: S3 bucket name
            s3_file_path: Full S3 key of the file to transfer
            server_ip: Target server IP address
            server_folder_path: Target directory on the server
            username: SSH username (default: 'ubuntu')

        Returns:
            bool: True if file was transferred successfully, False otherwise
        """
        ssh = None
        temp_dir = None
        try:
            # Set up SSH connection
            ssh = self.get_ssh(server_ip,username)

            folder_path = os.path.dirname(server_folder_path)
            # Create target directory on server if it doesn't exist
            ssh.execute_command(f"mkdir -p {folder_path}")

            # Set up temporary directory for download
            temp_dir = self._setup_temp_dir()

            filename = os.path.basename(s3_file_path)
            local_path = os.path.join(temp_dir, filename)
            remote_path = os.path.join(folder_path, filename)

            self.logger.debug(f"Transferring {s3_file_path} to {server_ip}:{remote_path}")

            # Download from S3
            if not self.storage.download_file(
                bucket_name=s3_bucket, object_name=s3_file_path, file_path=local_path
            ):
                self.logger.error(f"Failed to download {s3_file_path} from S3")
                return False

            # Upload to server
            if not ssh.upload_file(local_path, remote_path):
                self.logger.error(f"Failed to upload {filename} to server")
                return False

            self.logger.info(f"Successfully transferred {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Transfer failed: {str(e)}")
            return False
        finally:
            if temp_dir:
                self._cleanup_temp_dir(temp_dir)
            if ssh:
                ssh.disconnect()

    def transfer_file_to_s3(self,server_ip: str,server_file_path: str,s3_bucket: str,s3_file_path: str,username: str = "ubuntu") -> bool:
        """
        Transfer a single file from a remote server to S3.

        Args:
            server_ip: Source server IP address
            server_file_path: Full path to the file on the server
            s3_bucket: Target S3 bucket name
            s3_file_path: Target S3 key for the file
            username: SSH username (default: 'ubuntu')

        Returns:
            bool: True if file was transferred successfully, False otherwise
        """
        ssh = None
        temp_dir = None
        try:
            # Set up SSH connection
            ssh = self.get_ssh(server_ip,username)

            # Set up temporary directory for download
            temp_dir = self._setup_temp_dir()

            filename = os.path.basename(server_file_path)
            local_path = os.path.join(temp_dir, filename)

            self.logger.debug(f"Transferring {server_ip}:{server_file_path} to s3://{s3_bucket}/{s3_file_path}")

            # Download from server
            if not ssh.download_file(server_file_path, local_path):
                self.logger.error(f"Failed to download {server_file_path} from server")
                return False

            # Upload to S3
            if not self.storage.upload_file(file_path=local_path, bucket_name=s3_bucket, object_name=s3_file_path):
                self.logger.error(f"Failed to upload {filename} to S3")
                return False

            self.logger.info(f"Successfully transferred {filename} to S3")
            return True

        except Exception as e:
            self.logger.error(f"Transfer failed: {str(e)}")
            return False
        finally:
            if temp_dir:
                self._cleanup_temp_dir(temp_dir)
            if ssh:
                ssh.disconnect()

    def transfer_files_to_server(self,s3_bucket: str,s3_folder_path: str,server_ip: str,server_folder_path: str,
        username: str = "ubuntu",file_extensions: Optional[List[str]] = None,) -> bool:
        ssh = None
        success = True
        temp_dir = None

        try:
            # List files in S3
            objects = self.get_s3_file_paths(s3_bucket,s3_folder_path)

            # Set up SSH connection
            ssh = self.get_ssh(server_ip,username)
            folder_path = os.path.dirname(server_folder_path)
            # Create target directory on server if it doesn't exist
            ssh.execute_command(f"mkdir -p {folder_path}")

            # Set up temporary directory for downloads
            temp_dir = self._setup_temp_dir()

            # Transfer each file
            for obj in objects:
                s3_key = obj
                filename = os.path.basename(s3_key)
                local_path = os.path.join(temp_dir, filename)
                remote_path = os.path.join(folder_path, filename)

                self.logger.debug(f"Transferring {s3_key} to {server_ip}:{remote_path}")

                try:
                    # Download from S3
                    if not self.storage.download_file(
                        bucket_name=s3_bucket, object_name=s3_key, file_path=local_path
                    ):
                        self.logger.error(f"Failed to download {s3_key} from S3")
                        success = False
                        continue

                    # Upload to server
                    if not ssh.upload_file(local_path, remote_path):
                        self.logger.error(f"Failed to upload {filename} to server")
                        success = False
                        continue

                    self.logger.info(f"Successfully transferred {filename}")

                except Exception as e:
                    self.logger.error(f"Error transferring {s3_key}: {str(e)}")
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Transfer failed: {str(e)}")
            return False

        finally:
            if ssh:
                ssh.disconnect()
            if temp_dir:
                self._cleanup_temp_dir(temp_dir)

    def get_server_file_paths(self,server_ip,server_folder_path,username: str = "ubuntu",file_extensions: Optional[List[str]] = None,recursive: bool = True):
        ssh = self.get_ssh(server_ip,username)

        # Normalize paths
        server_path = server_folder_path.rstrip("/")

        # List files on the server
        find_cmd = f"find {server_path} -type f"
        if not recursive:
            find_cmd = f"find {server_path} -maxdepth 1 -type f"

        if file_extensions:
                ext_conditions = " -o ".join(
                    [f'-iname "*{ext}"' for ext in file_extensions]
                )
                find_cmd = f"{find_cmd} \\( {ext_conditions} \\)"

        result = ssh.execute_command(find_cmd)
        if not result or not result.stdout:
            self.logger.warning(f"No files found in {server_path}")
            ssh.disconnect()
            return []
        
        file_paths = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        self.logger.info(f"Found {len(file_paths)} file(s) in {server_path}")
        ssh.disconnect()
        return file_paths   

    def transfer_files_to_s3(self,server_ip: str,server_folder_path: str,s3_bucket: str,s3_folder_path: str,username: str = "ubuntu",file_extensions: Optional[List[str]] = None,recursive: bool = True) -> bool:
        success = True

        try:
            file_paths = self.get_server_file_paths(server_ip,server_folder_path,username,file_extensions,recursive)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for server_file_path in file_paths:
                    relative_path = os.path.relpath(server_file_path, server_folder_path)
                    s3_file_path = os.path.join(s3_folder_path, relative_path)
                    futures.append(executor.submit(self.transfer_file_to_s3,server_ip,server_file_path,s3_bucket,s3_file_path,username))
                results = [f.result() for f in futures]
                success = all(results)
            return success
            
        except Exception as e:
            self.logger.error(f"Error transferring {server_folder_path}: {str(e)}")
            success = False
            return success

    def get_s3_file_paths(self,s3_bucket,s3_folder_path):
        objects =  self.storage.list_objects(bucket_name=s3_bucket,prefix=s3_folder_path)
        if not objects:
            self.logger.warning(f"No files found in {s3_bucket}/{s3_folder_path}")
            return []
        return [obj["name"] for obj in objects]


def main():
    from scripts.file_logger import get_file_logger
    logger = get_file_logger('file_transfer')

    file_transfer = S3ToServerTransfer(logger=logger)
    server_ip = '157.151.235.14'
    s3_bucket = os.getenv("BUCKET_NAME")
    server_folder_path = '/home/ubuntu/rlhf-virginia/output/'
    s3_folder_path = 'media/projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/train/'
    file_transfer.transfer_files_to_s3(server_folder_path=server_folder_path,s3_folder_path=s3_folder_path,s3_bucket=s3_bucket,server_ip=server_ip)


if __name__ == "__main__":
    main()