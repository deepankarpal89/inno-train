#!/usr/bin/env python3
"""
S3 to Server File Transfer Utility

This script transfers files from S3 to a remote server using StorageClient and SshExecutor.
It handles downloading files from S3 and uploading them to the target server.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Import local modules
from .storage_client import StorageClient
from .ssh_executor import SshExecutor

# Load environment variables
load_dotenv()


class S3ToServerTransfer:
    def __init__(
        self, storage_type: str = "minio", logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the transfer utility.

        Args:
            storage_type: Type of storage ('minio' or 'aws_s3')
            logger: Optional logger instance for logging transfer operations
        """
        self.storage = StorageClient(storage_type=storage_type)
        self.temp_dir = None
        self.logger = logger or logging.getLogger(__name__)

    def _setup_temp_dir(self):
        """Create a temporary directory for file downloads."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="s3_to_server_")
            self.logger.debug(f"Using temporary directory: {self.temp_dir}")
        return self.temp_dir

    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir is not None and os.path.exists(self.temp_dir):
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary directory: {e}")

    def transfer_file_to_server(
        self,
        s3_bucket: str,
        s3_key: str,
        server_ip: str,
        server_path: str,
        username: str = "ubuntu",
    ) -> bool:
        """
        Transfer a single file from S3 to a remote server.

        Args:
            s3_bucket: S3 bucket name
            s3_key: Full S3 key of the file to transfer
            server_ip: Target server IP address
            server_path: Target directory on the server
            username: SSH username (default: 'ubuntu')

        Returns:
            bool: True if file was transferred successfully, False otherwise
        """
        ssh = None
        try:
            # Set up SSH connection
            ssh = SshExecutor(ip=server_ip, username=username)
            ssh.connect()

            folder_path = os.path.dirname(server_path)
            # Create target directory on server if it doesn't exist
            ssh.execute_command(f"mkdir -p {folder_path}")

            # Set up temporary directory for download
            self._setup_temp_dir()

            filename = os.path.basename(s3_key)
            local_path = os.path.join(self.temp_dir, filename)
            remote_path = os.path.join(folder_path, filename)

            self.logger.debug(f"Transferring {s3_key} to {server_ip}:{remote_path}")

            # Download from S3
            if not self.storage.download_file(
                bucket_name=s3_bucket, object_name=s3_key, file_path=local_path
            ):
                self.logger.error(f"Failed to download {s3_key} from S3")
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
            if ssh:
                ssh.disconnect()

    def transfer_file_to_s3(
        self,
        server_ip: str,
        server_path: str,
        s3_bucket: str,
        s3_key: str,
        username: str = "ubuntu",
    ) -> bool:
        """
        Transfer a single file from a remote server to S3.

        Args:
            server_ip: Source server IP address
            server_path: Full path to the file on the server
            s3_bucket: Target S3 bucket name
            s3_key: Target S3 key for the file
            username: SSH username (default: 'ubuntu')

        Returns:
            bool: True if file was transferred successfully, False otherwise
        """
        ssh = None
        try:
            # Set up SSH connection
            ssh = SshExecutor(ip=server_ip, username=username)
            ssh.connect()

            # Set up temporary directory for download
            self._setup_temp_dir()

            filename = os.path.basename(server_path)
            local_path = os.path.join(self.temp_dir, filename)

            self.logger.debug(
                f"Transferring {server_ip}:{server_path} to s3://{s3_bucket}/{s3_key}"
            )

            # Download from server
            if not ssh.download_file(server_path, local_path):
                self.logger.error(f"Failed to download {server_path} from server")
                return False

            # Upload to S3
            if not self.storage.upload_file(
                file_path=local_path, bucket_name=s3_bucket, object_name=s3_key
            ):
                self.logger.error(f"Failed to upload {filename} to S3")
                return False

            self.logger.info(f"Successfully transferred {filename} to S3")
            return True

        except Exception as e:
            self.logger.error(f"Transfer failed: {str(e)}")
            return False
        finally:
            if ssh:
                ssh.disconnect()

    def transfer_files_to_server(
        self,
        s3_bucket: str,
        s3_prefix: str,
        server_ip: str,
        server_path: str,
        username: str = "ubuntu",
        file_extensions: Optional[List[str]] = None,
        ) -> bool:
        """
        Transfer files from S3 to a remote server.

        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix (folder path)
            server_ip: Target server IP address
            server_path: Target directory on the server
            username: SSH username (default: 'ubuntu')
            file_extensions: Optional list of file extensions to filter by

        Returns:
            bool: True if all files were transferred successfully, False otherwise
        """
        ssh = None
        success = True

        try:
            # List files in S3
            objects = self.storage.list_objects(s3_bucket, prefix=s3_prefix)
            if not objects:
                self.logger.warning(f"No files found in {s3_bucket}/{s3_prefix}")
                return False

            # Filter by file extensions if specified
            if file_extensions:
                objects = [
                    obj
                    for obj in objects
                    if any(
                        obj["name"].lower().endswith(ext.lower())
                        for ext in file_extensions
                    )
                ]
                if not objects:
                    self.logger.warning(
                        f"No files with extensions {file_extensions} found in {s3_bucket}/{s3_prefix}"
                    )
                    return False

            self.logger.info(f"Found {len(objects)} file(s) to transfer")

            # Set up SSH connection
            ssh = SshExecutor(ip=server_ip, username=username)
            ssh.connect()
            folder_path = os.path.dirname(server_path)
            # Create target directory on server if it doesn't exist
            ssh.execute_command(f"mkdir -p {folder_path}")

            # Set up temporary directory for downloads
            self._setup_temp_dir()

            # Transfer each file
            for obj in objects:
                s3_key = obj["name"]
                filename = os.path.basename(s3_key)
                local_path = os.path.join(self.temp_dir, filename)
                remote_path = os.path.join(server_path, filename)

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
            # Clean up connections and temporary files
            if ssh:
                ssh.disconnect()
            self._cleanup()

    def transfer_files_to_s3(
        self,
        server_ip: str,
        server_path: str,
        s3_bucket: str,
        s3_prefix: str = "",
        username: str = "ubuntu",
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,  # Changed default to True
    ) -> bool:
        """
        Transfer files from a remote server to S3 while maintaining directory structure.

        Args:
            server_ip: Source server IP address
            server_path: Source directory on the server
            s3_bucket: Target S3 bucket name
            s3_prefix: Target S3 prefix (base folder path, default: "")
            username: SSH username (default: 'ubuntu')
            file_extensions: Optional list of file extensions to filter by
            recursive: Whether to transfer files in subdirectories (default: True)

        Returns:
            bool: True if all files were transferred successfully, False otherwise
        """
        ssh = None
        success = True

        try:
            # Set up SSH connection
            ssh = SshExecutor(ip=server_ip, username=username)
            ssh.connect()

            # Normalize paths
            server_path = server_path.rstrip("/")
            base_dir = (
                os.path.dirname(server_path)
                if not server_path.endswith("*")
                else os.path.dirname(server_path.rstrip("*"))
            )

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
                return False

            file_paths = [
                line.strip() for line in result.stdout.split("\n") if line.strip()
            ]
            self.logger.info(f"Found {len(file_paths)} file(s) to transfer")

            # Set up temporary directory for downloads
            self._setup_temp_dir()

            # Transfer each file
            for file_path in file_paths:
                try:
                    # Calculate relative path to maintain directory structure
                    rel_path = os.path.relpath(file_path, base_dir)
                    local_path = os.path.join(
                        self.temp_dir, os.path.basename(file_path)
                    )

                    # Download from server
                    if not ssh.download_file(file_path, local_path):
                        self.logger.error(f"Failed to download {file_path} from server")
                        success = False
                        continue

                    # Construct S3 key with full path
                    s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")

                    # Upload to S3
                    if not self.storage.upload_file(
                        file_path=local_path, bucket_name=s3_bucket, object_name=s3_key
                    ):
                        self.logger.error(f"Failed to upload {file_path} to S3")
                        success = False
                        continue

                    self.logger.info(
                        f"Successfully transferred {file_path} to s3://{s3_bucket}/{s3_key}"
                    )

                except Exception as e:
                    self.logger.error(f"Error transferring {file_path}: {str(e)}")
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Transfer failed: {str(e)}")
            return False
        finally:
            # Clean up connections and temporary files
            if ssh:
                ssh.disconnect()
            self._cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer files between S3 and a remote server"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Transfer direction"
    )

    # Parser for S3 to Server transfer
    s3_to_server = subparsers.add_parser(
        "s3-to-server", help="Transfer files from S3 to a remote server"
    )
    s3_to_server.add_argument("--bucket", required=True, help="S3 bucket name")
    s3_to_server.add_argument("--prefix", default="", help="S3 prefix (folder path)")
    s3_to_server.add_argument(
        "--server", required=True, help="Target server IP or hostname"
    )
    s3_to_server.add_argument(
        "--path", required=True, help="Target directory on the server"
    )
    s3_to_server.add_argument(
        "--user", default="ubuntu", help="SSH username (default: ubuntu)"
    )
    s3_to_server.add_argument(
        "--ext", nargs="+", help="File extensions to include (e.g., .csv .txt)"
    )

    # Parser for Server to S3 transfer
    server_to_s3 = subparsers.add_parser(
        "server-to-s3", help="Transfer files from a remote server to S3"
    )
    server_to_s3.add_argument(
        "--server", required=True, help="Source server IP or hostname"
    )
    server_to_s3.add_argument(
        "--path", required=True, help="Source directory on the server"
    )
    server_to_s3.add_argument("--bucket", required=True, help="S3 bucket name")
    server_to_s3.add_argument(
        "--prefix", default="", help="S3 prefix (folder path, default: root)"
    )
    server_to_s3.add_argument(
        "--user", default="ubuntu", help="SSH username (default: ubuntu)"
    )
    server_to_s3.add_argument(
        "--ext", nargs="+", help="File extensions to include (e.g., .csv .txt)"
    )
    server_to_s3.add_argument(
        "--recursive", action="store_true", help="Transfer files in subdirectories"
    )

    # Common arguments
    for p in [s3_to_server, server_to_s3]:
        p.add_argument(
            "--storage",
            default="minio",
            choices=["minio", "aws_s3"],
            help="Storage type (default: minio)",
        )

    args = parser.parse_args()

    transfer = S3ToServerTransfer(storage_type=args.storage)

    if args.command == "s3-to-server":
        success = transfer.transfer_files(
            s3_bucket=args.bucket,
            s3_prefix=args.prefix,
            server_ip=args.server,
            server_path=args.path,
            username=args.user,
            file_extensions=args.ext,
        )
        success_msg = "All files transferred successfully"
        failure_msg = "Some files failed to transfer"
    else:  # server-to-s3
        success = transfer.transfer_files_to_s3(
            server_ip=args.server,
            server_path=args.path,
            s3_bucket=args.bucket,
            s3_prefix=args.prefix,
            username=args.user,
            file_extensions=args.ext,
            recursive=args.recursive,
        )
        success_msg = "All files transferred to S3 successfully"
        failure_msg = "Some files failed to transfer to S3"

    if success:
        print(success_msg)
    else:
        print(failure_msg)
        exit(1)


if __name__ == "__main__":
    main()
