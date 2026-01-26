import paramiko
from typing import List, Dict, Optional, Tuple
import os
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import asyncio


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    stdout: str
    stderr: str
    return_code: int
    success: bool
    duration: float


class SshExecutor:
    def __init__(self, ip, username: str = "ubuntu", reuse_conection: bool = True):
        self.ip = ip
        self.username = username
        self.timeout = 200
        self.client = None
        self.reuse_conection = reuse_conection
        self.sftp = None
        load_dotenv()

    def get_sftp(self):
        if not self.sftp or not self.sftp.sock or self.sftp.sock.closed:
            self.sftp = self.client.open_sftp()
        return self.sftp

    def connect(self):

        if self.client is not None:
            return

        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_file_path = os.environ.get("SSH_KEY_PATH")
            ssh_file_pwd = os.environ.get("SSH_KEY_PASSWORD")
            private_key = paramiko.RSAKey.from_private_key_file(
                os.path.expanduser(ssh_file_path), password=ssh_file_pwd
            )
            client.connect(
                hostname=self.ip,
                username=self.username,
                pkey=private_key,
                timeout=self.timeout,
            )
            # Only set self.client after successful connection
            self.client = client
        except Exception as e:
            # Clean up failed client
            if client:
                try:
                    client.close()
                except:
                    pass
            raise Exception(f"Failed to connect to {self.ip}: {str(e)}")

    async def connect_async(self):
        """Asynchronous version of connect method"""
        if self.client is not None:
            return

        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_file_path = os.environ.get("SSH_KEY_PATH")
            ssh_file_pwd = os.environ.get("SSH_KEY_PASSWORD")
            private_key = paramiko.RSAKey.from_private_key_file(
                os.path.expanduser(ssh_file_path), password=ssh_file_pwd
            )
            await asyncio.to_thread(
                client.connect,
                hostname=self.ip,
                username=self.username,
                pkey=private_key,
                timeout=self.timeout,
            )
            # Only set self.client after successful connection
            self.client = client
        except Exception as e:
            # Clean up failed client
            if client:
                try:
                    client.close()
                except:
                    pass
            raise Exception(f"Failed to connect to {self.ip}: {str(e)}")

    def disconnect(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def execute_command(self, command: str, check=True):
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        start_time = time.time()
        try:
            stdin, stdout, stderr = self.client.exec_command(
                command, timeout=self.timeout, get_pty=True
            )
            exit_status = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode("utf-8").strip()
            stderr_str = stderr.read().decode("utf-8").strip()

            duration = time.time() - start_time
            success = exit_status == 0

            result = CommandResult(
                command=command,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=exit_status,
                success=success,
                duration=duration,
            )
            if check and exit_status != 0:
                raise Exception(
                    f"Command {command} failed with exit status {exit_status}: {stderr_str}"
                )

            return result
        except Exception as e:
            raise Exception(
                f"Failed to execute command {command} on {self.ip}: {str(e)}"
            )

    def upload_file(self, local_path, remote_path=None):
        if self.client is None:
            self.connect()
        if not remote_path:
            remote_path = os.path.basename(local_path)
        try:
            sftp = self.get_sftp()
            # Extract directory from remote_path
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:  # If there's a directory component
                try:
                    sftp.stat(remote_dir)  # Check if directory exists
                except FileNotFoundError:
                    # Create all parent directories
                    sftp.mkdir(remote_dir)
                    # Note: This only creates one level. For nested directories, use makedirs equivalent
            sftp.put(local_path, remote_path)
            sftp.close()
            return True
        except Exception as e:
            raise Exception(
                f"Failed to upload file {local_path} to {remote_path} on {self.ip}: {str(e)}"
            )

    def execute_script(self, script_path, check=True, remote_path=None):
        if self.client is None:
            self.connect()
        if not remote_path:
            remote_path = os.path.basename(script_path)
        if not os.path.isfile(script_path):
            raise Exception(f"Script {script_path} does not exist")
        try:
            self.upload_file(script_path, remote_path=remote_path)
            self.execute_command(f"chmod +x {remote_path}")
            result = self.execute_command(f"bash {remote_path}", check=check)
            return result
        except Exception as e:
            raise Exception(
                f"Failed to execute script {script_path} on {self.ip}: {str(e)}"
            )

    def download_file(self, remote_path, local_path=None):
        if self.client is None:
            self.connect()
        if not local_path:
            local_path = os.path.basename(remote_path)
        try:
            sftp = self.get_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            return True
        except Exception as e:
            raise Exception(
                f"Failed to download file {remote_path} to {local_path} on {self.ip}: {str(e)}"
            )

    def reconnect(self):
        """Attempt to reconnect to the remote server."""
        try:
            # Close existing connection if any
            if self.client:
                try:
                    self.client.close()
                except:
                    pass  # Ignore errors on close
                self.client = None

            # Connect with the same parameters as before
            self.connect()
            return True
        except Exception as e:
            print(f"Failed to reconnect to SSH server {self.ip}: {str(e)}")
            return False

    def is_connected(self):
        """Check if the SSH client is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        if self.client is None:
            return False

        try:
            # Try a simple command to check if connection is still active
            transport = self.client.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False

    def setup_aws_credentials(
        self, aws_access_key: str, aws_secret_key: str, region: str = "ap-south-1"
    ):
        """Configure AWS credentials on remote server by writing to ~/.aws/credentials.

        Args:
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            region: AWS region (default: ap-south-1)

        Returns:
            CommandResult: Result of the credential setup command
        """
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        # Create ~/.aws directory and write credentials file
        commands = [
            "mkdir -p ~/.aws",
            f"echo '[default]' > ~/.aws/credentials",
            f"echo 'aws_access_key_id = {aws_access_key}' >> ~/.aws/credentials",
            f"echo 'aws_secret_access_key = {aws_secret_key}' >> ~/.aws/credentials",
            f"echo '[default]' > ~/.aws/config",
            f"echo 'region = {region}' >> ~/.aws/config",
            f"echo 'output = json' >> ~/.aws/config",
            "chmod 600 ~/.aws/credentials",
            "chmod 600 ~/.aws/config",
            'echo "AWS credentials configured in ~/.aws/credentials"',
        ]
        combined_command = " && ".join(commands)
        return self.execute_command(combined_command, check=True)

    def download_from_s3(self, s3_bucket: str, s3_path: str, local_path: str):
        """Download file/folder from S3 to remote server using AWS CLI.

        Args:
            s3_bucket: S3 bucket name
            s3_path: S3 object key/path
            local_path: Destination path on remote server

        Returns:
            CommandResult: Result of the download command
        """
        if self.client is None:
            self.connect()

        # Ensure local directory exists
        local_dir = (
            os.path.dirname(local_path) if not s3_path.endswith("/") else local_path
        )
        if local_dir:
            self.execute_command(f"mkdir -p {local_dir}", check=False)

        # Download from S3
        command = f"aws s3 cp s3://{s3_bucket}/{s3_path} {local_path}"
        return self.execute_command(command, check=True)

    def upload_to_s3(
        self, local_path: str, s3_bucket: str, s3_path: str, recursive: bool = False
    ):
        """Upload file/folder from remote server to S3 using AWS CLI.

        Args:
            local_path: Source path on remote server
            s3_bucket: S3 bucket name
            s3_path: Destination S3 object key/path
            recursive: If True, use sync for directories; if False, use cp for single file

        Returns:
            CommandResult: Result of the upload command
        """
        if self.client is None:
            self.connect()

        # Remove trailing slashes to avoid double slashes
        local_path = local_path.rstrip("/")
        s3_path = s3_path.rstrip("/")

        if recursive:
            command = (
                f"aws s3 sync {local_path}/ s3://{s3_bucket}/{s3_path}/ --exclude *.log"
            )
        else:
            command = f"aws s3 cp {local_path} s3://{s3_bucket}/{s3_path}"
        return self.execute_command(command, check=True)

    def check_aws_cli_installed(self):
        """Check if AWS CLI is installed on remote server.

        Returns:
            bool: True if AWS CLI is installed, False otherwise
        """
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        result = self.execute_command("which aws", check=False)
        return result.success

    def verify_aws_credentials(self):
        """Verify AWS credentials are configured and working.

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        result = self.execute_command("aws sts get-caller-identity", check=False)
        return result.success

    def install_aws_cli(self):
        """Install AWS CLI v2 on remote server if not present.

        Returns:
            bool: True if installation successful, False otherwise
        """
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        try:
            commands = [
                'curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"',
                "unzip -q awscliv2.zip",
                "sudo ./aws/install",
                "rm -rf aws awscliv2.zip",
            ]
            for cmd in commands:
                self.execute_command(cmd, check=True)
            return True
        except Exception as e:
            print(f"Failed to install AWS CLI: {str(e)}")
            return False


def main():
    """Main function to test SSH connection and command execution.

    Usage:
        python ssh_executor.py <ip_address>

    Example:
        python ssh_executor.py 192.168.1.100
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ssh_executor.py <ip_address>")
        print("\nExample:")
        print("  python ssh_executor.py 192.168.1.100")
        sys.exit(1)

    ip_address = sys.argv[1]

    print(f"🔌 Connecting to {ip_address}...")

    try:
        # Create SSH executor instance
        executor = SshExecutor(ip=ip_address, username="ubuntu")

        # Establish connection
        executor.connect()
        print(f"✅ Connected to {ip_address}")

        # Test command: Create a file with echo
        test_message = (
            "Hello from InnoTone Training System! Connection test successful."
        )
        test_file = "~/innotone_test.txt"
        command = f'echo "{test_message}" > {test_file}'

        print(f"\n🚀 Testing command execution...")
        print(f"Command: {command}")
        print("=" * 70)

        # Execute the echo command
        result = executor.execute_command(command, check=False)

        # Display results
        print(f"\n📊 Command Results:")
        print(f"Exit Code: {result.return_code}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Success: {'✅' if result.success else '❌'}")

        if result.stdout:
            print(f"\n📤 STDOUT:")
            print(result.stdout)

        if result.stderr:
            print(f"\n📥 STDERR:")
            print(result.stderr)

        # Verify the file was created
        if result.success:
            print(f"\n🔍 Verifying file creation...")
            verify_result = executor.execute_command(f"cat {test_file}", check=False)

            if verify_result.success:
                print(f"✅ File created successfully!")
                print(f"📄 File contents:")
                print(f"   {verify_result.stdout}")
            else:
                print(f"❌ File verification failed")
                if verify_result.stderr:
                    print(f"Error: {verify_result.stderr}")

        # Disconnect
        executor.disconnect()
        print(f"\n🔌 Disconnected from {ip_address}")
        print(f"\n✅ Test completed successfully!")

        # Exit with command's exit code
        sys.exit(result.return_code)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
