import paramiko
import os
import time
import random
from functools import wraps
from typing import List, Dict, Optional, Tuple, Callable, Type
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import socket


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    stdout: str
    stderr: str
    return_code: int
    success: bool
    duration: float


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for retrying operations with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed, raise the exception
                        break

                    # Calculate delay with exponential backoff + jitter
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, 0.1),
                        max_delay,
                    )

                    print(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class SshExecutor:
    def __init__(
        self, ip: str, username: str = "ubuntu", reuse_connection: bool = True
    ):
        self.ip = ip
        self.username = username
        self.timeout = 200
        self.client = None
        self.reuse_connection = reuse_connection
        self.sftp = None
        self.temp_known_hosts = f"/tmp/known_hosts_{self.ip.replace('.', '_')}"

        load_dotenv()

    def validate_ssh_environment(self) -> Tuple[str, Optional[str]]:
        """Validate SSH environment variables and files."""
        ssh_file_path = os.environ.get("SSH_KEY_PATH")
        ssh_file_pwd = os.environ.get("SSH_KEY_PASSWORD")

        if ssh_file_path is None:
            raise ValueError("SSH_KEY_PATH environment variable is not set")

        expanded_path = os.path.expanduser(ssh_file_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"SSH key file not found: {expanded_path}")

        # Test if we can read the key file
        try:
            paramiko.RSAKey.from_private_key_file(expanded_path, password=ssh_file_pwd)
        except paramiko.PasswordRequiredException:
            raise ValueError(
                "SSH key requires password but SSH_KEY_PASSWORD is not set"
            )
        except paramiko.SSHException as e:
            raise ValueError(f"Invalid SSH key file: {str(e)}")

        return expanded_path, ssh_file_pwd

    def get_sftp(self) -> paramiko.SFTPClient:
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        if not self.sftp or not self.sftp.sock or self.sftp.sock.closed:
            if self.sftp is not None:
                try:
                    self.sftp.close()
                except Exception:
                    pass
            self.sftp = self.client.open_sftp()
        return self.sftp

    @retry_with_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(paramiko.SSHException, OSError, IOError),
    )
    def connect(self) -> None:

        if self.client is not None:
            return

        client = None
        try:
            client = paramiko.SSHClient()

            # load existing temp file it it exists
            if os.path.exists(self.temp_known_hosts):
                client.load_host_keys(self.temp_known_hosts)
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            ssh_file_path, ssh_file_pwd = self.validate_ssh_environment()
            private_key = paramiko.RSAKey.from_private_key_file(
                os.path.expanduser(ssh_file_path), password=ssh_file_pwd
            )
            client.connect(
                hostname=self.ip,
                username=self.username,
                pkey=private_key,
                timeout=self.timeout,
            )
            # Save the host key to temp file for future connections
            client.save_host_keys(self.temp_known_hosts)
            # Only set self.client after successful connection
            self.client = client
        except (paramiko.AuthenticationException, paramiko.SSHException) as e:
            if client:
                try:
                    client.close()
                except paramiko.SSHException:
                    pass  # Ignore cleanup errors
            raise Exception(f"SSH connection failed to {self.ip}: {str(e)}")
        except (OSError, IOError) as e:
            raise Exception(f"Network error connecting to {self.ip}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error connecting to {self.ip}: {str(e)}")

    @retry_with_backoff(
        max_retries=3,
        base_delay=2.0,
        exceptions=(paramiko.SSHException, OSError, IOError),
    )
    async def connect_async(self) -> None:
        """Asynchronous version of connect method"""
        if self.client is not None:
            return

        client = None
        try:
            client = paramiko.SSHClient()
            # load existing temp file if it exists
            if os.path.exists(self.temp_known_hosts):
                client.load_host_keys(self.temp_known_hosts)
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_file_path, ssh_file_pwd = self.validate_ssh_environment()
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
            # Save the host key to temp file for future connections
            client.save_host_keys(self.temp_known_hosts)
            # Only set self.client after successful connection
            self.client = client
        except (paramiko.AuthenticationException, paramiko.SSHException) as e:
            if client:
                try:
                    client.close()
                except paramiko.SSHException:
                    pass  # Ignore cleanup errors
            raise Exception(f"SSH connection failed to {self.ip}: {str(e)}")
        except (OSError, IOError) as e:
            raise Exception(f"Network error connecting to {self.ip}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error connecting to {self.ip}: {str(e)}")

    def disconnect(self) -> None:
        if self.client is not None:
            if self.sftp is not None:
                try:
                    self.sftp.close()
                except Exception:
                    pass
                self.sftp = None

            self.client.close()
            self.client = None
        # Clean up temporary known hosts file
        if os.path.exists(self.temp_known_hosts):
            try:
                os.remove(self.temp_known_hosts)
            except Exception:
                pass

    @retry_with_backoff(
        max_retries=2,
        base_delay=0.5,
        exceptions=(paramiko.SSHException, socket.timeout),
    )
    def execute_command(
        self, command: str, check: bool = True, get_pty: bool = True
    ) -> CommandResult:
        """Execute a command on the remote server.

        Args:
            command: The command to execute
            check: If True, raise exception on non-zero exit code
            get_pty: If True, allocate a pseudo-terminal. Set to False for
                     background commands (nohup) to prevent the process from
                     being killed when the SSH session closes.

        Returns:
            CommandResult: Result of the command execution
        """
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        start_time = time.time()
        try:
            stdin, stdout, stderr = self.client.exec_command(
                command, timeout=self.timeout, get_pty=get_pty
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
        except paramiko.SSHException as e:
            raise Exception(f"SSH error executing '{command}' on {self.ip}: {str(e)}")
        except socket.timeout as e:
            raise Exception(f"Timeout executing '{command}' on {self.ip}: {str(e)}")
        except Exception as e:
            raise Exception(
                f"Unexpected error executing '{command}' on {self.ip}: {str(e)}"
            )

    def expand_remote_path(self, remote_path: str) -> str:
        """Expand tilde (~) in remote path to actual home directory.

        SFTP doesn't expand ~ automatically, so we need to do it manually.
        """
        if remote_path.startswith("~"):
            # Get home directory from remote server
            result = self.execute_command("echo $HOME", check=False)
            if result.success and result.stdout:
                home_dir = result.stdout.strip()
                # Replace ~ with actual home directory
                if remote_path == "~":
                    return home_dir
                elif remote_path.startswith("~/"):
                    return remote_path.replace("~", home_dir, 1)
        return remote_path

    def ensure_remote_directory_ssh(self, remote_path: str) -> None:
        """Create nested directories using SSH command (more reliable)."""
        if not remote_path or remote_path == "/":
            return

        # Use mkdir -p which creates all parent directories
        command = f"mkdir -p {remote_path}"
        result = self.execute_command(command, check=False)

        if not result.success and "File exists" not in result.stderr:
            raise Exception(
                f"Failed to create directory {remote_path}: {result.stderr}"
            )

    @retry_with_backoff(
        max_retries=2,
        base_delay=1.0,
        exceptions=(paramiko.SSHException, IOError, OSError),
    )
    def upload_file(self, local_path: str, remote_path: Optional[str] = None) -> bool:
        if self.client is None:
            self.connect()
        if not remote_path:
            remote_path = os.path.basename(local_path)
        try:
            sftp = self.get_sftp()
            # Expand tilde in remote path
            expanded_remote_path = self.expand_remote_path(remote_path)
            # Extract directory from remote_path
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:  # If there's a directory component
                self.ensure_remote_directory_ssh(remote_dir)
            sftp.put(local_path, expanded_remote_path)

            return True
        except (IOError, OSError) as e:
            raise Exception(f"File system error uploading {local_path}: {str(e)}")
        except paramiko.SSHException as e:
            raise Exception(f"SFTP error uploading {local_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error uploading {local_path}: {str(e)}")

    @retry_with_backoff(
        max_retries=2,
        base_delay=1.0,
        exceptions=(paramiko.SSHException, IOError, OSError),
    )
    def execute_script(
        self, script_path: str, check: bool = True, remote_path: Optional[str] = None
    ) -> CommandResult:
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

    @retry_with_backoff(
        max_retries=2,
        base_delay=1.0,
        exceptions=(paramiko.SSHException, IOError, OSError),
    )
    def download_file(self, remote_path: str, local_path: Optional[str] = None) -> bool:
        if self.client is None:
            self.connect()
        if not local_path:
            local_path = os.path.basename(remote_path)

        sftp = None
        try:
            sftp = self.get_sftp()
            # Expand tilde in remote path
            expanded_remote_path = self.expand_remote_path(remote_path)
            sftp.get(expanded_remote_path, local_path)
            return True
        except Exception as e:
            raise Exception(
                f"Failed to download file {remote_path} to {local_path} on {self.ip}: {str(e)}"
            )

    def reconnect(self) -> bool:
        """Attempt to reconnect to the remote server."""
        try:
            # Close existing connection if any
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass  # Ignore errors on close
                self.client = None

            # Connect with the same parameters as before
            self.connect()
            return True
        except Exception as e:
            print(f"Failed to reconnect to SSH server {self.ip}: {str(e)}")
            return False

    def is_connected(self) -> bool:
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
        except (paramiko.SSHException, AttributeError):
            return False

    def setup_aws_credentials(
        self,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "ap-south-1",
    ) -> CommandResult:
        """Configure AWS credentials by uploading files."""
        if self.client is None:
            raise Exception(
                "SSH connection not established. Call connect() or connect_async() first."
            )

        # Read from environment variables if not provided
        if not aws_access_key:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        if not aws_secret_key:
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not region:
            region = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

        # Validate credentials are available
        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "AWS credentials not provided. Either pass aws_access_key and aws_secret_key "
                "or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

        # Create temporary files locally
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".credentials", delete=False
        ) as cred_file:
            cred_file.write(f"[default]\naws_access_key_id = {aws_access_key}\n")
            cred_file.write(f"aws_secret_access_key = {aws_secret_key}\n")
            cred_path = cred_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".config", delete=False
        ) as config_file:
            config_file.write(f"[default]\nregion = {region}\noutput = json\n")
            config_path = config_file.name

        try:
            # Upload files securely
            self.execute_command("mkdir -p ~/.aws")
            self.upload_file(cred_path, "~/.aws/credentials")
            self.upload_file(config_path, "~/.aws/config")
            self.execute_command("chmod 600 ~/.aws/credentials ~/.aws/config")

            return CommandResult(
                command="setup_aws_credentials",
                stdout="AWS credentials configured securely via file upload",
                stderr="",
                return_code=0,
                success=True,
                duration=0,
            )
        finally:
            # Clean up temporary files
            os.unlink(cred_path)
            os.unlink(config_path)

    def download_from_s3(
        self, s3_bucket: str, s3_path: str, local_path: str, recursive: bool = False
    ) -> CommandResult:
        """Download file/folder from S3 to remote server using AWS CLI.

        Args:
            s3_bucket: S3 bucket name
            s3_path: S3 object key/path
            local_path: Destination path on remote server
            recursive: If True, use sync for directories; if False, use cp for single file

        Returns:
            CommandResult: Result of the download command
        """
        if self.client is None:
            self.connect()

        # Ensure local directory exists
        if recursive:
            # For recursive downloads, create the target directory
            self.execute_command(f"mkdir -p {local_path}", check=False)
        else:
            # For single file downloads, create parent directory
            local_dir = os.path.dirname(local_path)
            if local_dir:
                self.execute_command(f"mkdir -p {local_dir}", check=False)

        # Remove trailing slashes for consistency
        s3_path = s3_path.rstrip("/")
        local_path = local_path.rstrip("/")

        # Download from S3
        if recursive:
            command = f"aws s3 cp s3://{s3_bucket}/{s3_path}/ {local_path}/ --recursive"
        else:
            command = f"aws s3 cp s3://{s3_bucket}/{s3_path} {local_path}"
        return self.execute_command(command, check=True)

    def upload_to_s3(
        self, local_path: str, s3_bucket: str, s3_path: str, recursive: bool = False
    ) -> CommandResult:
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

    def check_aws_cli_installed(self) -> bool:
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

    def verify_aws_credentials(self) -> bool:
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

    def install_aws_cli(self) -> bool:
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


def test_aws_functionality(executor: "SshExecutor") -> dict:
    """Test AWS CLI functionality on remote server.

    Args:
        executor: Connected SshExecutor instance

    Returns:
        dict: Test results with status and details
    """
    results = {
        "aws_cli_installed": False,
        "aws_cli_version": None,
        "aws_credentials_configured": False,
        "caller_identity": None,
        "s3_list_success": False,
        "s3_buckets": [],
        "error": None,
    }

    print("\n" + "=" * 70)
    print("☁️  AWS FUNCTIONALITY TEST")
    print("=" * 70)

    try:
        # Test 1: Check AWS CLI installation
        print("\n[AWS TEST 1] 🔍 Checking AWS CLI Installation...")
        aws_installed = executor.check_aws_cli_installed()
        results["aws_cli_installed"] = aws_installed

        if not aws_installed:
            print("❌ AWS CLI is not installed")
            results["error"] = "AWS CLI not installed"
            return results

        print("✅ AWS CLI is installed")

        # Get AWS CLI version
        version_result = executor.execute_command("aws --version", check=False)
        if version_result.success:
            results["aws_cli_version"] = version_result.stdout.strip()
            print(f"📦 Version: {results['aws_cli_version']}")

        # Test 1.5: Setup AWS credentials if available in environment
        print("\n[AWS TEST 1.5] 🔑 Setting up AWS Credentials from Environment...")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

        if aws_key and aws_secret:
            try:
                executor.setup_aws_credentials()
                print("✅ AWS credentials uploaded to remote server")
            except Exception as e:
                print(f"⚠️  Failed to upload credentials: {str(e)}")
        else:
            print("⚠️  AWS credentials not found in environment variables")
            print(
                "💡 Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test with credentials"
            )

        # Test 2: Verify AWS credentials
        print("\n[AWS TEST 2] 🔐 Verifying AWS Credentials...")
        creds_valid = executor.verify_aws_credentials()
        results["aws_credentials_configured"] = creds_valid

        if not creds_valid:
            print("❌ AWS credentials are not configured or invalid")
            print("💡 Hint: Configure credentials using:")
            print("   - aws configure")
            print(
                "   - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
            )
            results["error"] = "AWS credentials not configured"
            return results

        print("✅ AWS credentials are valid")

        # Get caller identity
        identity_result = executor.execute_command(
            "aws sts get-caller-identity", check=False
        )
        if identity_result.success:
            results["caller_identity"] = identity_result.stdout.strip()
            print(f"👤 Caller Identity:")
            # Pretty print JSON if possible
            try:
                import json

                identity_json = json.loads(identity_result.stdout)
                print(f"   Account: {identity_json.get('Account', 'N/A')}")
                print(f"   UserId: {identity_json.get('UserId', 'N/A')}")
                print(f"   Arn: {identity_json.get('Arn', 'N/A')}")
            except:
                print(f"   {identity_result.stdout[:200]}")

        # Test 3: List S3 buckets
        print("\n[AWS TEST 3] 🪣 Testing S3 Access (aws s3 ls)...")
        s3_result = executor.execute_command("aws s3 ls", check=False)

        if s3_result.success:
            results["s3_list_success"] = True
            print("✅ S3 access successful")

            if s3_result.stdout.strip():
                bucket_lines = s3_result.stdout.strip().split("\n")
                results["s3_buckets"] = bucket_lines
                print(f"\n📦 Found {len(bucket_lines)} S3 bucket(s):")
                print("-" * 70)
                for line in bucket_lines[:10]:  # Show first 10 buckets
                    print(f"   {line}")
                if len(bucket_lines) > 10:
                    print(f"   ... and {len(bucket_lines) - 10} more bucket(s)")
            else:
                print("📭 No S3 buckets found (or empty output)")
        else:
            print("❌ S3 access failed")
            if s3_result.stderr:
                print(f"Error: {s3_result.stderr[:200]}")
            results["error"] = f"S3 list failed: {s3_result.stderr}"

        # Test 4: Check S3 regions
        print("\n[AWS TEST 4] 🌍 Checking AWS Region Configuration...")
        region_result = executor.execute_command(
            "aws configure get region", check=False
        )
        if region_result.success and region_result.stdout.strip():
            print(f"✅ Configured region: {region_result.stdout.strip()}")
        else:
            print("⚠️  No default region configured")

        # Test 5: Test S3 API call with specific command
        print("\n[AWS TEST 5] 🔧 Testing S3 API (list-buckets)...")
        api_result = executor.execute_command(
            "aws s3api list-buckets --query 'Buckets[].Name' --output text", check=False
        )
        if api_result.success:
            print("✅ S3 API call successful")
            if api_result.stdout.strip():
                bucket_names = api_result.stdout.strip().split()
                print(f"📋 Bucket names: {', '.join(bucket_names[:5])}")
                if len(bucket_names) > 5:
                    print(f"   ... and {len(bucket_names) - 5} more")
        else:
            print("❌ S3 API call failed")

        print("\n" + "=" * 70)
        print("📊 AWS TEST SUMMARY")
        print("=" * 70)
        print(f"AWS CLI Installed: {'✅' if results['aws_cli_installed'] else '❌'}")
        print(
            f"Credentials Valid: {'✅' if results['aws_credentials_configured'] else '❌'}"
        )
        print(f"S3 Access: {'✅' if results['s3_list_success'] else '❌'}")
        print(f"S3 Buckets Found: {len(results['s3_buckets'])}")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ AWS Test Error: {str(e)}")
        results["error"] = str(e)
        import traceback

        traceback.print_exc()
        return results


def main():
    """Main function to test SSH connection and various SshExecutor functionalities.

    Usage:
        python ssh_executor.py <ip_address>

    Example:
        python ssh_executor.py 192.168.1.100
    """
    import sys
    import tempfile

    if len(sys.argv) < 2:
        print("Usage: python ssh_executor.py <ip_address>")
        print("\nExample:")
        print("  python ssh_executor.py 192.168.1.100")
        sys.exit(1)

    ip_address = sys.argv[1]
    test_results = []

    print("=" * 70)
    print(f"🧪 SSH EXECUTOR COMPREHENSIVE TEST SUITE")
    print(f"🎯 Target: {ip_address}")
    print("=" * 70)

    try:
        executor = SshExecutor(ip=ip_address, username="ubuntu")

        # TEST 1: Connection
        print("\n[TEST 1] 🔌 Testing Connection...")
        executor.connect()
        print(f"✅ Connected to {ip_address}")
        test_results.append(("Connection", True))

        # TEST 2: Connection Status Check
        print("\n[TEST 2] 🔍 Testing Connection Status Check...")
        is_connected = executor.is_connected()
        print(f"Connection Status: {'✅ Active' if is_connected else '❌ Inactive'}")
        test_results.append(("Connection Status", is_connected))

        # TEST 3: Simple Command Execution
        print("\n[TEST 3] 🚀 Testing Simple Command Execution...")
        result = executor.execute_command("echo 'Hello from InnoTone!'", check=False)
        print(f"Command: echo 'Hello from InnoTone!'")
        print(f"Exit Code: {result.return_code}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Output: {result.stdout}")
        test_results.append(("Simple Command", result.success))

        # TEST 4: System Information Commands
        print("\n[TEST 4] 📊 Testing System Information Commands...")
        commands = [
            ("Hostname", "hostname"),
            ("Uptime", "uptime"),
            ("Disk Usage", "df -h /"),
            ("Memory", "free -h"),
        ]
        for name, cmd in commands:
            result = executor.execute_command(cmd, check=False)
            print(f"\n{name}:")
            print(
                f"  {result.stdout[:100]}..."
                if len(result.stdout) > 100
                else f"  {result.stdout}"
            )
            test_results.append((f"System Info - {name}", result.success))

        # TEST 5: Directory Creation
        print("\n[TEST 5] 📁 Testing Directory Creation...")
        test_dir = "~/innotone_test_dir/nested/deep"
        executor.ensure_remote_directory_ssh(test_dir)
        verify = executor.execute_command(
            f"test -d {test_dir} && echo 'exists'", check=False
        )
        success = "exists" in verify.stdout
        print(f"Created directory: {test_dir}")
        print(f"Verification: {'✅ Directory exists' if success else '❌ Failed'}")
        test_results.append(("Directory Creation", success))

        # TEST 6: File Upload
        print("\n[TEST 6] 📤 Testing File Upload...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file from InnoTone Training System\n")
            f.write(f"Uploaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            local_file = f.name

        remote_file = "~/innotone_test_dir/uploaded_file.txt"
        upload_success = executor.upload_file(local_file, remote_file)
        print(f"Local file: {local_file}")
        print(f"Remote file: {remote_file}")
        print(f"Upload: {'✅ Success' if upload_success else '❌ Failed'}")

        # Verify upload
        cat_result = executor.execute_command(f"cat {remote_file}", check=False)
        print(f"File contents:\n{cat_result.stdout}")
        test_results.append(("File Upload", upload_success and cat_result.success))
        os.unlink(local_file)

        # TEST 7: File Download
        print("\n[TEST 7] � Testing File Download...")
        download_path = tempfile.mktemp(suffix=".txt")
        download_success = executor.download_file(remote_file, download_path)
        print(f"Remote file: {remote_file}")
        print(f"Local download path: {download_path}")
        print(f"Download: {'✅ Success' if download_success else '❌ Failed'}")

        if download_success and os.path.exists(download_path):
            with open(download_path, "r") as f:
                content = f.read()
            print(f"Downloaded content:\n{content}")
            os.unlink(download_path)
        test_results.append(("File Download", download_success))

        # TEST 8: Script Execution
        print("\n[TEST 8] 📜 Testing Script Execution...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Script execution test'\n")
            f.write("echo 'Current directory:' $(pwd)\n")
            f.write("echo 'Current user:' $(whoami)\n")
            f.write("exit 0\n")
            script_file = f.name

        script_result = executor.execute_script(
            script_file, check=False, remote_path="~/test_script.sh"
        )
        print(f"Script output:\n{script_result.stdout}")
        print(
            f"Script execution: {'✅ Success' if script_result.success else '❌ Failed'}"
        )
        test_results.append(("Script Execution", script_result.success))
        os.unlink(script_file)

        # TEST 9: AWS CLI Check
        print("\n[TEST 9] ☁️  Testing AWS CLI Availability...")
        aws_installed = executor.check_aws_cli_installed()
        print(f"AWS CLI installed: {'✅ Yes' if aws_installed else '❌ No'}")
        test_results.append(("AWS CLI Check", True))

        # TEST 10: Long-running Command
        print("\n[TEST 10] ⏱️  Testing Long-running Command...")
        sleep_result = executor.execute_command(
            "sleep 2 && echo 'Sleep completed'", check=False
        )
        print(f"Command duration: {sleep_result.duration:.2f}s")
        print(f"Output: {sleep_result.stdout}")
        test_results.append(("Long Command", sleep_result.success))

        # TEST 11: Error Handling (Failed Command)
        print("\n[TEST 11] ⚠️  Testing Error Handling...")
        error_result = executor.execute_command(
            "ls /nonexistent_directory_12345", check=False
        )
        print(f"Expected failure - Exit code: {error_result.return_code}")
        print(f"Error message: {error_result.stderr[:100]}")
        test_results.append(("Error Handling", not error_result.success))

        # TEST 12: AWS Functionality (Comprehensive)
        print("\n[TEST 12] ☁️  Running AWS Functionality Tests...")
        aws_results = test_aws_functionality(executor)
        test_results.append(("AWS CLI Installed", aws_results["aws_cli_installed"]))
        test_results.append(
            ("AWS Credentials Valid", aws_results["aws_credentials_configured"])
        )
        test_results.append(("AWS S3 Access", aws_results["s3_list_success"]))

        # TEST 14: Cleanup
        print("\n[TEST 14] 🧹 Testing Cleanup...")
        cleanup_result = executor.execute_command(
            "rm -rf ~/innotone_test_dir ~/test_script.sh", check=False
        )
        print(f"Cleanup: {'✅ Success' if cleanup_result.success else '❌ Failed'}")
        test_results.append(("Cleanup", cleanup_result.success))

        # TEST 15: Disconnect
        print("\n[TEST 15] 🔌 Testing Disconnection...")
        executor.disconnect()
        is_connected_after = executor.is_connected()
        print(
            f"Disconnected: {'✅ Success' if not is_connected_after else '❌ Still connected'}"
        )
        test_results.append(("Disconnection", not is_connected_after))

        # Summary
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, success in test_results if success)
        total = len(test_results)

        for test_name, success in test_results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status:12} - {test_name}")

        print("=" * 70)
        print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print("=" * 70)

        if passed == total:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
