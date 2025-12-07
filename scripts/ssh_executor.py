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
    def __init__(self, ip, username="ubuntu",reuse_conection=True):
        self.ip = ip
        self.username = username
        self.timeout = 120
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

        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_file_path = os.environ.get("SSH_KEY_PATH")
            ssh_file_pwd = os.environ.get("SSH_KEY_PASSWORD")
            private_key = paramiko.RSAKey.from_private_key_file(
                os.path.expanduser(ssh_file_path), password=ssh_file_pwd
            )
            self.client.connect(
                hostname=self.ip,
                username=self.username,
                pkey=private_key,
                timeout=self.timeout,
            )
        except Exception as e:
            raise Exception(f"Failed to connect to {self.ip}: {str(e)}")

    async def connect_async(self):
        """Asynchronous version of connect method"""
        if self.client is not None:
            return
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_file_path = os.environ.get("SSH_KEY_PATH")
            ssh_file_pwd = os.environ.get("SSH_KEY_PASSWORD")
            private_key = paramiko.RSAKey.from_private_key_file(
                os.path.expanduser(ssh_file_path), password=ssh_file_pwd
            )
            await asyncio.to_thread(
                self.client.connect,
                hostname=self.ip,
                username=self.username,
                pkey=private_key,
                timeout=self.timeout,
            )
        except Exception as e:
            raise Exception(f"Failed to connect to {self.ip}: {str(e)}")

    def disconnect(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def execute_command(self, command: str, check=True):
        if self.client is None:
            self.connect()

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
