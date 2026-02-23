"""
Simple script to run training on a GPU instance that already has data and scripts.

Usage:
    python simple_gpu_run.py <GPU_IP_ADDRESS>
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from scripts.ssh_executor import SshExecutor


async def run_training(gpu_ip: str):
    """Execute training script on GPU instance."""
    
    # Connect to GPU
    print(f"🔌 Connecting to {gpu_ip}...")
    ssh = SshExecutor(ip=gpu_ip, username="ubuntu")
    await asyncio.to_thread(ssh.connect)
    print("✅ Connected")
    
    # Make script executable and run it
    script_name = "run_docker_job.sh"
    
    print(f"🚀 Starting {script_name}...")
    await asyncio.to_thread(
        ssh.execute_command,
        f"chmod +x {script_name}",
        check=False
    )
    
    result = await asyncio.to_thread(
        ssh.execute_command,
        f"bash -c 'nohup bash {script_name} > {script_name}.log 2>&1 & disown'",
        False,
        False
    )
    
    if not result.success:
        print(f"❌ Failed: {result.stderr}")
        return
    
    # Verify it's running
    await asyncio.sleep(2)
    check = await asyncio.to_thread(
        ssh.execute_command,
        f"pgrep -f 'bash.*{script_name}'",
        check=False
    )
    
    if check.success and check.stdout.strip():
        print(f"✅ Training started with PID: {check.stdout.strip()}")
    else:
        print("❌ Script not running")
    
    await asyncio.to_thread(ssh.disconnect)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_gpu_run.py <GPU_IP_ADDRESS>")
        sys.exit(1)
    
    load_dotenv()
    asyncio.run(run_training(sys.argv[1]))
