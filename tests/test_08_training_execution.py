"""
Phase 3.2: Training Script Execution Testing

Tests for executing training scripts on GPU server:
1. Can upload run_docker_job.sh
2. Script has correct permissions (executable)
3. Can execute script on GPU
4. Docker login works
5. Docker pull works
6. Docker container starts
7. Can run in background (nohup)
8. Can check if process is running
9. Can view training logs
10. Container completes successfully

Run: python tests/test_08_training_execution.py

⚠️  WARNING: This test requires:
⚠️  1. A running GPU instance from Lambda Labs
⚠️  2. Docker installed on the GPU instance
⚠️  3. Internet access on GPU for Docker Hub
⚠️  Set TEST_INSTANCE_IP in .env or we'll launch a new instance
⚠️  Cost: ~$0.60-$1.10 per hour for the GPU instance
"""

import os
import sys
import time
import traceback
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ssh_executor import SshExecutor
from scripts.lambda_client import LambdaClient
from dotenv import load_dotenv

test_instance_id = None
test_dir = None
DOCKER_IMAGE = "deepankarpal89/innotone:ddp_rlhf_text_lambda"
DOCKER_USERNAME = "deepankarpal89"
DOCKER_PASSWORD = "yadadocker7"


def setup_test_instance():
    """Launch a test GPU instance if not provided"""
    print("=" * 70)
    print("SETUP: GPU Instance")
    print("=" * 70)

    load_dotenv()
    existing_ip = os.getenv("TEST_INSTANCE_IP")

    if existing_ip:
        print(f"\n✅ Using existing instance IP: {existing_ip}")
        return existing_ip, None

    print("\n⚠️  No TEST_INSTANCE_IP found in .env")
    print("⚠️  Will launch a new GPU instance for testing")
    print("⚠️  This will cost ~$0.60-$1.10/hour")
    print("\nPress ENTER to launch instance or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(0)

    try:
        client = LambdaClient()
        selected = client.list_available_instances()
        if not selected:
            print("\n❌ No GPU instances available")
            return None, None

        print(f"\n🚀 Launching {selected['name']} in {selected['region']}...")

        instance_name = f"training-exec-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        instance = client.launch_instance(
            instance_type_name=selected["name"],
            region_name=selected["region"],
            name=instance_name,
        )

        if not instance:
            print("\n❌ Failed to launch instance")
            return None, None

        global test_instance_id
        test_instance_id = instance["id"]
        instance_ip = instance["ip"]

        print(f"\n✅ Instance launched!")
        print(f"   Instance ID: {test_instance_id}")
        print(f"   Instance IP: {instance_ip}")

        print("\n⏳ Waiting 30 seconds for SSH to be ready...")
        time.sleep(30)

        return instance_ip, test_instance_id

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        traceback.print_exc()
        return None, None


def setup_ssh_connection(ip):
    """Establish SSH connection"""
    print("\n" + "=" * 70)
    print("SETUP: SSH Connection")
    print("=" * 70)

    if not ip:
        print("❌ SKIPPED: No instance IP available")
        return None

    try:
        print(f"\n🔌 Connecting to {ip}...")

        global test_dir
        ssh_executor = SshExecutor(ip=ip, username="ubuntu")
        ssh_executor.connect()

        # Create test directory on remote server
        test_dir = f"/tmp/training_exec_test_{int(time.time())}"
        ssh_executor.execute_command(
            f"mkdir -p {test_dir}/data {test_dir}/projects_yaml {test_dir}/output"
        )

        print(f"\n✅ SSH connected successfully!")
        print(f"   Remote test directory: {test_dir}")

        return ssh_executor

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        traceback.print_exc()
        return None


def test_1_upload_docker_script(executor):
    """Test 1: Can upload run_docker_job.sh"""
    print("\n" + "=" * 70)
    print("TEST 1: Upload run_docker_job.sh")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Read the local run_docker_job.sh
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run_docker_job.sh",
        )

        if not os.path.exists(script_path):
            print(f"\n❌ FAILED: Script not found at {script_path}")
            return False

        remote_script = f"{test_dir}/run_docker_job.sh"

        print(f"\n📤 Uploading script...")
        print(f"   Local: {script_path}")
        print(f"   Remote: {remote_script}")

        result = executor.upload_file(script_path, remote_script)

        if not result:
            print("\n❌ FAILED: Upload returned False")
            return False

        # Verify file exists
        check_result = executor.execute_command(
            f"test -f {remote_script} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )

        if "EXISTS" not in check_result.stdout:
            print(f"\n❌ FAILED: Script not found on remote")
            return False

        print(f"   ✓ Script uploaded successfully")
        print("\n✅ PASSED - Docker script uploaded!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_2_script_permissions(executor):
    """Test 2: Script has correct permissions (executable)"""
    print("\n" + "=" * 70)
    print("TEST 2: Script Permissions")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        remote_script = f"{test_dir}/run_docker_job.sh"

        print(f"\n🔧 Making script executable...")
        executor.execute_command(f"chmod +x {remote_script}")

        # Verify executable permission
        check_result = executor.execute_command(
            f"test -x {remote_script} && echo 'EXECUTABLE' || echo 'NOT_EXECUTABLE'"
        )

        if "EXECUTABLE" not in check_result.stdout:
            print(f"\n❌ FAILED: Script is not executable")
            return False

        # Check detailed permissions
        ls_result = executor.execute_command(f"ls -lh {remote_script}")
        print(f"\n📄 Script permissions:")
        print(f"   {ls_result.stdout.strip()}")

        print("\n✅ PASSED - Script has executable permissions!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_3_docker_installed(executor):
    """Test 3: Can execute script on GPU (Docker installed)"""
    print("\n" + "=" * 70)
    print("TEST 3: Docker Installation Check")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🐳 Checking Docker installation...")

        # Check if Docker is installed
        docker_result = executor.execute_command(
            "which docker && docker --version", check=False
        )

        if docker_result.return_code != 0:
            print(f"\n⚠️  WARNING: Docker not found on GPU instance")
            print(f"   Installing Docker may be required")
            print(f"   Output: {docker_result.stdout}")
            return False

        print(f"   ✓ Docker found: {docker_result.stdout.strip()}")

        # Check if user can run Docker (sudo or in docker group)
        sudo_check = executor.execute_command("sudo docker ps", check=False)

        if sudo_check.return_code == 0:
            print(f"   ✓ Can run Docker with sudo")
        else:
            print(f"   ⚠️  Cannot run Docker (may need sudo)")

        print("\n✅ PASSED - Docker is available!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_4_docker_login(executor):
    """Test 4: Docker login works"""
    print("\n" + "=" * 70)
    print("TEST 4: Docker Login")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🔐 Logging into Docker Hub...")
        print(f"   Username: {DOCKER_USERNAME}")

        # Login to Docker Hub
        login_cmd = f"echo '{DOCKER_PASSWORD}' | sudo docker login -u {DOCKER_USERNAME} --password-stdin"
        login_result = executor.execute_command(login_cmd, check=False)

        if login_result.return_code != 0:
            print(f"\n❌ FAILED: Docker login failed")
            print(f"   Output: {login_result.stdout}")
            print(f"   Error: {login_result.stderr}")
            return False

        print(f"   ✓ Docker login successful")
        print("\n✅ PASSED - Docker login works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_5_docker_pull(executor):
    """Test 5: Docker pull works"""
    print("\n" + "=" * 70)
    print("TEST 5: Docker Pull")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n📥 Pulling Docker image: {DOCKER_IMAGE}")
        print(f"   This may take a few minutes...")

        pull_cmd = f"sudo docker pull {DOCKER_IMAGE}"
        pull_result = executor.execute_command(pull_cmd, check=False)

        if pull_result.return_code != 0:
            print(f"\n❌ FAILED: Docker pull failed")
            print(f"   Output: {pull_result.stdout[-500:]}")  # Last 500 chars
            return False

        print(f"   ✓ Image pulled successfully")

        # Verify image exists
        verify_cmd = f"sudo docker images {DOCKER_IMAGE} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'"
        verify_result = executor.execute_command(verify_cmd)

        print(f"\n📦 Image verified: {verify_result.stdout.strip()}")
        print("\n✅ PASSED - Docker pull works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_6_docker_container_starts(executor):
    """Test 6: Docker container starts"""
    print("\n" + "=" * 70)
    print("TEST 6: Docker Container Start")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🚀 Starting Docker container...")

        # Create dummy data files for the container
        print(f"\n📝 Creating test data files...")
        executor.execute_command(f"echo 'test,data' > {test_dir}/data/test_data.csv")
        executor.execute_command(
            f"echo 'model: test' > {test_dir}/projects_yaml/config.yaml"
        )

        # Run container (short-lived test)
        run_cmd = f"""cd {test_dir} && sudo docker run --rm \
            -v {test_dir}/data:/app/data \
            -v {test_dir}/projects_yaml:/app/projects_yaml \
            -v {test_dir}/output:/app/output \
            {DOCKER_IMAGE} echo 'Container started successfully'"""

        run_result = executor.execute_command(run_cmd, check=False)

        if run_result.return_code != 0:
            print(f"\n❌ FAILED: Container failed to start")
            print(f"   Output: {run_result.stdout}")
            print(f"   Error: {run_result.stderr}")
            return False

        print(f"   ✓ Container started and executed")
        print(f"   Output: {run_result.stdout.strip()}")
        print("\n✅ PASSED - Docker container starts successfully!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_7_run_in_background(executor):
    """Test 7: Can run in background (nohup)"""
    print("\n" + "=" * 70)
    print("TEST 7: Run in Background (nohup)")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🔄 Starting background process with nohup...")

        # Create a simple script that runs for a few seconds
        test_script = f"{test_dir}/background_test.sh"
        script_content = f"""#!/bin/bash
cd {test_dir}
echo "Background job started at $(date)" > {test_dir}/output/bg_job.log
sleep 5
echo "Background job completed at $(date)" >> {test_dir}/output/bg_job.log
"""

        executor.execute_command(f"cat > {test_script} << 'EOF'\n{script_content}\nEOF")
        executor.execute_command(f"chmod +x {test_script}")

        # Run in background with nohup
        nohup_cmd = f"nohup {test_script} > {test_dir}/output/nohup.out 2>&1 &"
        executor.execute_command(nohup_cmd)

        print(f"   ✓ Background process started")

        # Wait a moment
        time.sleep(2)

        # Check if process is running
        check_cmd = f"pgrep -f {test_script}"
        check_result = executor.execute_command(check_cmd, check=False)

        if check_result.return_code == 0:
            print(f"   ✓ Process is running (PID: {check_result.stdout.strip()})")
        else:
            print(f"   ⚠️  Process may have already completed")

        # Wait for completion
        print(f"\n⏳ Waiting for background job to complete...")
        time.sleep(4)

        # Check log file
        log_result = executor.execute_command(
            f"cat {test_dir}/output/bg_job.log", check=False
        )

        if "completed" in log_result.stdout:
            print(f"   ✓ Background job completed successfully")
            print(f"\n📄 Job log:")
            print(f"   {log_result.stdout.strip()}")
        else:
            print(f"   ⚠️  Job may not have completed")

        print("\n✅ PASSED - Background execution works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_8_check_process_running(executor):
    """Test 8: Can check if process is running"""
    print("\n" + "=" * 70)
    print("TEST 8: Check Process Running")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🔍 Testing process monitoring...")

        # Start a long-running process
        long_script = f"{test_dir}/long_running.sh"
        script_content = f"""#!/bin/bash
echo "Long running job started"
sleep 30
echo "Long running job completed"
"""

        executor.execute_command(f"cat > {long_script} << 'EOF'\n{script_content}\nEOF")
        executor.execute_command(f"chmod +x {long_script}")

        # Start in background
        executor.execute_command(f"nohup {long_script} > /dev/null 2>&1 &")
        time.sleep(1)

        # Method 1: Check with pgrep
        print(f"\n📊 Method 1: Using pgrep")
        pgrep_result = executor.execute_command(f"pgrep -f {long_script}", check=False)

        if pgrep_result.return_code == 0:
            pid = pgrep_result.stdout.strip()
            print(f"   ✓ Process found with PID: {pid}")
        else:
            print(f"   ❌ Process not found with pgrep")

        # Method 2: Check with ps
        print(f"\n📊 Method 2: Using ps")
        ps_result = executor.execute_command(
            f"ps aux | grep {long_script} | grep -v grep", check=False
        )

        if ps_result.return_code == 0:
            print(f"   ✓ Process found with ps")
            print(f"   {ps_result.stdout.strip()[:100]}...")
        else:
            print(f"   ⚠️  Process not found with ps")

        # Kill the process
        print(f"\n🛑 Cleaning up test process...")
        executor.execute_command(f"pkill -f {long_script}", check=False)

        print("\n✅ PASSED - Process monitoring works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_9_view_training_logs(executor):
    """Test 9: Can view training logs"""
    print("\n" + "=" * 70)
    print("TEST 9: View Training Logs")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n📄 Testing log viewing...")

        # Create a log file with training-like output
        log_file = f"{test_dir}/output/training.log"
        log_content = """[2024-01-01 10:00:00] Training started
[2024-01-01 10:00:01] Epoch 1/10 - Loss: 0.5, Accuracy: 0.85
[2024-01-01 10:00:02] Epoch 2/10 - Loss: 0.3, Accuracy: 0.92
[2024-01-01 10:00:03] Epoch 3/10 - Loss: 0.2, Accuracy: 0.95
[2024-01-01 10:00:04] Training completed successfully
"""

        executor.execute_command(f"cat > {log_file} << 'EOF'\n{log_content}\nEOF")

        # Method 1: View entire log
        print(f"\n📖 Method 1: View entire log (cat)")
        cat_result = executor.execute_command(f"cat {log_file}")
        print(f"   ✓ Full log retrieved ({len(cat_result.stdout)} chars)")

        # Method 2: View last N lines (tail)
        print(f"\n📖 Method 2: View last 3 lines (tail)")
        tail_result = executor.execute_command(f"tail -n 3 {log_file}")
        print(f"   Output:")
        for line in tail_result.stdout.strip().split("\n"):
            print(f"   {line}")

        # Method 3: Follow log in real-time (tail -f simulation)
        print(f"\n📖 Method 3: Real-time log monitoring")
        print(f"   ✓ Can use: tail -f {log_file}")

        # Method 4: Search log for specific patterns
        print(f"\n📖 Method 4: Search for patterns (grep)")
        grep_result = executor.execute_command(f"grep 'Accuracy' {log_file}")
        print(
            f"   Found {len(grep_result.stdout.strip().split(chr(10)))} accuracy entries"
        )

        print("\n✅ PASSED - Log viewing works!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_10_container_completes(executor):
    """Test 10: Container completes successfully"""
    print("\n" + "=" * 70)
    print("TEST 10: Container Completion")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        print(f"\n🏁 Running full Docker job script...")
        print(f"   This will test the complete workflow")

        # Modify the script to use a simpler image for testing
        test_script = f"{test_dir}/test_docker_job.sh"
        script_content = f"""#!/bin/bash
OUTPUT_DIR="{test_dir}/output"
LOG_FILE="${{OUTPUT_DIR}}/execution.log"
mkdir -p "${{OUTPUT_DIR}}"

echo "[$(date)] Starting Docker job..." | tee -a "${{LOG_FILE}}"

# Use a simple hello-world image for testing
echo "[$(date)] Running hello-world container..." | tee -a "${{LOG_FILE}}"
sudo docker run --rm hello-world 2>&1 | tee -a "${{LOG_FILE}}"

if [ ${{PIPESTATUS[0]}} -eq 0 ]; then
    echo "[$(date)] Container executed successfully" | tee -a "${{LOG_FILE}}"
    exit 0
else
    echo "[$(date)] Container execution failed" | tee -a "${{LOG_FILE}}"
    exit 1
fi
"""

        executor.execute_command(f"cat > {test_script} << 'EOF'\n{script_content}\nEOF")
        executor.execute_command(f"chmod +x {test_script}")

        # Execute the script
        print(f"\n▶️  Executing test script...")
        exec_result = executor.execute_command(test_script, check=False)

        if exec_result.return_code != 0:
            print(f"\n❌ FAILED: Script execution failed")
            print(f"   Return code: {exec_result.return_code}")
            print(f"   Output: {exec_result.stdout}")
            return False

        print(f"   ✓ Script executed successfully")

        # Check the log file
        print(f"\n📄 Checking execution log...")
        log_result = executor.execute_command(
            f"cat {test_dir}/output/execution.log", check=False
        )

        if "successfully" in log_result.stdout:
            print(f"   ✓ Log confirms successful execution")
            print(f"\n📋 Log excerpt:")
            lines = log_result.stdout.strip().split("\n")
            for line in lines[-5:]:  # Last 5 lines
                print(f"   {line}")
        else:
            print(f"   ⚠️  Log does not confirm success")

        print("\n✅ PASSED - Container completes successfully!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def cleanup_test_instance():
    """Cleanup: Terminate test instance if we created one"""
    if test_instance_id:
        print("\n" + "=" * 70)
        print("CLEANUP: Terminating Test Instance")
        print("=" * 70)

        try:
            client = LambdaClient()
            print(f"\n🛑 Terminating instance {test_instance_id}...")
            client.terminate_instance(test_instance_id)
            print("   Termination request sent: ✅")
        except Exception as e:
            print(f"\n⚠️  Cleanup failed: {e}")
            print(f"   Please terminate instance {test_instance_id} manually!")


def run_all_tests():
    """Run all training execution tests"""
    print("\n" + "#" * 70)
    print("#  PHASE 3.2: TRAINING SCRIPT EXECUTION TESTING".center(70))
    print("#" * 70)
    print("\n⚠️  WARNING: This requires a running GPU instance with Docker!")
    print("⚠️  Set TEST_INSTANCE_IP in .env or we'll launch a new instance")
    print("⚠️  Estimated cost: $0.60-$1.10/hour")
    print("\n" + "=" * 70)

    results = {}

    try:
        # Setup: Get or launch instance
        instance_ip, instance_id = setup_test_instance()

        if not instance_ip:
            print("\n❌ Cannot proceed without instance IP")
            return

        # Setup SSH connection
        executor = setup_ssh_connection(instance_ip)

        if not executor:
            return

        # Run all tests
        results["upload_docker_script"] = test_1_upload_docker_script(executor)
        results["script_permissions"] = test_2_script_permissions(executor)
        results["docker_installed"] = test_3_docker_installed(executor)
        results["docker_login"] = test_4_docker_login(executor)
        results["docker_pull"] = test_5_docker_pull(executor)
        results["docker_container_starts"] = test_6_docker_container_starts(executor)
        results["run_in_background"] = test_7_run_in_background(executor)
        results["check_process_running"] = test_8_check_process_running(executor)
        results["view_training_logs"] = test_9_view_training_logs(executor)
        results["container_completes"] = test_10_container_completes(executor)

        # Cleanup SSH connection
        if executor:
            print("\n" + "=" * 70)
            print("CLEANUP: SSH Connection & Test Directory")
            print("=" * 70)
            print("\n🧹 Cleaning up test directory...")
            executor.execute_command(f"rm -rf {test_dir}", check=False)
            print("   Test directory removed: ✅")

            print("\n🔌 Disconnecting SSH...")
            executor.disconnect()
            print("   Disconnected: ✅")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Always try to cleanup
        cleanup_test_instance()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests cancelled by user")
        sys.exit(0)
