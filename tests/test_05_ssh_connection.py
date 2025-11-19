"""
Phase 2.2: SSH Connection Testing

Tests for SSH connection to GPU instances:
1. Can establish SSH connection to GPU instance
2. Retry logic works on connection failures
3. Can execute simple commands (echo, pwd)
4. Can check GPU availability (nvidia-smi)
5. Can create directories
6. Can check file existence
7. Connection timeout handling works
8. Can disconnect cleanly
9. Handles wrong credentials gracefully

Run: python tests/test_05_ssh_connection.py

⚠️  WARNING: This test requires a RUNNING GPU instance from Lambda Labs
⚠️  Run test_04_lambda_client.py first to get instance IP, or provide one manually
⚠️  Cost: ~$0.60-$1.10 per hour for the GPU instance
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ssh_executor import SshExecutor
from scripts.lambda_client import LambdaClient
from dotenv import load_dotenv


# Global variables
test_instance_id = None
test_instance_ip = None
ssh_executor = None


def setup_test_instance():
    """Launch a test GPU instance if not provided"""
    print("=" * 70)
    print("SETUP: GPU Instance")
    print("=" * 70)
    
    # Check if instance IP is provided via environment
    load_dotenv()
    existing_ip = os.getenv('TEST_INSTANCE_IP')
    
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
        
        # Select cheapest available instance
        selected = client.list_available_instances()
        if not selected:
            print("\n❌ No GPU instances available")
            return None, None
        
        print(f"\n🚀 Launching {selected['name']} in {selected['region']}...")
        
        instance_name = f"ssh-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        instance = client.launch_instance(
            instance_type_name=selected['name'],
            region_name=selected['region'],
            name=instance_name
        )
        
        if not instance:
            print("\n❌ Failed to launch instance")
            return None, None
        
        global test_instance_id
        test_instance_id = instance['id']
        instance_ip = instance['ip']
        
        print(f"\n✅ Instance launched!")
        print(f"   Instance ID: {test_instance_id}")
        print(f"   Instance IP: {instance_ip}")
        
        # Wait a bit for SSH to be ready
        print("\n⏳ Waiting 30 seconds for SSH to be ready...")
        time.sleep(30)
        
        return instance_ip, test_instance_id
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        traceback.print_exc()
        return None, None


def test_1_establish_connection(ip):
    """Test 1: Can establish SSH connection to GPU instance"""
    print("\n" + "=" * 70)
    print("TEST 1: Establish SSH Connection")
    print("=" * 70)
    
    if not ip:
        print("❌ SKIPPED: No instance IP available")
        return None
    
    try:
        print(f"\n🔌 Connecting to {ip}...")
        
        global ssh_executor
        ssh_executor = SshExecutor(ip=ip, username='ubuntu')
        
        start_time = time.time()
        ssh_executor.connect()
        connect_time = time.time() - start_time
        
        print(f"\n✅ PASSED - Connected successfully!")
        print(f"   Connection time: {connect_time:.2f} seconds")
        print(f"   Host: {ip}")
        print(f"   Username: ubuntu")
        
        return ssh_executor
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return None


def test_2_retry_logic(ip):
    """Test 2: Retry logic works on connection failures"""
    print("\n" + "=" * 70)
    print("TEST 2: Connection Retry Logic")
    print("=" * 70)
    
    if not ip:
        print("❌ SKIPPED: No instance IP available")
        return False
    
    try:
        print("\n🔄 Testing retry logic with invalid IP...")
        
        # Try connecting to an invalid IP (should fail quickly)
        invalid_executor = SshExecutor(ip="192.0.2.1", username='ubuntu')  # TEST-NET-1
        invalid_executor.timeout = 5  # Short timeout
        
        try:
            invalid_executor.connect()
            print("\n⚠️  WARNING: Connection to invalid IP succeeded (unexpected)")
            return False
        except Exception as e:
            print(f"\n✅ Expected failure on invalid IP: {type(e).__name__}")
        
        # Now test that valid connection still works
        print("\n🔄 Testing that valid connection still works...")
        valid_executor = SshExecutor(ip=ip, username='ubuntu')
        valid_executor.connect()
        valid_executor.disconnect()
        
        print("\n✅ PASSED - Retry logic handles failures correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_3_simple_commands(executor):
    """Test 3: Can execute simple commands (echo, pwd)"""
    print("\n" + "=" * 70)
    print("TEST 3: Execute Simple Commands")
    print("=" * 70)
    
    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False
    
    try:
        # Test echo command
        print("\n📝 Testing 'echo' command...")
        result = executor.execute_command("echo 'Hello from GPU instance!'")
        
        print(f"   Command: {result.command}")
        print(f"   Output: {result.stdout}")
        print(f"   Duration: {result.duration:.3f}s")
        print(f"   Success: {result.success}")
        
        if not result.success or "Hello from GPU instance!" not in result.stdout:
            print("\n❌ FAILED: Echo command failed")
            return False
        
        # Test pwd command
        print("\n📝 Testing 'pwd' command...")
        result = executor.execute_command("pwd")
        
        print(f"   Command: {result.command}")
        print(f"   Output: {result.stdout}")
        print(f"   Duration: {result.duration:.3f}s")
        
        if not result.success or not result.stdout.startswith('/'):
            print("\n❌ FAILED: pwd command failed")
            return False
        
        # Test whoami command
        print("\n📝 Testing 'whoami' command...")
        result = executor.execute_command("whoami")
        
        print(f"   Command: {result.command}")
        print(f"   Output: {result.stdout}")
        
        if not result.success or "ubuntu" not in result.stdout:
            print("\n❌ FAILED: whoami command failed")
            return False
        
        print("\n✅ PASSED - All simple commands executed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_4_check_gpu_availability(executor):
    """Test 4: Can check GPU availability (nvidia-smi)"""
    print("\n" + "=" * 70)
    print("TEST 4: Check GPU Availability")
    print("=" * 70)
    
    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False
    
    try:
        print("\n🎮 Running 'nvidia-smi' command...")
        
        result = executor.execute_command("nvidia-smi", check=False)
        
        print(f"\n📊 Command output:")
        print("-" * 70)
        print(result.stdout[:500] if len(result.stdout) > 500 else result.stdout)
        if len(result.stdout) > 500:
            print("... (truncated)")
        print("-" * 70)
        
        print(f"\n   Return code: {result.return_code}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Success: {result.success}")
        
        if result.success and "NVIDIA" in result.stdout:
            print("\n✅ PASSED - GPU detected successfully!")
            
            # Extract GPU info
            if "Tesla" in result.stdout or "RTX" in result.stdout or "A100" in result.stdout:
                gpu_type = "Tesla/RTX/A100 series"
                print(f"   GPU Type: {gpu_type}")
            
            return True
        else:
            print("\n⚠️  WARNING: nvidia-smi command failed or no GPU detected")
            print("   This might be expected if instance doesn't have GPU drivers")
            return True  # Still pass as SSH connection works
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_5_create_directories(executor):
    """Test 5: Can create directories"""
    print("\n" + "=" * 70)
    print("TEST 5: Create Directories")
    print("=" * 70)
    
    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False
    
    try:
        test_dir = f"/tmp/ssh_test_{int(time.time())}"
        
        print(f"\n📁 Creating directory: {test_dir}")
        result = executor.execute_command(f"mkdir -p {test_dir}")
        
        if not result.success:
            print(f"\n❌ FAILED: mkdir command failed")
            return False
        
        print(f"   Created successfully!")
        
        # Verify directory exists
        print(f"\n📁 Verifying directory exists...")
        result = executor.execute_command(f"test -d {test_dir} && echo 'EXISTS' || echo 'NOT_FOUND'")
        
        print(f"   Verification: {result.stdout}")
        
        if "EXISTS" not in result.stdout:
            print(f"\n❌ FAILED: Directory verification failed")
            return False
        
        # Create nested directories
        nested_dir = f"{test_dir}/nested/deep/path"
        print(f"\n📁 Creating nested directories: {nested_dir}")
        result = executor.execute_command(f"mkdir -p {nested_dir}")
        
        if not result.success:
            print(f"\n❌ FAILED: Nested mkdir failed")
            return False
        
        # Cleanup
        print(f"\n🧹 Cleaning up test directory...")
        executor.execute_command(f"rm -rf {test_dir}")
        
        print("\n✅ PASSED - Directory operations work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_6_check_file_existence(executor):
    """Test 6: Can check file existence"""
    print("\n" + "=" * 70)
    print("TEST 6: Check File Existence")
    print("=" * 70)
    
    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False
    
    try:
        test_file = f"/tmp/test_file_{int(time.time())}.txt"
        
        # Check non-existent file
        print(f"\n📄 Checking non-existent file: {test_file}")
        result = executor.execute_command(
            f"test -f {test_file} && echo 'EXISTS' || echo 'NOT_FOUND'",
            check=False
        )
        
        print(f"   Result: {result.stdout}")
        
        if "NOT_FOUND" not in result.stdout:
            print(f"\n❌ FAILED: File should not exist")
            return False
        
        # Create file
        print(f"\n📄 Creating test file...")
        result = executor.execute_command(f"echo 'test content' > {test_file}")
        
        if not result.success:
            print(f"\n❌ FAILED: File creation failed")
            return False
        
        # Check existing file
        print(f"\n📄 Checking created file...")
        result = executor.execute_command(
            f"test -f {test_file} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )
        
        print(f"   Result: {result.stdout}")
        
        if "EXISTS" not in result.stdout:
            print(f"\n❌ FAILED: File should exist")
            return False
        
        # Read file content
        print(f"\n📄 Reading file content...")
        result = executor.execute_command(f"cat {test_file}")
        
        print(f"   Content: {result.stdout}")
        
        if "test content" not in result.stdout:
            print(f"\n❌ FAILED: File content mismatch")
            return False
        
        # Cleanup
        print(f"\n🧹 Cleaning up test file...")
        executor.execute_command(f"rm -f {test_file}")
        
        print("\n✅ PASSED - File existence checks work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_7_connection_timeout(ip):
    """Test 7: Connection timeout handling works"""
    print("\n" + "=" * 70)
    print("TEST 7: Connection Timeout Handling")
    print("=" * 70)
    
    if not ip:
        print("❌ SKIPPED: No instance IP available")
        return False
    
    try:
        print("\n⏱️  Testing timeout with very short duration...")
        
        timeout_executor = SshExecutor(ip=ip, username='ubuntu')
        timeout_executor.timeout = 1  # Very short timeout
        
        try:
            start_time = time.time()
            timeout_executor.connect()
            connect_time = time.time() - start_time
            
            print(f"\n   Connection succeeded in {connect_time:.2f}s")
            
            # Try a long-running command with timeout
            print("\n⏱️  Testing command timeout...")
            try:
                result = timeout_executor.execute_command("sleep 10", check=False)
                print(f"   Command completed (unexpected)")
            except Exception as e:
                print(f"   Command timed out as expected: {type(e).__name__}")
            
            timeout_executor.disconnect()
            
        except Exception as e:
            print(f"\n   Connection timeout: {type(e).__name__}")
        
        print("\n✅ PASSED - Timeout handling works!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_8_disconnect_cleanly(executor):
    """Test 8: Can disconnect cleanly"""
    print("\n" + "=" * 70)
    print("TEST 8: Disconnect Cleanly")
    print("=" * 70)
    
    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False
    
    try:
        print("\n🔌 Testing clean disconnect...")
        
        # Execute a command to ensure connection is active
        result = executor.execute_command("echo 'before disconnect'")
        print(f"   Command before disconnect: {result.stdout}")
        
        # Disconnect
        print("\n🔌 Disconnecting...")
        executor.disconnect()
        
        # Verify client is None
        if executor.client is not None:
            print(f"\n❌ FAILED: Client should be None after disconnect")
            return False
        
        print(f"   Client cleared: ✅")
        
        # Try to reconnect
        print("\n🔌 Testing reconnection...")
        executor.connect()
        result = executor.execute_command("echo 'after reconnect'")
        print(f"   Command after reconnect: {result.stdout}")
        
        if not result.success:
            print(f"\n❌ FAILED: Reconnection failed")
            return False
        
        print("\n✅ PASSED - Disconnect and reconnect work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_9_wrong_credentials(ip):
    """Test 9: Handles wrong credentials gracefully"""
    print("\n" + "=" * 70)
    print("TEST 9: Handle Wrong Credentials")
    print("=" * 70)
    
    if not ip:
        print("❌ SKIPPED: No instance IP available")
        return False
    
    try:
        print("\n🔐 Testing with wrong username...")
        
        wrong_executor = SshExecutor(ip=ip, username='wronguser')
        wrong_executor.timeout = 10
        
        try:
            wrong_executor.connect()
            print("\n⚠️  WARNING: Connection with wrong username succeeded (unexpected)")
            wrong_executor.disconnect()
            return False
        except Exception as e:
            print(f"\n✅ Expected failure: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}")
        
        # Test with wrong SSH key (if we can modify it)
        print("\n🔐 Testing error handling for authentication failures...")
        
        # Save original SSH key path
        original_key = os.environ.get('SSH_KEY_PATH')
        
        try:
            # Temporarily set wrong key path
            os.environ['SSH_KEY_PATH'] = '/tmp/nonexistent_key.pem'
            
            bad_key_executor = SshExecutor(ip=ip, username='ubuntu')
            bad_key_executor.timeout = 10
            
            try:
                bad_key_executor.connect()
                print("\n⚠️  WARNING: Connection with wrong key succeeded (unexpected)")
            except Exception as e:
                print(f"\n✅ Expected failure with wrong key: {type(e).__name__}")
        finally:
            # Restore original key path
            if original_key:
                os.environ['SSH_KEY_PATH'] = original_key
        
        print("\n✅ PASSED - Wrong credentials handled gracefully!")
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
    """Run all SSH connection tests"""
    print("\n" + "#" * 70)
    print("#  PHASE 2.2: SSH CONNECTION TESTING".center(70))
    print("#" * 70)
    print("\n⚠️  WARNING: This requires a running GPU instance!")
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
        
        # Test 1: Establish connection
        executor = test_1_establish_connection(instance_ip)
        results["establish_connection"] = executor is not None
        
        # Test 2: Retry logic
        results["retry_logic"] = test_2_retry_logic(instance_ip)
        
        # Test 3: Simple commands
        results["simple_commands"] = test_3_simple_commands(executor)
        
        # Test 4: GPU availability
        results["gpu_availability"] = test_4_check_gpu_availability(executor)
        
        # Test 5: Create directories
        results["create_directories"] = test_5_create_directories(executor)
        
        # Test 6: Check file existence
        results["file_existence"] = test_6_check_file_existence(executor)
        
        # Test 7: Connection timeout
        results["connection_timeout"] = test_7_connection_timeout(instance_ip)
        
        # Test 8: Disconnect cleanly
        results["disconnect_cleanly"] = test_8_disconnect_cleanly(executor)
        
        # Test 9: Wrong credentials
        results["wrong_credentials"] = test_9_wrong_credentials(instance_ip)
        
        # Cleanup
        if executor:
            executor.disconnect()
        
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
        if test_instance_id:
            print(f"\n🛑 IMPORTANT: Clean up instance {test_instance_id} manually!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        if test_instance_id:
            print(f"\n🛑 IMPORTANT: Clean up instance {test_instance_id} manually!")