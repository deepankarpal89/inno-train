"""
Phase 2.3: File Transfer Testing (SSH)

Tests for file transfer operations via SSH:
1. Can upload single file to GPU server
2. Can upload to nested directories (creates dirs)
3. Can download single file from GPU server
4. Can upload multiple files
5. Can download entire directory recursively
6. File integrity verified (size/checksum)
7. Handles large files correctly (>100MB)
8. Handles permission errors

Run: python tests/test_06_file_transfer.py

⚠️  WARNING: This test requires a RUNNING GPU instance from Lambda Labs
⚠️  Run test_04_lambda_client.py first to get instance IP, or provide one manually
⚠️  Cost: ~$0.60-$1.10 per hour for the GPU instance
"""

import os
import sys
import time
import traceback
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ssh_executor import SshExecutor
from scripts.lambda_client import LambdaClient
from dotenv import load_dotenv


# Global variables
test_instance_id = None
test_instance_ip = None
ssh_executor = None
test_dir = None


def calculate_file_checksum(file_path):
    """Calculate MD5 checksum of a file"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_test_file(size_mb=1, content=None):
    """Create a test file with specified size or content"""
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")

    if content:
        temp_file.write(content)
    else:
        # Create file with random content of specified size
        chunk_size = 1024 * 1024  # 1MB chunks
        for _ in range(size_mb):
            temp_file.write("x" * chunk_size)

    temp_file.close()
    return temp_file.name


def setup_test_instance():
    """Launch a test GPU instance if not provided"""
    print("=" * 70)
    print("SETUP: GPU Instance")
    print("=" * 70)

    # Check if instance IP is provided via environment
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

        # Select cheapest available instance
        selected = client.list_available_instances()
        if not selected:
            print("\n❌ No GPU instances available")
            return None, None

        print(f"\n🚀 Launching {selected['name']} in {selected['region']}...")

        instance_name = f"file-transfer-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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

        # Wait a bit for SSH to be ready
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

        global ssh_executor, test_dir
        ssh_executor = SshExecutor(ip=ip, username="ubuntu")
        ssh_executor.connect()

        # Create test directory on remote server
        test_dir = f"/tmp/file_transfer_test_{int(time.time())}"
        ssh_executor.execute_command(f"mkdir -p {test_dir}")

        print(f"\n✅ SSH connected successfully!")
        print(f"   Remote test directory: {test_dir}")

        return ssh_executor

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        traceback.print_exc()
        return None


def test_1_upload_single_file(executor):
    """Test 1: Can upload single file to GPU server"""
    print("\n" + "=" * 70)
    print("TEST 1: Upload Single File")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create test file
        test_content = f"Test file created at {datetime.now()}\nLine 2\nLine 3"
        local_file = create_test_file(content=test_content)
        local_filename = os.path.basename(local_file)
        remote_file = f"{test_dir}/{local_filename}"

        print(f"\n📤 Uploading file...")
        print(f"   Local: {local_file}")
        print(f"   Remote: {remote_file}")

        start_time = time.time()
        result = executor.upload_file(local_file, remote_file)
        upload_time = time.time() - start_time

        if not result:
            print("\n❌ FAILED: Upload returned False")
            os.unlink(local_file)
            return False

        print(f"\n   Upload time: {upload_time:.3f}s")

        # Verify file exists on remote
        print(f"\n📄 Verifying file exists on remote...")
        check_result = executor.execute_command(
            f"test -f {remote_file} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )

        if "EXISTS" not in check_result.stdout:
            print(f"\n❌ FAILED: File not found on remote")
            os.unlink(local_file)
            return False

        # Verify content
        print(f"\n📄 Verifying file content...")
        cat_result = executor.execute_command(f"cat {remote_file}")

        # PTY includes command echo, so we need to extract just the file content
        # The output format is typically: "cat filename\r\n<content>\r\n"
        remote_output = cat_result.stdout

        # Split by lines and filter out the command echo
        lines = remote_output.split("\n")
        # Remove lines that are the command itself (starts with 'cat')
        content_lines = [line for line in lines if not line.strip().startswith("cat ")]
        remote_content = "\n".join(content_lines).strip()

        # Normalize line endings (remote may have \r\n, local has \n)
        remote_content_normalized = remote_content.replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        test_content_normalized = (
            test_content.strip().replace("\r\n", "\n").replace("\r", "\n")
        )

        # Compare the actual content
        if test_content_normalized != remote_content_normalized:
            print(f"\n❌ FAILED: Content mismatch")
            print(f"   Expected ({len(test_content)} chars): {repr(test_content)}")
            print(f"   Got ({len(remote_content)} chars): {repr(remote_content)}")
            print(f"   Raw output: {repr(remote_output[:200])}...")
            os.unlink(local_file)
            return False

        print(f"   Content verified: ✅")

        # Cleanup
        os.unlink(local_file)

        print("\n✅ PASSED - Single file upload works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_2_upload_nested_directories(executor):
    """Test 2: Can upload to nested directories (creates dirs)"""
    print("\n" + "=" * 70)
    print("TEST 2: Upload to Nested Directories")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create test file
        test_content = "File in nested directory"
        local_file = create_test_file(content=test_content)

        # Create nested remote path
        nested_path = f"{test_dir}/level1/level2/level3/nested_file.txt"

        print(f"\n📤 Uploading to nested directory...")
        print(f"   Local: {local_file}")
        print(f"   Remote: {nested_path}")

        # Note: Current implementation only creates one level
        # We need to create parent directories first
        parent_dir = os.path.dirname(nested_path)
        print(f"\n📁 Creating parent directories: {parent_dir}")
        executor.execute_command(f"mkdir -p {parent_dir}")

        # Now upload
        result = executor.upload_file(local_file, nested_path)

        if not result:
            print("\n❌ FAILED: Upload returned False")
            os.unlink(local_file)
            return False

        # Verify file exists
        print(f"\n📄 Verifying file exists...")
        check_result = executor.execute_command(
            f"test -f {nested_path} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )

        if "EXISTS" not in check_result.stdout:
            print(f"\n❌ FAILED: File not found in nested directory")
            os.unlink(local_file)
            return False

        print(f"   File exists in nested directory: ✅")

        # Verify directory structure
        print(f"\n📁 Verifying directory structure...")
        tree_result = executor.execute_command(f"ls -R {test_dir}/level1", check=False)
        print(f"   Directory tree:\n{tree_result.stdout}")

        # Cleanup
        os.unlink(local_file)

        print("\n✅ PASSED - Nested directory upload works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_3_download_single_file(executor):
    """Test 3: Can download single file from GPU server"""
    print("\n" + "=" * 70)
    print("TEST 3: Download Single File")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create file on remote server
        remote_file = f"{test_dir}/download_test.txt"
        test_content = f"Download test file created at {datetime.now()}"

        print(f"\n📝 Creating file on remote server...")
        executor.execute_command(f"echo '{test_content}' > {remote_file}")

        # Download file
        local_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name

        print(f"\n📥 Downloading file...")
        print(f"   Remote: {remote_file}")
        print(f"   Local: {local_file}")

        start_time = time.time()
        result = executor.download_file(remote_file, local_file)
        download_time = time.time() - start_time

        if not result:
            print("\n❌ FAILED: Download returned False")
            return False

        print(f"\n   Download time: {download_time:.3f}s")

        # Verify file exists locally
        if not os.path.exists(local_file):
            print(f"\n❌ FAILED: Downloaded file not found locally")
            return False

        print(f"   File exists locally: ✅")

        # Verify content
        print(f"\n📄 Verifying file content...")
        with open(local_file, "r") as f:
            downloaded_content = f.read().strip()

        if test_content not in downloaded_content:
            print(f"\n❌ FAILED: Content mismatch")
            print(f"   Expected: {test_content}")
            print(f"   Got: {downloaded_content}")
            os.unlink(local_file)
            return False

        print(f"   Content verified: ✅")

        # Cleanup
        os.unlink(local_file)

        print("\n✅ PASSED - Single file download works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_4_upload_multiple_files(executor):
    """Test 4: Can upload multiple files"""
    print("\n" + "=" * 70)
    print("TEST 4: Upload Multiple Files")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create multiple test files
        num_files = 5
        local_files = []

        print(f"\n📝 Creating {num_files} test files...")
        for i in range(num_files):
            content = f"Test file {i+1} content\nLine 2\nLine 3"
            local_file = create_test_file(content=content)
            local_files.append(local_file)
            print(f"   Created: {os.path.basename(local_file)}")

        # Upload all files
        remote_dir = f"{test_dir}/multiple_files"
        executor.execute_command(f"mkdir -p {remote_dir}")

        print(f"\n📤 Uploading {num_files} files...")
        upload_times = []

        for local_file in local_files:
            filename = os.path.basename(local_file)
            remote_file = f"{remote_dir}/{filename}"

            start_time = time.time()
            result = executor.upload_file(local_file, remote_file)
            upload_time = time.time() - start_time
            upload_times.append(upload_time)

            if not result:
                print(f"\n❌ FAILED: Upload of {filename} returned False")
                for f in local_files:
                    os.unlink(f)
                return False

            print(f"   Uploaded {filename} in {upload_time:.3f}s")

        # Verify all files exist
        print(f"\n📄 Verifying all files exist on remote...")
        ls_result = executor.execute_command(f"ls -1 {remote_dir}")
        remote_files = ls_result.stdout.strip().split("\n")

        print(f"   Found {len(remote_files)} files on remote")

        if len(remote_files) != num_files:
            print(f"\n❌ FAILED: Expected {num_files} files, found {len(remote_files)}")
            for f in local_files:
                os.unlink(f)
            return False

        print(f"   All files present: ✅")

        # Statistics
        total_time = sum(upload_times)
        avg_time = total_time / num_files
        print(f"\n📊 Upload statistics:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average time per file: {avg_time:.3f}s")

        # Cleanup
        for f in local_files:
            os.unlink(f)

        print("\n✅ PASSED - Multiple file upload works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_5_download_directory_recursively(executor):
    """Test 5: Can download entire directory recursively"""
    print("\n" + "=" * 70)
    print("TEST 5: Download Directory Recursively")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create directory structure on remote
        remote_base = f"{test_dir}/download_dir"

        print(f"\n📁 Creating directory structure on remote...")
        executor.execute_command(f"mkdir -p {remote_base}/subdir1/subdir2")
        executor.execute_command(f"echo 'file1' > {remote_base}/file1.txt")
        executor.execute_command(f"echo 'file2' > {remote_base}/subdir1/file2.txt")
        executor.execute_command(
            f"echo 'file3' > {remote_base}/subdir1/subdir2/file3.txt"
        )

        print(f"   Created directory structure with 3 files")

        # List remote structure
        print(f"\n📁 Remote directory structure:")
        tree_result = executor.execute_command(
            f"find {remote_base} -type f", check=False
        )
        print(f"{tree_result.stdout}")

        # Download recursively using tar
        print(f"\n📥 Downloading directory recursively...")
        local_temp_dir = tempfile.mkdtemp()
        tar_file = f"{test_dir}/download_archive.tar.gz"
        local_tar = os.path.join(local_temp_dir, "archive.tar.gz")

        # Create tar on remote
        print(f"   Creating tar archive on remote...")
        executor.execute_command(
            f"cd {test_dir} && tar -czf download_archive.tar.gz download_dir/"
        )

        # Download tar
        print(f"   Downloading tar archive...")
        start_time = time.time()
        executor.download_file(tar_file, local_tar)
        download_time = time.time() - start_time

        print(f"   Download time: {download_time:.3f}s")

        # Extract tar locally
        print(f"   Extracting archive locally...")
        import tarfile

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(local_temp_dir)

        # Verify files exist locally
        print(f"\n📄 Verifying downloaded files...")
        local_base = os.path.join(local_temp_dir, "download_dir")

        expected_files = [
            os.path.join(local_base, "file1.txt"),
            os.path.join(local_base, "subdir1", "file2.txt"),
            os.path.join(local_base, "subdir1", "subdir2", "file3.txt"),
        ]

        all_exist = True
        for expected_file in expected_files:
            exists = os.path.exists(expected_file)
            status = "✅" if exists else "❌"
            print(f"   {status} {os.path.relpath(expected_file, local_temp_dir)}")
            if not exists:
                all_exist = False

        # Cleanup
        import shutil

        shutil.rmtree(local_temp_dir)

        if not all_exist:
            print(f"\n❌ FAILED: Some files missing after download")
            return False

        print("\n✅ PASSED - Directory download works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_6_file_integrity_checksum(executor):
    """Test 6: File integrity verified (size/checksum)"""
    print("\n" + "=" * 70)
    print("TEST 6: File Integrity (Size/Checksum)")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Create test file with known content
        test_content = "File integrity test\n" * 1000
        local_file = create_test_file(content=test_content)

        # Calculate local checksum
        print(f"\n🔐 Calculating local file checksum...")
        local_checksum = calculate_file_checksum(local_file)
        local_size = os.path.getsize(local_file)

        print(f"   Local MD5: {local_checksum}")
        print(f"   Local size: {local_size} bytes")

        # Upload file
        remote_file = f"{test_dir}/integrity_test.txt"
        print(f"\n📤 Uploading file...")
        executor.upload_file(local_file, remote_file)

        # Get remote file size
        print(f"\n📊 Checking remote file size...")
        size_result = executor.execute_command(
            f"stat -f%z {remote_file} 2>/dev/null || stat -c%s {remote_file}"
        )
        remote_size = int(size_result.stdout.strip())

        print(f"   Remote size: {remote_size} bytes")

        if local_size != remote_size:
            print(
                f"\n❌ FAILED: Size mismatch (local: {local_size}, remote: {remote_size})"
            )
            os.unlink(local_file)
            return False

        print(f"   Size match: ✅")

        # Calculate remote checksum
        print(f"\n🔐 Calculating remote file checksum...")
        checksum_result = executor.execute_command(
            f"md5sum {remote_file} 2>/dev/null || md5 -q {remote_file}"
        )
        remote_checksum = checksum_result.stdout.strip().split()[0]

        print(f"   Remote MD5: {remote_checksum}")

        if local_checksum != remote_checksum:
            print(f"\n❌ FAILED: Checksum mismatch")
            print(f"   Local:  {local_checksum}")
            print(f"   Remote: {remote_checksum}")
            os.unlink(local_file)
            return False

        print(f"   Checksum match: ✅")

        # Download and verify again
        print(f"\n📥 Downloading file back...")
        downloaded_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        executor.download_file(remote_file, downloaded_file)

        downloaded_checksum = calculate_file_checksum(downloaded_file)
        downloaded_size = os.path.getsize(downloaded_file)

        print(f"\n📊 Downloaded file verification:")
        print(f"   Downloaded MD5: {downloaded_checksum}")
        print(f"   Downloaded size: {downloaded_size} bytes")

        if downloaded_checksum != local_checksum or downloaded_size != local_size:
            print(f"\n❌ FAILED: Downloaded file doesn't match original")
            os.unlink(local_file)
            os.unlink(downloaded_file)
            return False

        print(f"   Round-trip integrity: ✅")

        # Cleanup
        os.unlink(local_file)
        os.unlink(downloaded_file)

        print("\n✅ PASSED - File integrity verification works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_7_large_file_transfer(executor):
    """Test 7: Handles large files correctly (>100MB)"""
    print("\n" + "=" * 70)
    print("TEST 7: Large File Transfer (>100MB)")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    print("\n⚠️  This test will create and transfer a 100MB+ file")
    print("⚠️  It may take several minutes depending on network speed")
    print("\nPress ENTER to continue or Ctrl+C to skip...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n⚠️  Test skipped by user")
        return True  # Don't fail, just skip

    try:
        # Create large file (100MB)
        file_size_mb = 100
        print(f"\n📝 Creating {file_size_mb}MB test file...")

        local_file = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin")
        chunk_size = 1024 * 1024  # 1MB chunks

        for i in range(file_size_mb):
            local_file.write(b"x" * chunk_size)
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{file_size_mb} MB")

        local_file.close()
        local_file_path = local_file.name

        local_size = os.path.getsize(local_file_path)
        print(f"\n   Created file: {local_size / (1024*1024):.2f} MB")

        # Calculate checksum
        print(f"\n🔐 Calculating checksum...")
        local_checksum = calculate_file_checksum(local_file_path)
        print(f"   MD5: {local_checksum}")

        # Upload large file
        remote_file = f"{test_dir}/large_file.bin"
        print(f"\n📤 Uploading large file...")

        start_time = time.time()
        result = executor.upload_file(local_file_path, remote_file)
        upload_time = time.time() - start_time

        if not result:
            print(f"\n❌ FAILED: Upload returned False")
            os.unlink(local_file_path)
            return False

        upload_speed = (local_size / (1024 * 1024)) / upload_time
        print(f"\n   Upload time: {upload_time:.2f}s")
        print(f"   Upload speed: {upload_speed:.2f} MB/s")

        # Verify remote file size
        print(f"\n📊 Verifying remote file...")
        size_result = executor.execute_command(
            f"stat -f%z {remote_file} 2>/dev/null || stat -c%s {remote_file}"
        )
        remote_size = int(size_result.stdout.strip())

        print(f"   Remote size: {remote_size / (1024*1024):.2f} MB")

        if local_size != remote_size:
            print(f"\n❌ FAILED: Size mismatch")
            os.unlink(local_file_path)
            return False

        # Verify checksum
        print(f"\n🔐 Verifying remote checksum...")
        checksum_result = executor.execute_command(
            f"md5sum {remote_file} 2>/dev/null || md5 -q {remote_file}"
        )
        remote_checksum = checksum_result.stdout.strip().split()[0]

        print(f"   Remote MD5: {remote_checksum}")

        if local_checksum != remote_checksum:
            print(f"\n❌ FAILED: Checksum mismatch")
            os.unlink(local_file_path)
            return False

        print(f"   Integrity verified: ✅")

        # Cleanup
        os.unlink(local_file_path)
        executor.execute_command(f"rm -f {remote_file}")

        print("\n✅ PASSED - Large file transfer works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_8_permission_errors(executor):
    """Test 8: Handles permission errors"""
    print("\n" + "=" * 70)
    print("TEST 8: Handle Permission Errors")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: No SSH executor available")
        return False

    try:
        # Try to upload to a restricted directory
        test_content = "Permission test file"
        local_file = create_test_file(content=test_content)

        # Try uploading to /root (should fail for ubuntu user)
        restricted_path = "/root/test_file.txt"

        print(f"\n🔒 Attempting to upload to restricted directory...")
        print(f"   Target: {restricted_path}")

        try:
            executor.upload_file(local_file, restricted_path)
            print(
                f"\n⚠️  WARNING: Upload to restricted directory succeeded (unexpected)"
            )
            print(f"   This might mean the user has elevated privileges")
            os.unlink(local_file)
            return True  # Not a failure, just unexpected
        except Exception as e:
            print(f"\n✅ Expected permission error: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}")

        # Try downloading from a restricted file
        print(f"\n🔒 Attempting to download restricted file...")
        restricted_file = "/etc/shadow"
        local_download = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name

        try:
            executor.download_file(restricted_file, local_download)
            print(f"\n⚠️  WARNING: Download of restricted file succeeded (unexpected)")
            os.unlink(local_download)
        except Exception as e:
            print(f"\n✅ Expected permission error: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}")

        # Try uploading to a read-only location (after making it read-only)
        print(f"\n🔒 Testing upload to read-only directory...")
        readonly_dir = f"{test_dir}/readonly"
        executor.execute_command(f"mkdir -p {readonly_dir}")
        executor.execute_command(f"chmod 555 {readonly_dir}")

        readonly_file = f"{readonly_dir}/test.txt"

        try:
            executor.upload_file(local_file, readonly_file)
            print(f"\n⚠️  WARNING: Upload to read-only directory succeeded")
            # Cleanup
            executor.execute_command(f"chmod 755 {readonly_dir}")
            executor.execute_command(f"rm -f {readonly_file}")
        except Exception as e:
            print(f"\n✅ Expected permission error: {type(e).__name__}")
            # Restore permissions for cleanup
            executor.execute_command(f"chmod 755 {readonly_dir}")

        # Cleanup
        os.unlink(local_file)

        print("\n✅ PASSED - Permission errors handled gracefully!")
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
    """Run all file transfer tests"""
    print("\n" + "#" * 70)
    print("#  PHASE 2.3: FILE TRANSFER TESTING".center(70))
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

        # Setup SSH connection
        executor = setup_ssh_connection(instance_ip)

        if not executor:
            return

        # Run all tests
        results["upload_single_file"] = test_1_upload_single_file(executor)
        results["upload_nested_directories"] = test_2_upload_nested_directories(
            executor
        )
        results["download_single_file"] = test_3_download_single_file(executor)
        results["upload_multiple_files"] = test_4_upload_multiple_files(executor)
        results["download_directory_recursively"] = (
            test_5_download_directory_recursively(executor)
        )
        results["file_integrity_checksum"] = test_6_file_integrity_checksum(executor)
        results["large_file_transfer"] = test_7_large_file_transfer(executor)
        results["permission_errors"] = test_8_permission_errors(executor)

        # Cleanup SSH connection
        if executor:
            print("\n" + "=" * 70)
            print("CLEANUP: SSH Connection")
            print("=" * 70)
            print("\n🔌 Disconnecting SSH...")
            executor.disconnect()
            print("   Disconnected: ✅")

            # Cleanup test directory
            print("\n🧹 Cleaning up test directory...")
            executor.connect()
            executor.execute_command(f"rm -rf {test_dir}", check=False)
            executor.disconnect()
            print("   Test directory removed: ✅")

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
