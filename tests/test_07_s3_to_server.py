"""
Phase 3.1: S3 to Server Transfer Testing

Tests for transferring files from S3/MinIO to GPU server:
1. Can transfer train dataset from S3 to GPU
2. Can transfer eval dataset from S3 to GPU
3. Can transfer config YAML from S3 to GPU
4. Can transfer training script to GPU
5. All files land in correct directories
6. Can verify files exist on GPU server
7. Handles missing S3 files
8. Cleanup temporary files

Run: python tests/test_07_s3_to_server.py

⚠️  WARNING: This test requires:
⚠️  1. MinIO/S3 running with test files
⚠️  2. A running GPU instance from Lambda Labs
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

from scripts.storage_client import StorageClient
from scripts.ssh_executor import SshExecutor
from scripts.lambda_client import LambdaClient
from dotenv import load_dotenv

test_instance_id = None
test_dir = None
TEST_BUCKET = "innotone-training-test"
TEST_S3_PREFIX = "test-transfer/"


def setup_storage_client():
    """Initialize storage client and upload test files"""
    print("=" * 70)
    print("SETUP: Storage Client & Test Files")
    print("=" * 70)

    try:
        storage_client = StorageClient(storage_type="minio")

        print(f"\n📦 Storage Client Initialized:")
        print(f"   Type: {storage_client.storage_type}")
        print(f"   Bucket: {TEST_BUCKET}")

        # Ensure bucket exists
        storage_client.ensure_bucket_exists(TEST_BUCKET)

        # Create and upload test files
        print(f"\n📤 Uploading test files to S3...")

        # 1. Train dataset
        train_data = "epoch,loss,accuracy\n1,0.5,0.85\n2,0.3,0.92\n3,0.2,0.95\n"
        train_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        train_file.write(train_data)
        train_file.close()

        storage_client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=f"{TEST_S3_PREFIX}data/train_dataset.csv",
            file_path=train_file.name,
            content_type="text/csv",
        )
        os.unlink(train_file.name)
        print(f"   ✓ Uploaded train_dataset.csv")

        # 2. Eval dataset
        eval_data = "epoch,val_loss,val_accuracy\n1,0.6,0.80\n2,0.4,0.88\n3,0.3,0.91\n"
        eval_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
        eval_file.write(eval_data)
        eval_file.close()

        storage_client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=f"{TEST_S3_PREFIX}data/eval_dataset.csv",
            file_path=eval_file.name,
            content_type="text/csv",
        )
        os.unlink(eval_file.name)
        print(f"   ✓ Uploaded eval_dataset.csv")

        # 3. Config YAML
        config_yaml = """model:
  name: test-model
  epochs: 10
  batch_size: 32
training:
  learning_rate: 0.001
  optimizer: adam
"""
        config_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        )
        config_file.write(config_yaml)
        config_file.close()

        storage_client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=f"{TEST_S3_PREFIX}configs/training_config.yaml",
            file_path=config_file.name,
            content_type="application/x-yaml",
        )
        os.unlink(config_file.name)
        print(f"   ✓ Uploaded training_config.yaml")

        # 4. Training script
        training_script = """#!/bin/bash
echo "Starting training..."
echo "Training complete!"
"""
        script_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh")
        script_file.write(training_script)
        script_file.close()

        storage_client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=f"{TEST_S3_PREFIX}scripts/train.sh",
            file_path=script_file.name,
            content_type="application/x-sh",
        )
        os.unlink(script_file.name)
        print(f"   ✓ Uploaded train.sh")

        print(f"\n✅ Storage setup complete!")
        return storage_client

    except Exception as e:
        print(f"\n❌ Storage setup failed: {e}")
        traceback.print_exc()
        return None


def setup_test_instance():
    print("\n" + "=" * 70)
    print("SETUP: GPU Instance")
    print("=" * 70)
    load_dotenv()
    existing_ip = os.getenv("TEST_INSTANCE_IP")
    if existing_ip:
        print(f"\n✅ Using existing instance IP: {existing_ip}")
        return existing_ip, None
    print(
        "\n⚠️  No TEST_INSTANCE_IP in .env. Press ENTER to launch or Ctrl+C to cancel..."
    )
    try:
        input()
    except KeyboardInterrupt:
        sys.exit(0)
    try:
        client = LambdaClient()
        selected = client.list_available_instances()
        if not selected:
            return None, None
        print(f"\n🚀 Launching {selected['name']}...")
        instance = client.launch_instance(
            selected["name"],
            selected["region"],
            f"s3-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        if not instance:
            return None, None
        global test_instance_id
        test_instance_id = instance["id"]
        print(f"\n✅ Instance launched: {instance['ip']}")
        print("\n⏳ Waiting 30s for SSH...")
        time.sleep(30)
        return instance["ip"], test_instance_id
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return None, None


def setup_ssh_connection(ip):
    print("\n" + "=" * 70)
    print("SETUP: SSH Connection")
    print("=" * 70)
    if not ip:
        return None
    try:
        print(f"\n🔌 Connecting to {ip}...")
        global test_dir
        ssh_executor = SshExecutor(ip=ip, username="ubuntu")
        ssh_executor.connect()
        test_dir = f"/tmp/s3_transfer_test_{int(time.time())}"
        ssh_executor.execute_command(
            f"mkdir -p {test_dir}/data {test_dir}/configs {test_dir}/scripts"
        )
        print(f"\n✅ SSH connected! Test dir: {test_dir}")
        return ssh_executor
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return None


def test_1_transfer_train_dataset(storage, executor):
    """Test 1: Can transfer train dataset from S3 to GPU"""
    print("\n" + "=" * 70)
    print("TEST 1: Transfer Train Dataset from S3 to GPU")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}data/train_dataset.csv"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        remote_path = f"{test_dir}/data/train_dataset.csv"

        print(f"\n📥 Downloading from S3...")
        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if not success:
            print("\n❌ FAILED: Could not download from S3")
            return False

        print(f"   ✓ Downloaded from S3")

        print(f"\n📤 Uploading to GPU server...")
        executor.upload_file(local_temp, remote_path)
        print(f"   ✓ Uploaded to GPU")

        check_result = executor.execute_command(
            f"test -f {remote_path} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )

        os.unlink(local_temp)

        if "EXISTS" in check_result.stdout:
            print("\n✅ PASSED - Train dataset transferred successfully!")
            return True
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_7_handle_missing_s3_files(storage, executor):
    """Test 7: Handles missing S3 files"""
    print("\n" + "=" * 70)
    print("TEST 7: Handle Missing S3 Files")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}data/non_existent_file.csv"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

        print(f"\n📥 Attempting to download non-existent file...")

        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if os.path.exists(local_temp):
            os.unlink(local_temp)

        if not success:
            print(f"   ✓ Gracefully handled missing file")
            print("\n✅ PASSED - Missing S3 files handled gracefully!")
            return True

        print(f"\n❌ FAILED: Download should have failed")
        return False

    except Exception as e:
        print(f"   ✓ Exception handled: {type(e).__name__}")
        print("\n✅ PASSED - Missing S3 files handled gracefully!")
        return True


def test_8_cleanup_temporary_files(storage, executor):
    """Test 8: Cleanup temporary files"""
    print("\n" + "=" * 70)
    print("TEST 8: Cleanup Temporary Files")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        print(f"\n🧹 Cleaning up test files on GPU...")

        executor.execute_command(f"rm -rf {test_dir}", check=False)
        print(f"   ✓ Removed {test_dir}")

        check_result = executor.execute_command(
            f"test -d {test_dir} && echo 'EXISTS' || echo 'NOT_FOUND'", check=False
        )

        if "NOT_FOUND" in check_result.stdout:
            print(f"   ✓ Directory removed successfully")

        print(f"\n🧹 Cleaning up test files on S3...")

        objects = storage.list_objects(bucket_name=TEST_BUCKET, prefix=TEST_S3_PREFIX)

        deleted_count = 0
        for obj in objects:
            storage.delete_file(bucket_name=TEST_BUCKET, object_name=obj["name"])
            deleted_count += 1

        print(f"   ✓ Deleted {deleted_count} S3 objects")

        print("\n✅ PASSED - Cleanup completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_2_transfer_eval_dataset(storage, executor):
    """Test 2: Can transfer eval dataset from S3 to GPU"""
    print("\n" + "=" * 70)
    print("TEST 2: Transfer Eval Dataset from S3 to GPU")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}data/eval_dataset.csv"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        remote_path = f"{test_dir}/data/eval_dataset.csv"

        print(f"\n📥 Downloading from S3...")
        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if not success:
            print("\n❌ FAILED: Could not download from S3")
            return False

        print(f"   ✓ Downloaded from S3")

        print(f"\n📤 Uploading to GPU server...")
        executor.upload_file(local_temp, remote_path)
        print(f"   ✓ Uploaded to GPU")

        check_result = executor.execute_command(
            f"test -f {remote_path} && echo 'EXISTS' || echo 'NOT_FOUND'"
        )

        os.unlink(local_temp)

        if "EXISTS" in check_result.stdout:
            print("\n✅ PASSED - Eval dataset transferred successfully!")
            return True
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_3_transfer_config_yaml(storage, executor):
    """Test 3: Can transfer config YAML from S3 to GPU"""
    print("\n" + "=" * 70)
    print("TEST 3: Transfer Config YAML from S3 to GPU")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}configs/training_config.yaml"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml").name
        remote_path = f"{test_dir}/configs/training_config.yaml"

        print(f"\n📥 Downloading config from S3...")
        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if not success:
            print("\n❌ FAILED: Could not download from S3")
            return False

        print(f"   ✓ Downloaded from S3")

        print(f"\n📤 Uploading to GPU server...")
        executor.upload_file(local_temp, remote_path)
        print(f"   ✓ Uploaded to GPU")

        cat_result = executor.execute_command(f"cat {remote_path}")

        os.unlink(local_temp)

        if "test-model" in cat_result.stdout:
            print("\n✅ PASSED - Config YAML transferred successfully!")
            return True
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_4_transfer_training_script(storage, executor):
    """Test 4: Can transfer training script to GPU"""
    print("\n" + "=" * 70)
    print("TEST 4: Transfer Training Script from S3 to GPU")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}scripts/train.sh"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".sh").name
        remote_path = f"{test_dir}/scripts/train.sh"

        print(f"\n📥 Downloading script from S3...")
        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if not success:
            print("\n❌ FAILED: Could not download from S3")
            return False

        print(f"   ✓ Downloaded from S3")

        print(f"\n📤 Uploading to GPU server...")
        executor.upload_file(local_temp, remote_path)
        print(f"   ✓ Uploaded to GPU")

        print(f"\n🔧 Making script executable...")
        executor.execute_command(f"chmod +x {remote_path}")

        check_result = executor.execute_command(
            f"test -x {remote_path} && echo 'EXECUTABLE' || echo 'NOT_EXECUTABLE'"
        )

        os.unlink(local_temp)

        if "EXECUTABLE" in check_result.stdout:
            print("\n✅ PASSED - Training script transferred successfully!")
            return True
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_5_verify_directory_structure(executor):
    """Test 5: All files land in correct directories"""
    print("\n" + "=" * 70)
    print("TEST 5: Verify Directory Structure")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: SSH executor not available")
        return False

    try:
        print(f"\n📁 Checking directory structure...")

        expected_files = [
            f"{test_dir}/data/train_dataset.csv",
            f"{test_dir}/data/eval_dataset.csv",
            f"{test_dir}/configs/training_config.yaml",
            f"{test_dir}/scripts/train.sh",
        ]

        all_exist = True
        for file_path in expected_files:
            check_result = executor.execute_command(
                f"test -f {file_path} && echo 'EXISTS' || echo 'NOT_FOUND'", check=False
            )

            exists = "EXISTS" in check_result.stdout
            status = "✓" if exists else "❌"
            relative_path = file_path.replace(test_dir + "/", "")
            print(f"   {status} {relative_path}")

            if not exists:
                all_exist = False

        if all_exist:
            print("\n✅ PASSED - All files in correct directories!")
            return True
        return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_6_verify_files_exist(executor):
    """Test 6: Can verify files exist on GPU server"""
    print("\n" + "=" * 70)
    print("TEST 6: Verify Files Exist on GPU Server")
    print("=" * 70)

    if not executor:
        print("❌ SKIPPED: SSH executor not available")
        return False

    try:
        print(f"\n📊 Checking file sizes and permissions...")

        ls_result = executor.execute_command(
            f"ls -lh {test_dir}/data/ {test_dir}/configs/ {test_dir}/scripts/",
            check=False,
        )

        print(f"\n{ls_result.stdout}")

        count_result = executor.execute_command(f"find {test_dir} -type f | wc -l")

        file_count = int(count_result.stdout.strip())
        print(f"\n📊 Total files transferred: {file_count}")

        if file_count >= 4:
            print("\n✅ PASSED - All files verified on GPU!")
            return True
        else:
            print(f"\n❌ FAILED: Expected at least 4 files, found {file_count}")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_7_handle_missing_s3_files(storage, executor):
    """Test 7: Handles missing S3 files"""
    print("\n" + "=" * 70)
    print("TEST 7: Handle Missing S3 Files")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        s3_object = f"{TEST_S3_PREFIX}data/non_existent_file.csv"
        local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

        print(f"\n📥 Attempting to download non-existent file...")

        success = storage.download_file(
            bucket_name=TEST_BUCKET, object_name=s3_object, file_path=local_temp
        )

        if os.path.exists(local_temp):
            os.unlink(local_temp)

        if not success:
            print(f"   ✓ Gracefully handled missing file")
            print("\n✅ PASSED - Missing S3 files handled gracefully!")
            return True

        print(f"\n❌ FAILED: Download should have failed")
        return False

    except Exception as e:
        print(f"   ✓ Exception handled: {type(e).__name__}")
        print("\n✅ PASSED - Missing S3 files handled gracefully!")
        return True


def test_8_cleanup_temporary_files(storage, executor):
    """Test 8: Cleanup temporary files"""
    print("\n" + "=" * 70)
    print("TEST 8: Cleanup Temporary Files")
    print("=" * 70)

    if not storage or not executor:
        print("❌ SKIPPED: Storage or SSH executor not available")
        return False

    try:
        print(f"\n🧹 Cleaning up test files on GPU...")

        executor.execute_command(f"rm -rf {test_dir}", check=False)
        print(f"   ✓ Removed {test_dir}")

        check_result = executor.execute_command(
            f"test -d {test_dir} && echo 'EXISTS' || echo 'NOT_FOUND'", check=False
        )

        if "NOT_FOUND" in check_result.stdout:
            print(f"   ✓ Directory removed successfully")

        print(f"\n🧹 Cleaning up test files on S3...")

        objects = storage.list_objects(bucket_name=TEST_BUCKET, prefix=TEST_S3_PREFIX)

        deleted_count = 0
        for obj in objects:
            storage.delete_file(bucket_name=TEST_BUCKET, object_name=obj["name"])
            deleted_count += 1

        print(f"   ✓ Deleted {deleted_count} S3 objects")

        print("\n✅ PASSED - Cleanup completed successfully!")
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
    """Run all S3 to server transfer tests"""
    print("\n" + "#" * 70)
    print("#  PHASE 3.1: S3 TO SERVER TRANSFER TESTING".center(70))
    print("#" * 70)
    print("\n⚠️  WARNING: This requires MinIO/S3 and a running GPU instance!")
    print("⚠️  Set TEST_INSTANCE_IP in .env or we'll launch a new instance")
    print("⚠️  Estimated cost: $0.60-$1.10/hour")
    print("\n" + "=" * 70)

    results = {}

    try:
        storage = setup_storage_client()
        if not storage:
            print("\n❌ Cannot proceed without storage client")
            return

        instance_ip, instance_id = setup_test_instance()
        if not instance_ip:
            print("\n❌ Cannot proceed without instance IP")
            return

        executor = setup_ssh_connection(instance_ip)
        if not executor:
            return

        results["transfer_train_dataset"] = test_1_transfer_train_dataset(
            storage, executor
        )
        results["transfer_eval_dataset"] = test_2_transfer_eval_dataset(
            storage, executor
        )
        results["transfer_config_yaml"] = test_3_transfer_config_yaml(storage, executor)
        results["transfer_training_script"] = test_4_transfer_training_script(
            storage, executor
        )
        results["verify_directory_structure"] = test_5_verify_directory_structure(
            executor
        )
        results["verify_files_exist"] = test_6_verify_files_exist(executor)
        results["handle_missing_s3_files"] = test_7_handle_missing_s3_files(
            storage, executor
        )
        results["cleanup_temporary_files"] = test_8_cleanup_temporary_files(
            storage, executor
        )

        if executor:
            print("\n" + "=" * 70)
            print("CLEANUP: SSH Connection")
            print("=" * 70)
            print("\n🔌 Disconnecting SSH...")
            executor.disconnect()
            print("   Disconnected: ✅")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        cleanup_test_instance()

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
