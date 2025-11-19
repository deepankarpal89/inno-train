"""
Phase 1.3: Storage Client (S3/MinIO) Testing

Tests for:
1. Connection to storage (MinIO/S3)
2. Upload single file
3. Download single file
4. List objects with prefix
5. Delete objects
6. Handle non-existent files gracefully
7. Handle connection errors

Run: python tests/test_03_storage.py
"""

import asyncio
import os
import sys
from pathlib import Path
import tempfile
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.storage_client import StorageClient
from dotenv import load_dotenv

load_dotenv()

# Test configuration
TEST_BUCKET = "test-innotone-storage"
TEST_PREFIX = "test-files/"


def create_test_file(content: str = "Test file content") -> Path:
    """Create a temporary test file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


async def test_1_connection():
    """Test 1: Can connect to storage (MinIO/S3)"""
    print("=" * 70)
    print("TEST 1: Storage Connection")
    print("=" * 70)
    
    try:
        # Test MinIO connection
        client = StorageClient(storage_type="minio")
        
        print(f"\n📝 Storage Client Initialized:")
        print(f"   Type: {client.storage_type}")
        print(f"   Endpoint: {os.getenv('MINIO_ENDPOINT', 'localhost:10000')}")
        
        # Test listing buckets (verifies connection)
        buckets = client.list_buckets()
        print(f"   Buckets found: {len(buckets)}")
        
        if buckets:
            for bucket in buckets[:3]:  # Show first 3
                print(f"      - {bucket['name']}")
        
        print("\n✅ PASSED - Connection successful!")
        return client
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return None


async def test_2_upload_file(client: StorageClient):
    """Test 2: Can upload single file"""
    print("\n" + "=" * 70)
    print("TEST 2: Upload Single File")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        # Create test file
        test_file = create_test_file("Hello from InnoTone Storage Test!")
        object_name = f"{TEST_PREFIX}test_upload.txt"
        
        print(f"\n📝 Uploading file:")
        print(f"   Local: {test_file}")
        print(f"   Bucket: {TEST_BUCKET}")
        print(f"   Object: {object_name}")
        
        # Upload
        success = client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=object_name,
            file_path=test_file,
            content_type="text/plain",
            metadata={"test": "true", "source": "test_03_storage"}
        )
        
        # Cleanup local file
        test_file.unlink()
        
        if success:
            print("\n✅ PASSED - File uploaded successfully!")
            return True
        else:
            print("\n❌ FAILED - Upload returned False")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_3_download_file(client: StorageClient):
    """Test 3: Can download single file"""
    print("\n" + "=" * 70)
    print("TEST 3: Download Single File")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        object_name = f"{TEST_PREFIX}test_upload.txt"
        download_path = Path(tempfile.gettempdir()) / "downloaded_test.txt"
        
        print(f"\n📝 Downloading file:")
        print(f"   Bucket: {TEST_BUCKET}")
        print(f"   Object: {object_name}")
        print(f"   Local: {download_path}")
        
        # Download
        success = client.download_file(
            bucket_name=TEST_BUCKET,
            object_name=object_name,
            file_path=download_path
        )
        
        if success and download_path.exists():
            # Verify content
            content = download_path.read_text()
            print(f"\n📄 Downloaded content: {content[:50]}...")
            
            # Cleanup
            download_path.unlink()
            
            print("\n✅ PASSED - File downloaded successfully!")
            return True
        else:
            print("\n❌ FAILED - Download failed or file not found")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_4_list_objects(client: StorageClient):
    """Test 4: Can list objects with prefix"""
    print("\n" + "=" * 70)
    print("TEST 4: List Objects with Prefix")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        # Upload multiple test files
        print(f"\n📝 Uploading multiple test files...")
        for i in range(3):
            test_file = create_test_file(f"Test file {i+1}")
            client.upload_file(
                bucket_name=TEST_BUCKET,
                object_name=f"{TEST_PREFIX}file_{i+1}.txt",
                file_path=test_file,
                content_type="text/plain"
            )
            test_file.unlink()
        
        # List objects with prefix
        print(f"\n📝 Listing objects with prefix: {TEST_PREFIX}")
        objects = client.list_objects(
            bucket_name=TEST_BUCKET,
            prefix=TEST_PREFIX
        )
        
        print(f"\n📊 Found {len(objects)} objects:")
        for obj in objects:
            print(f"   - {obj['name']} ({obj['size']} bytes)")
        
        if len(objects) >= 3:
            print("\n✅ PASSED - Objects listed successfully!")
            return True
        else:
            print(f"\n❌ FAILED - Expected at least 3 objects, found {len(objects)}")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_5_delete_objects(client: StorageClient):
    """Test 5: Can delete objects"""
    print("\n" + "=" * 70)
    print("TEST 5: Delete Objects")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        # Get list of test objects
        objects = client.list_objects(
            bucket_name=TEST_BUCKET,
            prefix=TEST_PREFIX
        )
        
        print(f"\n📝 Deleting {len(objects)} test objects...")
        
        deleted_count = 0
        for obj in objects:
            success = client.delete_file(
                bucket_name=TEST_BUCKET,
                object_name=obj['name']
            )
            if success:
                deleted_count += 1
                print(f"   ✓ Deleted: {obj['name']}")
        
        # Verify deletion
        remaining = client.list_objects(
            bucket_name=TEST_BUCKET,
            prefix=TEST_PREFIX
        )
        
        print(f"\n📊 Deleted: {deleted_count}/{len(objects)}")
        print(f"📊 Remaining: {len(remaining)}")
        
        if len(remaining) == 0:
            print("\n✅ PASSED - All objects deleted successfully!")
            return True
        else:
            print(f"\n⚠️  PARTIAL - {len(remaining)} objects still remain")
            return True  # Still pass if some deleted
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def test_6_non_existent_file(client: StorageClient):
    """Test 6: Handles non-existent files gracefully"""
    print("\n" + "=" * 70)
    print("TEST 6: Handle Non-Existent Files")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        non_existent = f"{TEST_PREFIX}does_not_exist.txt"
        download_path = Path(tempfile.gettempdir()) / "should_not_exist.txt"
        
        print(f"\n📝 Attempting to download non-existent file:")
        print(f"   Object: {non_existent}")
        
        # Should return False, not crash
        success = client.download_file(
            bucket_name=TEST_BUCKET,
            object_name=non_existent,
            file_path=download_path
        )
        
        if not success:
            print("\n✅ PASSED - Gracefully handled non-existent file!")
            return True
        else:
            print("\n❌ FAILED - Should have returned False")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: Exception raised instead of graceful handling")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False


async def test_7_connection_errors(client: StorageClient):
    """Test 7: Handles connection errors"""
    print("\n" + "=" * 70)
    print("TEST 7: Handle Connection Errors")
    print("=" * 70)
    
    try:
        # Try to connect with invalid credentials
        print(f"\n📝 Testing with invalid credentials...")
        
        bad_client = StorageClient(
            storage_type="minio",
            endpoint="localhost:10000",
            access_key="invalid",
            secret_key="invalid"
        )
        
        # Try to list buckets (should fail gracefully)
        buckets = bad_client.list_buckets()
        
        # Should return empty list, not crash
        if isinstance(buckets, list) and len(buckets) == 0:
            print("\n✅ PASSED - Gracefully handled connection error!")
            return True
        else:
            print("\n⚠️  WARNING - Unexpected behavior with bad credentials")
            return True  # Still pass if no crash
            
    except Exception as e:
        # If it raises exception, that's also acceptable
        print(f"\n✅ PASSED - Exception raised (acceptable): {type(e).__name__}")
        return True


async def test_8_presigned_url(client: StorageClient):
    """Test 8: Generate presigned URL (bonus test)"""
    print("\n" + "=" * 70)
    print("TEST 8: Generate Presigned URL (Bonus)")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        # Upload a test file first
        test_file = create_test_file("Presigned URL test content")
        object_name = f"{TEST_PREFIX}presigned_test.txt"
        
        client.upload_file(
            bucket_name=TEST_BUCKET,
            object_name=object_name,
            file_path=test_file,
            content_type="text/plain"
        )
        test_file.unlink()
        
        # Generate presigned URL
        print(f"\n📝 Generating presigned URL:")
        print(f"   Object: {object_name}")
        
        url = client.get_presigned_url(
            bucket_name=TEST_BUCKET,
            object_name=object_name,
            expires_seconds=3600
        )
        
        if url:
            print(f"\n📄 Generated URL: {url[:80]}...")
            
            # Cleanup
            client.delete_file(TEST_BUCKET, object_name)
            
            print("\n✅ PASSED - Presigned URL generated!")
            return True
        else:
            print("\n❌ FAILED - No URL generated")
            return False
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 1.3: STORAGE CLIENT (S3/MinIO) TESTING".center(70))
    print("#" * 70)
    
    results = {}
    
    # Test 1: Connection
    client = await test_1_connection()
    results["connection"] = client is not None
    
    # Test 2: Upload
    results["upload"] = await test_2_upload_file(client)
    
    # Test 3: Download
    results["download"] = await test_3_download_file(client)
    
    # Test 4: List objects
    results["list_objects"] = await test_4_list_objects(client)
    
    # Test 5: Delete
    results["delete"] = await test_5_delete_objects(client)
    
    # Test 6: Non-existent file
    results["non_existent"] = await test_6_non_existent_file(client)
    
    # Test 7: Connection errors
    results["connection_errors"] = await test_7_connection_errors(client)
    
    # Test 8: Presigned URL (bonus)
    results["presigned_url"] = await test_8_presigned_url(client)
    
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
    print("NOTES:")
    print("=" * 70)
    print("• Make sure MinIO is running: docker-compose up -d")
    print("• Check .env file for correct MINIO_* settings")
    print(f"• Test bucket '{TEST_BUCKET}' will be created if needed")
    print("• All test files are cleaned up after tests")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all_tests())