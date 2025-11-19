"""
Phase 1.2: Project YAML Builder Testing

Tests for:
1. Parsing training request JSON correctly
2. Generating proper YAML structure
3. All required fields populated
4. S3 paths constructed correctly
5. Train/eval dataset paths
6. Config path
7. Upload YAML to S3/MinIO
8. Download and verify uploaded YAML

Run: python tests/test_02_yaml_builder.py
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.project_yaml_builder import ProjectYamlBuilder
from scripts.storage_client import StorageClient
from dotenv import load_dotenv

load_dotenv()


def get_sample_training_request():
    """Get sample training request JSON for testing"""
    return {
        "training_run_id": "test-run-12345",
        "project": {
            "id": "test-project-abc",
            "name": "Test Spam Classifier",
            "description": "Testing YAML builder",
            "task_type": "text_classification",
        },
        "prompt": {
            "id": "prompt-123",
            "name": "spam classifier prompt",
            "content": "You are a helpful assistant that classifies text messages.",
            "think_flag": True,
            "seed_text": "Let me think...",
        },
        "train_dataset": {
            "id": "train-dataset-456",
            "name": "spam train",
            "file_name": "projects/test-project-abc/train/spam_train.csv",
            "file_format": "csv",
            "text_col": "text",
            "label_col": "label",
        },
        "eval_dataset": {
            "id": "eval-dataset-789",
            "name": "spam eval",
            "file_name": "projects/test-project-abc/eval/spam_eval.csv",
            "file_format": "csv",
            "text_col": "text",
            "label_col": "label",
        },
        "metadata": {},
    }


def test_1_parse_json():
    """Test 1: Parse training request JSON correctly"""
    print("=" * 70)
    print("TEST 1: Parse Training Request JSON")
    print("=" * 70)

    try:
        data = get_sample_training_request()
        builder = ProjectYamlBuilder()
        
        # This should not raise any errors
        builder._add_data(data)
        
        print("\n📝 JSON parsed successfully")
        print(f"   Project ID: {builder.yaml_data.get('project_id')}")
        print(f"   Training Run ID: {builder.yaml_data.get('training_run_id')}")
        print(f"   Total fields: {len(builder.yaml_data)}")
        
        assert builder.yaml_data is not None
        assert len(builder.yaml_data) > 0
        
        print("\n✅ PASSED")
        return True, builder

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_2_yaml_structure(builder):
    """Test 2: Verify proper YAML structure is generated"""
    print("\n" + "=" * 70)
    print("TEST 2: YAML Structure Generation")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        yaml_data = builder.yaml_data
        
        # Check that YAML can be serialized
        yaml_str = yaml.safe_dump(yaml_data, sort_keys=False)
        
        # Check that it can be parsed back
        parsed = yaml.safe_load(yaml_str)
        
        print("\n📝 YAML Structure:")
        print(f"   Serializable: ✓")
        print(f"   Parseable: ✓")
        print(f"   Total keys: {len(parsed)}")
        
        assert parsed == yaml_data
        
        print("\n✅ PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_required_fields(builder):
    """Test 3: Verify all required fields are populated"""
    print("\n" + "=" * 70)
    print("TEST 3: Required Fields Population")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        required_fields = [
            # Project fields
            "project_name",
            "project_id",
            "project_description",
            "project_task_type",
            "training_run_id",
            # Prompt fields
            "prompt_id",
            "prompt_name",
            "system_prompt",
            "think_flag",
            # Dataset fields
            "train_dataset_id",
            "eval_dataset_id",
            # Model fields
            "model_name",
            "ref_model_name",
            # Training params
            "no_epochs",
            "no_iterations",
            "loss",
        ]
        
        missing_fields = []
        populated_fields = []
        
        for field in required_fields:
            if field in builder.yaml_data:
                populated_fields.append(field)
            else:
                missing_fields.append(field)
        
        print(f"\n📊 Field Check:")
        print(f"   Required: {len(required_fields)}")
        print(f"   Populated: {len(populated_fields)}")
        print(f"   Missing: {len(missing_fields)}")
        
        if missing_fields:
            print(f"\n⚠️  Missing fields: {missing_fields}")
        
        assert len(missing_fields) == 0, f"Missing required fields: {missing_fields}"
        
        print("\n✅ PASSED - All required fields populated")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_s3_paths(builder):
    """Test 4: Verify S3 paths are constructed correctly"""
    print("\n" + "=" * 70)
    print("TEST 4: S3 Path Construction")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        yaml_data = builder.yaml_data
        
        # Check training S3 path
        training_s3_path = yaml_data.get("training_s3_path")
        expected_training_path = f"media/projects/{yaml_data['project_id']}/{yaml_data['training_run_id']}"
        
        print(f"\n📝 Training S3 Path:")
        print(f"   Expected: {expected_training_path}")
        print(f"   Actual:   {training_s3_path}")
        print(f"   Match: {'✓' if training_s3_path == expected_training_path else '✗'}")
        
        assert training_s3_path == expected_training_path, "Training S3 path mismatch"
        
        # Check config S3 path
        config_s3_path = yaml_data.get("config_s3_path")
        expected_config_path = f"{training_s3_path}/config.yaml"
        
        print(f"\n📝 Config S3 Path:")
        print(f"   Expected: {expected_config_path}")
        print(f"   Actual:   {config_s3_path}")
        print(f"   Match: {'✓' if config_s3_path == expected_config_path else '✗'}")
        
        assert config_s3_path == expected_config_path, "Config S3 path mismatch"
        
        print("\n✅ PASSED - S3 paths correct")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_train_dataset_path(builder):
    """Test 5: Verify train dataset path is correct"""
    print("\n" + "=" * 70)
    print("TEST 5: Train Dataset Path")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        yaml_data = builder.yaml_data
        
        train_s3_path = yaml_data.get("train_s3_path")
        train_file_name = yaml_data.get("train_file_name")
        train_file_path = yaml_data.get("train_file_path")
        
        print(f"\n📝 Train Dataset Paths:")
        print(f"   S3 Path:   {train_s3_path}")
        print(f"   File Name: {train_file_name}")
        print(f"   File Path: {train_file_path}")
        
        # Verify S3 path format
        assert train_s3_path.startswith("media/"), "Train S3 path should start with 'media/'"
        
        # Verify file name is basename of S3 path
        assert train_file_name == os.path.basename(train_s3_path), "File name should be basename of S3 path"
        
        # Verify file path format
        assert train_file_path == f"data/{train_file_name}", "File path should be in data/ directory"
        
        print("\n✅ PASSED - Train dataset paths correct")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_eval_dataset_path(builder):
    """Test 6: Verify eval dataset path is correct"""
    print("\n" + "=" * 70)
    print("TEST 6: Eval Dataset Path")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        yaml_data = builder.yaml_data
        
        eval_s3_path = yaml_data.get("eval_s3_path")
        eval_file_name = yaml_data.get("eval_file_name")
        eval_file_path = yaml_data.get("eval_file_path")
        cv_file_path = yaml_data.get("cv_file_path")
        
        print(f"\n📝 Eval Dataset Paths:")
        print(f"   S3 Path:   {eval_s3_path}")
        print(f"   File Name: {eval_file_name}")
        print(f"   File Path: {eval_file_path}")
        print(f"   CV Path:   {cv_file_path}")
        
        # Verify S3 path format
        assert eval_s3_path.startswith("media/"), "Eval S3 path should start with 'media/'"
        
        # Verify file name is basename of S3 path
        assert eval_file_name == os.path.basename(eval_s3_path), "File name should be basename of S3 path"
        
        # Verify file path format
        assert eval_file_path == f"data/{eval_file_name}", "File path should be in data/ directory"
        
        # Verify CV path matches eval path
        assert cv_file_path == eval_file_path, "CV file path should match eval file path"
        
        print("\n✅ PASSED - Eval dataset paths correct")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_config_path(builder):
    """Test 7: Verify config path is correct"""
    print("\n" + "=" * 70)
    print("TEST 7: Config Path")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        yaml_data = builder.yaml_data
        
        config_s3_path = yaml_data.get("config_s3_path")
        config_file_path = yaml_data.get("config_file_path")
        
        print(f"\n📝 Config Paths:")
        print(f"   S3 Path:   {config_s3_path}")
        print(f"   File Path: {config_file_path}")
        
        # Verify S3 path ends with config.yaml
        assert config_s3_path.endswith("config.yaml"), "Config S3 path should end with config.yaml"
        
        # Verify file path
        assert config_file_path == "projects_yaml/config.yaml", "Config file path should be projects_yaml/config.yaml"
        
        print("\n✅ PASSED - Config paths correct")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_upload_to_storage(builder):
    """Test 8: Upload YAML to S3/MinIO"""
    print("\n" + "=" * 70)
    print("TEST 8: Upload YAML to Storage")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        storage_type = os.getenv("STORAGE_TYPE", "minio")
        bucket_name = os.getenv("BUCKET_NAME", "innotone-media")
        
        print(f"\n📝 Storage Configuration:")
        print(f"   Type: {storage_type}")
        print(f"   Bucket: {bucket_name}")
        print(f"   Object: {builder.yaml_data['config_s3_path']}")
        
        # Attempt upload
        print("\n⏳ Uploading...")
        success = builder.save_to_s3()
        
        if success:
            print("✓ Upload successful")
        else:
            print("✗ Upload failed")
        
        assert success, "Upload to storage failed"
        
        print("\n✅ PASSED - YAML uploaded successfully")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print(f"\n⚠️  Note: Make sure {os.getenv('STORAGE_TYPE', 'MinIO')} is running")
        import traceback
        traceback.print_exc()
        return False


def test_9_download_and_verify(builder):
    """Test 9: Download and verify uploaded YAML"""
    print("\n" + "=" * 70)
    print("TEST 9: Download and Verify YAML")
    print("=" * 70)

    if not builder:
        print("❌ SKIPPED: No builder available")
        return False

    try:
        storage_type = os.getenv("STORAGE_TYPE", "minio")
        bucket_name = os.getenv("BUCKET_NAME", "innotone-media")
        object_name = builder.yaml_data['config_s3_path']
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Download from storage
            storage = StorageClient(storage_type=storage_type)
            print(f"\n⏳ Downloading from {storage_type}...")
            success = storage.download_file(bucket_name, object_name, temp_path)
            
            assert success, "Download failed"
            print("✓ Download successful")
            
            # Read and parse downloaded YAML
            with open(temp_path, 'r') as f:
                downloaded_data = yaml.safe_load(f)
            
            print("\n📊 Verification:")
            print(f"   Original keys: {len(builder.yaml_data)}")
            print(f"   Downloaded keys: {len(downloaded_data)}")
            
            # Verify key fields match
            key_fields = ["project_id", "training_run_id", "model_name", "config_s3_path"]
            mismatches = []
            
            for field in key_fields:
                original = builder.yaml_data.get(field)
                downloaded = downloaded_data.get(field)
                match = original == downloaded
                print(f"   {field}: {'✓' if match else '✗'}")
                if not match:
                    mismatches.append(field)
            
            assert len(mismatches) == 0, f"Field mismatches: {mismatches}"
            
            print("\n✅ PASSED - Downloaded YAML verified")
            return True
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all YAML builder tests"""
    print("\n" + "#" * 70)
    print("#  PHASE 1.2: PROJECT YAML BUILDER TESTING".center(70))
    print("#" * 70)

    results = {}

    # Test 1: Parse JSON
    success, builder = test_1_parse_json()
    results["parse_json"] = success

    # Test 2: YAML structure
    results["yaml_structure"] = test_2_yaml_structure(builder)

    # Test 3: Required fields
    results["required_fields"] = test_3_required_fields(builder)

    # Test 4: S3 paths
    results["s3_paths"] = test_4_s3_paths(builder)

    # Test 5: Train dataset path
    results["train_dataset_path"] = test_5_train_dataset_path(builder)

    # Test 6: Eval dataset path
    results["eval_dataset_path"] = test_6_eval_dataset_path(builder)

    # Test 7: Config path
    results["config_path"] = test_7_config_path(builder)

    # Test 8: Upload to storage
    results["upload_yaml"] = test_8_upload_to_storage(builder)

    # Test 9: Download and verify
    results["download_verify"] = test_9_download_and_verify(builder)

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
        if not results.get("upload_yaml") or not results.get("download_verify"):
            print(f"\n💡 Tip: Make sure {os.getenv('STORAGE_TYPE', 'MinIO')} is running:")
            if os.getenv('STORAGE_TYPE', 'minio') == 'minio':
                print("   docker-compose up -d  # or your MinIO startup command")


if __name__ == "__main__":
    run_all_tests()