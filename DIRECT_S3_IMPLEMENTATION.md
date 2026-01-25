# Direct S3 Transfer Implementation

## Overview

Implemented direct S3 access from GPU instances, eliminating the data bottleneck through the inno-train server. Data now flows directly between S3 and GPU instances using AWS CLI commands executed via SSH.

## Changes Made

### 1. Enhanced `ssh_executor.py`

Added new methods for S3 operations on GPU instances:

- **`setup_aws_credentials()`**: Configures AWS credentials on remote GPU via environment variables
- **`download_from_s3()`**: Downloads files from S3 to GPU using `aws s3 cp`
- **`upload_to_s3()`**: Uploads files from GPU to S3 using `aws s3 cp` or `aws s3 sync`
- **`check_aws_cli_installed()`**: Verifies AWS CLI is available on GPU
- **`install_aws_cli()`**: Installs AWS CLI v2 if not present

### 2. Updated `training_workflow.py`

#### Removed Dependencies

- Removed import: `from scripts.s3_to_server_transfer import S3ToServerTransfer`
- Removed initialization: `self.file_transfer = S3ToServerTransfer(logger=self.logger)`

#### New Methods

- **`_setup_aws_on_gpu()`**: Ensures AWS CLI is installed and credentials are configured on GPU instance

#### Modified Methods

- **`_transfer_all_files()`**:
  - Now downloads files directly from S3 to GPU using SSH commands
  - Calls `_setup_aws_on_gpu()` first
  - Downloads train dataset, eval dataset, and config in parallel
  - Still uploads training script via SSH (needed for execution)

- **`_download_outputs()`**:
  - Renamed to better reflect it now uploads (not downloads)
  - Uses `ssh_executor.upload_to_s3()` to sync output directory to S3
  - No intermediate transfer through inno-train server

- **`_download_file_from_s3()`**: New helper method for parallel S3 downloads

## Data Flow

### Before (Old Architecture)

```
Training Start:
S3 → inno-train (temp) → SSH/SFTP → GPU

Training End:
GPU → SSH/SFTP → inno-train (temp) → S3
```

### After (New Architecture)

```
Training Start:
S3 → (AWS CLI via SSH) → GPU

Training End:
GPU → (AWS CLI via SSH) → S3
```

## Benefits

✅ **Faster Transfers**: Direct S3 access eliminates intermediate hop
✅ **Lower Bandwidth**: No data flows through inno-train server
✅ **Scalability**: Multiple GPU instances can transfer in parallel
✅ **Cost Reduction**: Less egress from inno-train server
✅ **No Docker Changes**: `run_docker_job.sh` remains untouched

## How It Works

1. **Setup Phase** (in `_setup_aws_on_gpu`):
   - Check if AWS CLI is installed on GPU
   - Install AWS CLI v2 if needed
   - Configure AWS credentials via environment variables

2. **Download Phase** (in `_transfer_all_files`):
   - Execute `aws s3 cp s3://bucket/path local/path` via SSH
   - Downloads happen in parallel for all files
   - Creates necessary directories on GPU

3. **Upload Phase** (in `_download_outputs`):
   - Execute `aws s3 sync local/output/ s3://bucket/path/` via SSH
   - Syncs entire output directory to S3
   - Excludes log files to reduce transfer size

## Environment Variables Required

Ensure these are set in `.env`:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (default: ap-south-1)
- `BUCKET_NAME`: S3 bucket name

## Testing Checklist

- [ ] Verify AWS CLI installs correctly on fresh GPU instance
- [ ] Test credential configuration works
- [ ] Verify training dataset downloads from S3
- [ ] Verify eval dataset downloads from S3
- [ ] Verify config file downloads from S3
- [ ] Verify training executes successfully
- [ ] Verify outputs upload to S3 after training
- [ ] Check S3 bucket for uploaded files
- [ ] Verify no data flows through inno-train server

## Rollback Plan

If issues occur, you can temporarily revert by:

1. Re-add import: `from scripts.s3_to_server_transfer import S3ToServerTransfer`
2. Re-add initialization: `self.file_transfer = S3ToServerTransfer(logger=self.logger)`
3. Revert `_transfer_all_files()` and `_download_outputs()` methods

The old `s3_to_server_transfer.py` file is still available for reference.

## Next Steps

1. Test with a small training job
2. Monitor logs for any AWS CLI errors
3. Verify S3 transfer speeds are improved
4. Consider removing `s3_to_server_transfer.py` after successful testing
