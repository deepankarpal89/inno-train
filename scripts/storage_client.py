import os
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import boto3
from datetime import timedelta

load_dotenv()


class StorageClient:

    def __init__(self, **kwargs):
        """Initialize AWS S3 storage client."""
        self.storage_type = "aws_s3"
        self.bucket_prefix = os.getenv("S3_BUCKET_PREFIX", "")
        self._init_aws_s3(**kwargs)

    def _init_aws_s3(self, **kwargs):

        self.client = boto3.client(
            "s3",
            aws_access_key_id=kwargs.get("access_key")
            or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=kwargs.get("secret_key")
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=kwargs.get("region")
            or os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
        )
        self.bucket_url = f"https://s3.{self.client.meta.region_name}.amazonaws.com"

    def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all buckets in the storage service.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing bucket information
                                with keys: 'name' and 'creation_date'
        """
        try:
            response = self.client.list_buckets()
            return [
                {"name": bucket["Name"], "creation_date": bucket["CreationDate"]}
                for bucket in response.get("Buckets", [])
            ]
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []

    def _get_bucket_name(self, bucket_name: str):
        if self.bucket_prefix:
            return f"{self.bucket_prefix}{bucket_name}"
        return bucket_name

    def ensure_bucket_exists(self, bucket_name: str):
        bucket_name = self._get_bucket_name(bucket_name)
        try:
            self.client.head_bucket(Bucket=bucket_name)
        except self.client.exceptions.NoSuchBucket:
            self.client.create_bucket(Bucket=bucket_name)

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            self.ensure_bucket_exists(bucket_name)

            extra_args = {"ContentType": content_type, "Metadata": metadata or {}}
            self.client.upload_file(
                str(file_path), bucket_name, object_name, ExtraArgs=extra_args
            )
            return True
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(
        self, bucket_name: str, object_name: str, file_path: Union[str, Path]
    ):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            self.client.download_file(bucket_name, object_name, str(file_path))
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    def delete_file(self, bucket_name: str, object_name: str):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
            return True
        except Exception as e:
            print(f"Error deleting file {e}")
            return False

    def list_objects(self, bucket_name: str, prefix: str = ""):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            objects = []

            result = self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            for obj in result.get("Contents", []):
                if not obj["Key"].endswith("/"):
                    objects.append(
                        {
                            "name": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["Etag"],
                        }
                    )
            return objects
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def get_presigned_url(
        self, bucket_name: str, object_name: str, expires_seconds: int = 3600
    ):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=expires_seconds,
            )
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
