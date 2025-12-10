import os
from minio import Minio
from minio.error import S3Error
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import boto3
from datetime import timedelta

load_dotenv()


class StorageClient:

    def __init__(self, storage_type: Optional[str] = None, **kwargs):

        self.storage_type = storage_type or os.getenv("STORAGE_TYPE", "minio").lower()
        self.buket_prefix = os.getenv("S3_BUCKET_PREFIX", "")

        if self.storage_type == "minio":
            self._init_minio(**kwargs)
        elif self.storage_type == "aws_s3":
            self._init_aws_s3(**kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

    def _init_minio(self, **kwargs):

        self.client = Minio(
            endpoint=kwargs.get("endpoint")
            or os.getenv("MINIO_ENDPOINT", "localhost:10000"),
            access_key=kwargs.get("access_key")
            or os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=kwargs.get("secret_key")
            or os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=kwargs.get("secure")
            or os.getenv("MINIO_SECURE", "false").lower() == "true",
            region=kwargs.get("region") or os.getenv("MINIO_REGION"),
        )
        self.bucket_url = f"http://{self.client._base_url}"

    def _init_aws_3(self, **kwargs):

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
            if self.storage_type == "minio":
                buckets = self.client.list_buckets()
                return [{"name": bucket.name, "creation_date": bucket.creation_date} 
                    for bucket in buckets]
            else:
                # AWS S3
                response = self.client.list_buckets()
                return [{"name": bucket["Name"], "creation_date": bucket["CreationDate"]} 
                    for bucket in response.get("Buckets", [])]
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []
    def _get_bucket_name(self, bucket_name: str):
        if self.storage_type == "aws_s3" and self.bucket_prefix:
            return f"{self.bucket_prefix}{bucket_name}"
        return bucket_name

    def ensure_bucket_exists(self, bucket_name: str):
        bucket_name = self._get_bucket_name(bucket_name)

        if self.storage_type == "minio":
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        else:
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

            if self.storage_type == "minio":
                self.client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                    content_type=content_type,
                    metadata=metadata or {},
                )
            else:
                # aws s3
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
            if self.storage_type == "minio":
                self.client.fget_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(file_path),
                )
            else:
                # aws s3
                self.client.download_file(bucket_name, object_name, str(file_path))
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    def delete_file(self, bucket_name: str, object_name: str):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            if self.storage_type == "minio":
                self.client.remove_object(bucket_name, object_name)
            else:
                self.client.delete_object(Bucket=bucket_name, Key=object_name)
            return True
        except Exception as e:
            print(f"Error deleting file {e}")
            return False

    def list_objects(self, bucket_name: str, prefix: str = ""):
        try:
            bucket_name = self._get_bucket_name(bucket_name)
            objects = []

            if self.storage_type == "minio":
                result = self.client.list_objects(
                    bucket_name, prefix=prefix, recursive=True
                )
                for obj in result:
                    if not obj.object_name.endswith('/'):
                        objects.append(
                            {
                                "name": obj.object_name,
                                "size": obj.size,
                                "last_modified": obj.last_modified,
                                "etag": obj.etag,
                            }
                        )
            else:
                # aws s3
                result = self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                for obj in result.get("Contents", []):
                    if not obj['Key'].endswith('/'):
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

            if self.storage_type == "minio":
                return self.client.presigned_get_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    expires=timedelta(seconds=expires_seconds),
                )
            else:
                # aws s3
                return self.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": object_name},
                    ExpiresIn=expires_seconds,
                )

        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None