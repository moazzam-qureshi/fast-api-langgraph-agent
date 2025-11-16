"""MinIO service for object storage operations."""
import os
from typing import BinaryIO, Optional
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)


class MinIOService:
    """Service class for MinIO operations."""
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
        )
        
        self.default_bucket = "documents"
        
        # Only try to create bucket if not in Railway environment or if MinIO is available
        if os.getenv("RAILWAY_ENVIRONMENT") != "production":
            try:
                self._ensure_bucket_exists(self.default_bucket)
            except Exception as e:
                logger.warning(f"Could not create bucket on startup: {e}. Will retry on first use.")
    
    def _ensure_bucket_exists(self, bucket_name: str) -> None:
        """Ensure the bucket exists, create if it doesn't."""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
        except S3Error as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            raise
    
    def upload_file(
        self,
        file_data: BinaryIO,
        object_name: str,
        content_type: str,
        file_size: int,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            file_data: File binary data
            object_name: Name to store the object as
            content_type: MIME type of the file
            file_size: Size of the file in bytes
            bucket_name: Optional bucket name (defaults to self.default_bucket)
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        
        try:
            # Ensure bucket exists before uploading
            self._ensure_bucket_exists(bucket_name)
            
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type
            )
            logger.info(f"Successfully uploaded {object_name} to {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading file {object_name}: {e}")
            return False
    
    def download_file(self, object_name: str, bucket_name: Optional[str] = None) -> Optional[bytes]:
        """
        Download a file from MinIO.
        
        Args:
            object_name: Name of the object to download
            bucket_name: Optional bucket name
            
        Returns:
            bytes: File content if successful, None otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Error downloading file {object_name}: {e}")
            return None
    
    def get_presigned_url(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        Generate a pre-signed URL for downloading a file.
        
        Args:
            object_name: Name of the object
            bucket_name: Optional bucket name
            expires: URL expiration time
            
        Returns:
            str: Pre-signed URL if successful, None otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=expires
            )
            
            # Replace internal MinIO endpoint with external endpoint for browser access
            # This handles the case where we're running in Docker
            if "minio:9000" in url:
                external_endpoint = os.getenv("MINIO_EXTERNAL_ENDPOINT", "localhost:9000")
                url = url.replace("minio:9000", external_endpoint)
            
            return url
        except S3Error as e:
            logger.error(f"Error generating pre-signed URL for {object_name}: {e}")
            return None
    
    def delete_file(self, object_name: str, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a file from MinIO.
        
        Args:
            object_name: Name of the object to delete
            bucket_name: Optional bucket name
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"Successfully deleted {object_name} from {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting file {object_name}: {e}")
            return False
    
    def file_exists(self, object_name: str, bucket_name: Optional[str] = None) -> bool:
        """
        Check if a file exists in MinIO.
        
        Args:
            object_name: Name of the object to check
            bucket_name: Optional bucket name
            
        Returns:
            bool: True if exists, False otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False


# Global instance
minio_service = MinIOService()