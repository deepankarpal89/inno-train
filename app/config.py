from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Union


class Settings(BaseSettings):
    """Application settings and configuration."""

    # Application settings
    app_name: str = "InnoTrain"
    app_version: str = "1.0.0"
    app_description: str = "EC2 Docker Training Simulator"
    debug: bool = False

    # Server configuration
    port: int = Field(8001, alias="PORT")  # Default port 8001 if not specified in .env
    host: str = "0.0.0.0"  # Default host to listen on

    def __init__(self, **data):
        super().__init__(**data)
        print(f"[CONFIG] Server will run on {self.host}:{self.port}")
        print(f"[CONFIG] Environment PORT: {data.get('PORT', 'Not set')}")
        print(f"[CONFIG] Final port value: {self.port}")

    # Simulation timings (in seconds)
    ec2_startup_time: float = 2.0
    ec2_termination_time: float = 1.0
    docker_execution_time: float = 3.0

    # Lambda API settings
    lambda_api_key: str = Field(..., alias="LAMBDA_API_KEY")

    # SSH settings
    ssh_key_path: str = Field(..., alias="SSH_KEY_PATH")
    ssh_key_password: str = Field(..., alias="SSH_KEY_PASSWORD")
    ssh_key_name: str = Field(..., alias="SSH_KEY_NAME")

    # Database settings
    db_name: str = Field(..., alias="DB_NAME")
    db_user: str = Field(..., alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_host: str = Field(..., alias="DB_HOST")
    db_port: Union[str, int] = Field(..., alias="DB_PORT")

    # Storage settings
    storage_type: str = Field(..., alias="STORAGE_TYPE")

    # MinIO settings
    minio_endpoint: str = Field(..., alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_secure: Union[bool, str] = Field(..., alias="MINIO_SECURE")
    minio_api_port: Union[str, int] = Field(..., alias="MINIO_API_PORT")
    minio_console_port: Union[str, int] = Field(..., alias="MINIO_CONSOLE_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Make environment variable names case-insensitive
        extra = "ignore"  # Ignore extra fields in .env that aren't in the model

    @property
    def minio_config(self) -> dict:
        """Get MinIO configuration as a dictionary."""
        return {
            "endpoint": self.minio_endpoint,
            "access_key": self.minio_access_key,
            "secret_key": self.minio_secret_key,
            "secure": (
                self.minio_secure
                if isinstance(self.minio_secure, bool)
                else self.minio_secure.lower() == "true"
            ),
        }

    @property
    def db_url(self) -> str:
        """Construct the database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Create settings instance
settings = Settings()
