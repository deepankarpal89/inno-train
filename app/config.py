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
    db_type: str = Field("postgresql", alias="DB_TYPE")  # 'postgresql' or 'sqlite'
    db_name: str = Field(..., alias="DB_NAME")
    db_user: str = Field(None, alias="DB_USER")
    db_password: str = Field(None, alias="DB_PASSWORD")
    db_host: str = Field(None, alias="DB_HOST")
    db_port: Union[str, int] = Field(None, alias="DB_PORT")

    # Storage settings (AWS S3 only)
    storage_type: str = Field("aws_s3", alias="STORAGE_TYPE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Make environment variable names case-insensitive
        extra = "ignore"  # Ignore extra fields in .env that aren't in the model

    @property
    def db_url(self) -> str:
        """Construct the database URL based on database type."""
        if self.db_type.lower() == "sqlite":
            # For SQLite, use aiosqlite driver
            return f"sqlite+aiosqlite:///./{self.db_name}"
        else:
            # For PostgreSQL, use asyncpg driver
            return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Create settings instance
settings = Settings()
