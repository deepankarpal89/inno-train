from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    app_name: str = "InnoTrain"
    app_version: str = "1.0.0"
    app_description: str = "EC2 Docker Training Simulator"
    debug: bool = False

    # Simulation timings (in seconds)
    ec2_startup_time: float = 2.0
    ec2_termination_time: float = 1.0
    docker_execution_time: float = 3.0

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
