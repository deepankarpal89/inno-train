from fastapi import Depends
from typing import Generator

from app.services.ec2_simulator import EC2Simulator
from app.services.docker_simulator import DockerSimulator


def get_ec2_simulator() -> Generator[EC2Simulator, None, None]:
    """Dependency that provides an EC2Simulator instance"""
    yield EC2Simulator()


def get_docker_simulator() -> Generator[DockerSimulator, None, None]:
    """Dependency that provides a DockerSimulator instance"""
    yield DockerSimulator()
