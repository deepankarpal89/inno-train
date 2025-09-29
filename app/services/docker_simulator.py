import asyncio
import uuid
import logging
from typing import Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class DockerSimulator:
    """Simulates Docker operations"""

    def __init__(self):
        self.docker_execution_time = settings.docker_execution_time

    async def run_hello_world(self, instance_id: str) -> Dict[str, Any]:
        """Simulate running Docker hello-world on EC2 instance"""
        logger.info(f"🐳 Running Docker hello-world on instance {instance_id}")

        # Simulate Docker pull and run time
        await asyncio.sleep(self.docker_execution_time)

        # Simulate hello-world output
        output = """
        Hello from Docker!
        This message shows that your installation appears to be working correctly.

        To generate this message, Docker took the following steps:
        1. The Docker client contacted the Docker daemon.
        2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
        3. The Docker daemon created a new container from that image which runs the
            executable that produces the output you are currently reading.
        4. The Docker daemon streamed that output to the Docker client, which sent it
            to your terminal.

        To try something more ambitious, you can run an Ubuntu container with:
        $ docker run -it ubuntu bash
        """.strip()

        result = {
            "status": "success",
            "output": output,
            "container_id": f"container-{uuid.uuid4().hex[:8]}",
            "execution_time": f"{self.docker_execution_time:.1f}s",
        }

        logger.info(f"✅ Docker hello-world completed on instance {instance_id}")
        return result
