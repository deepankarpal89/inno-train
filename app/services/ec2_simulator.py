import asyncio
import uuid
import logging
from typing import Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class EC2Simulator:
    """Simulates EC2 instance operations"""

    def __init__(self):
        self.ec2_startup_time = settings.ec2_startup_time
        self.ec2_termination_time = settings.ec2_termination_time

    async def create_instance(self, instance_type: str, region: str) -> Dict[str, Any]:
        """Simulate EC2 instance creation"""
        instance_id = f"i-{uuid.uuid4().hex[:8]}"
        logger.info(
            f"🚀 Creating EC2 instance {instance_id} ({instance_type}) in {region}"
        )

        # Simulate instance startup time
        await asyncio.sleep(self.ec2_startup_time)

        instance_info = {
            "instance_id": instance_id,
            "instance_type": instance_type,
            "region": region,
            "state": "running",
            "public_ip": f"52.{uuid.uuid4().hex[:3]}.{uuid.uuid4().hex[:3]}.{uuid.uuid4().hex[:2]}",
            "launch_time": int(time.time()),
        }

        logger.info(f"✅ EC2 instance {instance_id} is running")
        return instance_info

    async def terminate_instance(self, instance_id: str) -> bool:
        """Simulate EC2 instance termination"""
        logger.info(f"🛑 Terminating EC2 instance {instance_id}")

        # Simulate termination time
        await asyncio.sleep(self.ec2_termination_time)

        logger.info(f"✅ EC2 instance {instance_id} terminated")
        return True
