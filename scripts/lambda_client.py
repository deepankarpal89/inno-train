# lambda_cloud/client.py
import requests
from requests.auth import HTTPBasicAuth
import os
from typing import List, Optional
from dotenv import load_dotenv
import time
import logging


class LambdaClient:
    """A client for interacting with the Lambda Cloud API."""

    BASE_URL = "https://cloud.lambda.ai/api/v1"

    def __init__(self, api_key: str = None, base_url: str = None, logger=None):
        """Initialize the Lambda Cloud client."""
        load_dotenv()

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        if api_key is None:

            api_key = os.getenv("LAMBDA_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided and LAMBDA_API_KEY environment variable not set"
                )

        self.ssh_key_name = os.getenv("SSH_KEY_NAME")

        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(api_key, "")
        self.base_url = base_url or self.BASE_URL

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request to the Lambda Cloud API."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            self.logger.info(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                self.logger.info(f"Response: {e.response.text}")
            raise

    def list_available_instances(self):
        """Get the cheapest available GPU instance."""
        data = self._make_request("GET", "/instance-types")["data"]

        # Filter for available A100 instances (x86 architecture only)
        avail_gpus = {
            k: v
            for k, v in data.items()
            if len(v["regions_with_capacity_available"]) > 0
            and self.get_gpu_architecture(k) == "x86"
            and "a10" in k.lower()
        }

        # Check if any instances are available
        if not avail_gpus:
            self.logger.info("No A100 GPU instances currently available")
            return None

        # Sort by price and select cheapest
        sorted_gpus = sorted(
            avail_gpus.items(),
            key=lambda x: x[1]["instance_type"]["price_cents_per_hour"],
        )
        # sorted_gpus = self.filter_eight_gpu_instances(sorted_gpus)
        print(f"sorted_gpus: {sorted_gpus[0]}")
        selected_gpu = sorted_gpus[0][0]
        selected_region = avail_gpus[selected_gpu]["regions_with_capacity_available"][
            0
        ]["name"]

        resp = {"name": selected_gpu, "region": selected_region}
        return resp

    def filter_eight_gpu_instances(self, sorted_gpus):
        """Filter for 8 GPU instances."""
        return [x for x in sorted_gpus if x[1]["instance_type"]["specs"]["gpus"] == 8]

    @staticmethod
    def get_gpu_architecture(instance_type_name: str) -> str:
        """
        Determine if GPU instance uses ARM or x86 architecture.

        Args:
            instance_type_name: Name of the instance type (e.g., 'gpu_1x_gh200')

        Returns:
            'ARM' for ARM-based instances, 'x86' for AMD/Intel-based instances
        """
        # GH200 (Grace Hopper) uses ARM architecture
        arm_indicators = ["gh200", "grace"]

        instance_lower = instance_type_name.lower()

        if any(indicator in instance_lower for indicator in arm_indicators):
            return "ARM"
        else:
            return "x86"

    def list_instances(self):
        """List all Lambda Cloud instances."""
        resp = self._make_request("GET", "/instances")
        return resp["data"]

    def get_instance(self, instance_id: str):
        """Get details of a specific instance."""
        resp = self._make_request("GET", f"/instances/{instance_id}")
        return resp["data"] if resp.get("data") else None

    def get_instance_status(self, instance_id: str):
        """Get the status of a specific instance."""
        resp = self.get_instance(instance_id)
        return resp["status"] if resp else None

    def terminate_instance(self, instance_id: str):
        """Terminate a specific instance."""
        try:
            self._make_request(
                "POST",
                "/instance-operations/terminate",
                json={"instance_ids": [instance_id]},
            )
            return True
        except Exception as e:
            self.logger.info(f"Failed to terminate instance {instance_id}: {e}")
            return False

    def list_ssh_keys(self):
        """List all available SSH keys."""
        resp = self._make_request("GET", "/ssh-keys")
        return resp["data"]

    def launch_instance(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: List[str] = None,
        name: str = None,
        file_system_names: List[str] = None,
        **kwargs,
    ):
        """Launch a new instance."""
        payload = {
            "instance_type_name": instance_type_name,
            "region_name": region_name,
            "ssh_key_names": ssh_key_names or [self.ssh_key_name],
            **kwargs,
        }

        if name:
            payload["name"] = name
        if file_system_names:
            payload["file_system_names"] = file_system_names

        try:
            resp = self._make_request(
                "POST", "/instance-operations/launch", json=payload
            )
            instance_id = resp["data"]["instance_ids"][0]
            self.logger.info(
                f"Instance {instance_id} launched, waiting for active state..."
            )

            # Poll for active status with retry logic
            max_wait_time = 600  # 10 minutes max
            poll_interval = 10  # Start with 10 seconds
            elapsed_time = 0
            retry_count = 0
            max_retries = 3

            while elapsed_time < max_wait_time:
                try:
                    status = self.get_instance_status(instance_id)

                    if status == "active":
                        self.logger.info(f"Instance {instance_id} is active.")
                        return self.get_instance(instance_id)

                    self.logger.info(
                        f"Status: {status}, waiting {poll_interval}s... (elapsed: {elapsed_time}s)"
                    )
                    time.sleep(poll_interval)
                    elapsed_time += poll_interval
                    retry_count = 0  # Reset retry count on successful status check

                except Exception as poll_error:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.info(
                            f"Max retries exceeded during polling: {poll_error}"
                        )
                        # Instance might still be launching, return what we have
                        self.logger.info(
                            f"Returning instance {instance_id} (may not be active yet)"
                        )
                        return self.get_instance(instance_id)

                    self.logger.info(
                        f"Polling error (retry {retry_count}/{max_retries}): {poll_error}"
                    )
                    time.sleep(5)  # Short wait before retry
                    elapsed_time += 5

            # Timeout reached
            self.logger.info(
                f"Timeout waiting for instance to be active after {max_wait_time}s"
            )
            return self.get_instance(instance_id)

        except Exception as e:
            self.logger.info(f"Failed to launch instance: {e}")
            return None

    def list_file_systems(self):
        """List all available file systems."""
        resp = self._make_request("GET", "/file-systems")
        return resp["data"] if resp else None

    def get_instance_file_systems(self, instance_id: str):
        """Get file systems attached to a specific instance."""
        instance = self.get_instance(instance_id)
        if not instance:
            return []

        all_file_systems = self.list_file_systems()

        return [
            fs
            for fs in all_file_systems
            if fs["name"] in (instance["file_system_names"] or [])
        ]

    def get_file_system(self, identifier: str, by_name: bool = False):
        """Get a file system by ID or name."""
        file_systems = self.list_file_systems()
        if by_name:
            for fs in file_systems:
                if fs["name"] == identifier:
                    return fs
        else:
            for fs in file_systems:
                if fs["id"] == identifier:
                    return fs
        return None


def main():
    import json

    client = LambdaClient()

    # print("=" * 70)
    # print("CURRENTLY RUNNING INSTANCES")
    # print("=" * 70)
    # instances = client.list_instances()
    # print(json.dumps(instances, indent=2))

    # print("\n" + "=" * 70)
    # print("AVAILABLE GPU INSTANCE TYPES")
    # print("=" * 70)
    # available_response = client._make_request("GET", "/instance-types")
    # print(json.dumps(available_response, indent=2))

    print("\n" + "=" * 70)
    print("FILTERED AVAILABLE GPU INSTANCES")
    print("=" * 70)
    cheapest = client.list_available_instances()
    if cheapest:
        print(f"Cheapest available: {json.dumps(cheapest, indent=2)}")
    else:
        print("No GPU instances currently available")


if __name__ == "__main__":
    main()
