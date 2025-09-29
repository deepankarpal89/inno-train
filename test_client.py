#!/usr/bin/env python3
"""
Test client for InnoTrain API
Demonstrates how to interact with the training simulator
"""

import requests
import time
import json
from typing import Dict, Any


class InnoTrainClient:
    """Client for interacting with InnoTrain API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def start_training_job(
        self,
        job_name: str = "test-job",
        instance_type: str = "t2.micro",
        region: str = "us-east-1",
    ) -> Dict[str, Any]:
        """Start a new training job"""
        payload = {
            "job_name": job_name,
            "instance_type": instance_type,
            "region": region,
        }

        response = requests.post(f"{self.base_url}/train", json=payload)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        response = requests.get(f"{self.base_url}/status/{job_id}")
        response.raise_for_status()
        return response.json()

    def list_jobs(self) -> Dict[str, Any]:
        """List all training jobs"""
        response = requests.get(f"{self.base_url}/jobs")
        response.raise_for_status()
        return response.json()

    def delete_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a completed job"""
        response = requests.delete(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, job_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Wait for a job to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status["status"] in ["completed", "failed"]:
                return status

            print(f"Job {job_id} status: {status['status']}")
            time.sleep(2)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")


def main():
    """Main test function"""
    print("🚀 InnoTrain API Test Client")
    print("=" * 40)

    client = InnoTrainClient()

    try:
        # Test 1: Start a training job
        print("\n1. Starting training job...")
        job_response = client.start_training_job(
            job_name="demo-job", instance_type="t2.small", region="us-west-2"
        )

        job_id = job_response["job_id"]
        print(f"✅ Job started: {job_id}")
        print(f"   Status: {job_response['status']}")
        print(f"   Message: {job_response['message']}")

        # Test 2: Monitor job progress
        print(f"\n2. Monitoring job {job_id}...")
        final_status = client.wait_for_completion(job_id)

        print(f"✅ Job completed!")
        print(f"   Final status: {final_status['status']}")
        print(f"   Instance ID: {final_status['instance_id']}")
        print(f"   Started: {final_status['started_at']}")
        print(f"   Completed: {final_status['completed_at']}")

        # Test 3: Show Docker output
        if final_status.get("docker_result"):
            docker_result = final_status["docker_result"]
            print(f"\n3. Docker execution results:")
            print(f"   Status: {docker_result['status']}")
            print(f"   Container ID: {docker_result['container_id']}")
            print(f"   Execution time: {docker_result['execution_time']}")
            print(f"   Output preview: {docker_result['output'][:100]}...")

        # Test 4: List all jobs
        print(f"\n4. Listing all jobs...")
        jobs_response = client.list_jobs()
        print(f"   Total jobs: {jobs_response['total_jobs']}")

        for job in jobs_response["jobs"]:
            print(f"   - {job['job_id']}: {job['job_name']} ({job['status']})")

        # Test 5: Clean up
        print(f"\n5. Cleaning up job {job_id}...")
        delete_response = client.delete_job(job_id)
        print(f"✅ {delete_response['message']}")

        print(f"\n🎉 All tests completed successfully!")

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to InnoTrain API")
        print("   Make sure the server is running: python main.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
