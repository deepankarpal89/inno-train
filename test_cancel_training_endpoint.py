"""
Script to cancel a training job via the InnoTrain API.
"""

import argparse
import sys
from typing import Dict, Any
from dotenv import load_dotenv
import requests
import os

# Configuration
load_dotenv()
BASE_URL = os.getenv("BASE_URL")


def cancel_training_job(job_uuid: str) -> Dict[str, Any]:
    """Cancel a training job and return the response.
    
    Args:
        job_uuid: The UUID of the training job to cancel
        
    Returns:
        Dict containing the API response
        
    Raises:
        SystemExit: If the request fails
    """
    print("🚀 Canceling training job...")
    cancel_endpoint = f"{BASE_URL}/v1/training/jobs/{job_uuid}/cancel"
    
    try:
        response = requests.post(
            cancel_endpoint,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        print(response.json())
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Failed to cancel job: HTTP {e.response.status_code}")
        if e.response.text:
            print(f"Error details: {e.response.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error while canceling job: {e}")
        sys.exit(1)


def main(job_uuid: str) -> None:
    """Main function to cancel a training job.
    
    Args:
        job_uuid: The UUID of the training job to cancel
    """
    result = cancel_training_job(job_uuid)
    
    if result and result.get('success'):
        job_uuid = result.get('job_uuid')
        print(f"✅ Training job canceled successfully!")
        print(f"🔗 Job UUID: {job_uuid}")
    else:
        print("❌ Failed to cancel training job")
        if result:
            print(f"Error: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cancel a training job via the InnoTrain API."
    )
    parser.add_argument(
        "job_uuid", 
        type=str, 
        help="UUID of the training job to cancel"
    )
    args = parser.parse_args()
    main(args.job_uuid)