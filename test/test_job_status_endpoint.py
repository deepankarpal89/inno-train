"""
Script to get job status via the InnoTrain API endpoint
"""

import argparse
import sys
from dotenv import load_dotenv
import requests
import os
import json
from typing import Dict, Any

load_dotenv()
# Default to port 8001 if BASE_URL is not set in .env
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api")


def get_job_status(job_uuid: str) -> Dict[str, Any]:
    """Get job status from the InnoTrain API endpoint."""
    url = f"{BASE_URL}/v1/training/jobs/{job_uuid}"
    print(f"🌐 Making request to: {url}")

    try:
        response = requests.get(url, timeout=1000)
        print(f"📡 Response Status: {response.status_code}")

        # Try to parse JSON, but handle cases where response might not be JSON
        try:
            result = response.json()
            print("📦 Response JSON:", json.dumps(result, indent=2))
            return result
        except ValueError:
            print(f"⚠️  Response is not JSON: {response.text}")
            return {
                "success": False,
                "message": f"Invalid JSON response: {response.text}",
            }

    except requests.exceptions.HTTPError as e:
        error_msg = f"❌ HTTP Error: {e.response.status_code}"
        if e.response.text:
            error_msg += f"\n📝 Details: {e.response.text}"
        print(error_msg)
        return {"success": False, "message": str(e)}

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Request failed: {str(e)}"
        print(error_msg)
        return {"success": False, "message": str(e)}


def main(job_uuid: str) -> None:
    """Main function to get job status."""
    print(f"\n🔍 Checking status for job: {job_uuid}")
    print("-" * 50)

    result = get_job_status(job_uuid)

    print("\n📊 Results:")
    print("-" * 50)

    if result and isinstance(result, dict):
        if result.get("success", False):
            print(f"✅ Job status retrieved successfully!")
            print(f"🔗 Job UUID: {result.get('job_uuid')}")
            print(f"🔄 Status: {result.get('status', 'unknown')}")
            return
        else:
            print(f"❌ Failed to get job status")
            print(f"📝 Error: {result.get('message', 'No error details provided')}")
    else:
        print(f"❌ Unexpected response format: {result}")

    # Print additional debug info
    print("\n🔧 Debug Info:")
    print(f"- Base URL: {BASE_URL}")
    print(f"- Job UUID: {job_uuid}")
    print("\n💡 Tip: Make sure the server is running and the job UUID is correct")

    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get job status via the InnoTrain API endpoint."
    )
    parser.add_argument("job_uuid", type=str, help="UUID of the training job")
    args = parser.parse_args()
    main(args.job_uuid)
