"""
Test script for the eval file paths endpoint.

Usage:
    python test/scripts/test_eval_file_paths_endpoint.py <job_uuid>

Example:
    python test/scripts/test_eval_file_paths_endpoint.py 514a7ace-9bd0-4ec4-8e7f-050dac536673
"""

import sys
import requests
import json
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_eval_file_paths_endpoint(
    job_uuid: str, base_url: Optional[str] = None
) -> None:
    """
    Test the eval file paths endpoint for a given job UUID.

    Args:
        job_uuid: The training job UUID to test
        base_url: Base URL of the API (default: reads from BASE_URL env variable)
    """
    # Get base URL from environment variable if not provided
    if base_url is None:
        base_url = os.getenv("BASE_URL", "http://localhost:8001/api")
    print("=" * 70)
    print("🧪 Testing Eval File Paths Endpoint")
    print("=" * 70)
    print(f"Job UUID: {job_uuid}")
    print(f"Base URL: {base_url}")
    print()

    # Construct the endpoint URL
    endpoint = f"{base_url}/v1/training/jobs/{job_uuid}/eval-file-paths"
    print(f"📡 Endpoint: {endpoint}")
    print()

    try:
        # Make the GET request
        print("🔄 Making request...")
        response = requests.get(endpoint, timeout=10)

        # Print response status
        print(f"📊 Status Code: {response.status_code}")
        print()

        # Parse and display response
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS - Response Data:")
            print("-" * 70)
            print(json.dumps(data, indent=2))
            print("-" * 70)
            print()

            # Display key information
            if data.get("success"):
                print("📁 File Paths:")
                print(f"  Train: {data.get('train_file_path')}")
                print(f"  CV:    {data.get('cv_file_path')}")
                print()
                print(f"🎯 Best Iteration: {data.get('best_iteration')}")
                print(f"🎯 Best Epoch: {data.get('best_epoch')}")
            else:
                print(
                    f"⚠️  Request succeeded but operation failed: {data.get('message')}"
                )

        elif response.status_code == 404:
            print(f"❌ NOT FOUND - Job {job_uuid} not found")
            print(f"Response: {response.text}")

        elif response.status_code == 500:
            print("❌ SERVER ERROR")
            print(f"Response: {response.text}")

        else:
            print(f"❌ UNEXPECTED STATUS CODE: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Could not connect to the API")
        print(f"   Make sure the server is running at {base_url}")

    except requests.exceptions.Timeout:
        print("❌ TIMEOUT ERROR - Request took too long")

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST ERROR: {str(e)}")

    except json.JSONDecodeError:
        print("❌ JSON DECODE ERROR - Invalid JSON response")
        print(f"Response text: {response.text}")

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")

    print()
    print("=" * 70)


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("❌ Error: Job UUID is required")
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <job_uuid> [base_url]")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} 514a7ace-9bd0-4ec4-8e7f-050dac536673")
        print(
            f"  python {sys.argv[0]} 514a7ace-9bd0-4ec4-8e7f-050dac536673 http://localhost:8000"
        )
        sys.exit(1)

    job_uuid = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else None

    test_eval_file_paths_endpoint(job_uuid, base_url)


if __name__ == "__main__":
    main()
