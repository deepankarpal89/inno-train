"""
Script to get accuracy metrics via the InnoTrain API endpoint
"""

import argparse
import sys
from dotenv import load_dotenv
import requests
import os
import json
from typing import Dict, Any

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/api")


def get_accuracy_metrics(job_uuid: str) -> Dict[str, Any]:
    """Get accuracy metrics from the InnoTrain API endpoint."""
    url = f"{BASE_URL}/v1/training/jobs/{job_uuid}/metrics/accuracy"
    print(f"🌐 Making request to: {url}")

    try:
        response = requests.get(url, timeout=1000)
        print(f"📡 Response Status: {response.status_code}")

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
    """Main function to get accuracy metrics."""
    print(f"\n🔍 Fetching accuracy metrics for job: {job_uuid}")
    print("=" * 70)

    result = get_accuracy_metrics(job_uuid)

    print("\n📊 Results:")
    print("=" * 70)

    if result and isinstance(result, dict):
        if result.get("success", False):
            print(f"✅ Accuracy metrics retrieved successfully!")
            print(f"\n🔗 Job UUID: {result.get('job_uuid')}")
            print(f"🔢 Total Iterations: {result.get('iterations', 0)}")

            metrics = result.get("metrics", {})
            if metrics:
                print(f"\n📈 Overall Job Metrics:")
                print(f"  - Train Accuracy: {metrics.get('train_accuracy')}")
                print(f"  - Eval Accuracy: {metrics.get('eval_accuracy')}")
                print(f"  - Best Eval UUID: {metrics.get('best_eval_uuid')}")

            train_accs = result.get("train_accuracies", [])
            eval_accs = result.get("eval_accuracies", [])

            if train_accs or eval_accs:
                print(f"\n📊 Per-Iteration Accuracies:")
                for i, (train_acc, eval_acc) in enumerate(
                    zip(train_accs, eval_accs), 1
                ):
                    print(f"  Iteration {i}:")
                    print(f"    - Train: {train_acc}")
                    print(f"    - Eval:  {eval_acc}")

            print(f"\n💬 Message: {result.get('message')}")
            return
        else:
            print(f"⚠️  Request completed but job not ready")
            print(f"📝 Message: {result.get('message', 'No message provided')}")
            print(f"\n💡 This usually means:")
            print(f"   - Job is still running")
            print(f"   - Job hasn't completed yet")
            print(f"   - Metrics haven't been populated")
    else:
        print(f"❌ Unexpected response format: {result}")

    print("\n🔧 Debug Info:")
    print(f"- Base URL: {BASE_URL}")
    print(f"- Job UUID: {job_uuid}")
    print("\n💡 Tip: Make sure the server is running and the job has completed")

    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get accuracy metrics via the InnoTrain API endpoint."
    )
    parser.add_argument("job_uuid", type=str, help="UUID of the training job")
    args = parser.parse_args()
    main(args.job_uuid)
