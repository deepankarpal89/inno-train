#!/usr/bin/env python3
"""
Script to start and monitor a training job via the InnoTrain API.

Commands to run this script:

1. Make sure the FastAPI server is running in another terminal:
   uvicorn app.main:app --reload --port 8001

2. Set up environment variables in .env file:
   BASE_URL=http://localhost:8001/api
   TRAIN_REQUEST_PATH=path/to/your/request.json

3. Run this script:
   python start_training.py
"""

import os
import sys
import json
import time
import requests
from pprint import pprint
from pathlib import Path
from dotenv import load_dotenv

# Configuration
load_dotenv()

# Check required environment variables
PORT = os.getenv("PORT", "8000")

# Construct BASE_URL if not set explicitly
BASE_URL = os.getenv("BASE_URL")
if not BASE_URL:
    BASE_URL = f"http://localhost:{PORT}/api"
    print(f"BASE_URL not set, using default: {BASE_URL}")

TRAIN_REQUEST_PATH = os.getenv("TRAIN_REQUEST_PATH")
if not TRAIN_REQUEST_PATH:
    print("❌ Error: TRAIN_REQUEST_PATH environment variable is not set")
    print("Please set TRAIN_REQUEST_PATH in your .env file")
    sys.exit(1)

# Endpoints
START_TRAINING_ENDPOINT = f"{BASE_URL}/v1/training/start"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/v1/training/jobs"  # Updated to match actual endpoint


# Load the training request data
try:
    with open(TRAIN_REQUEST_PATH, "r") as f:
        raw_data = json.load(f)

    # Extract the actual request data from the nested structure if needed
    if "data" in raw_data and "request_data" in raw_data["data"]:
        request_data = raw_data["data"]["request_data"]
    else:
        request_data = raw_data

    print(f"Loaded training request from {TRAIN_REQUEST_PATH}")

except Exception as e:
    print(f"❌ Error loading training request: {e}")
    sys.exit(1)


def start_training_job(request_data):
    """Start a training job and return the job ID."""
    print("🚀 Starting training job...")
    try:
        response = requests.post(
            START_TRAINING_ENDPOINT,
            json={"data": {"request_data": request_data}},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        print(response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ non api - Error starting training job: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response ({e.response.status_code}): {e.response.text}")
        sys.exit(1)


def check_job_status(job_uuid):
    """Check the status of a job by its UUID."""
    status_url = f"{JOB_STATUS_ENDPOINT}/{job_uuid}"
    try:
        response = requests.get(status_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking job status: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response ({e.response.status_code}): {e.response.text}")
        return None


def main():
    """Main function to start and monitor a training job."""
    # Print info about the request
    print(f"\n📋 Training Request Summary:")
    print(f"  Project ID: {request_data.get('project', {}).get('id')}")
    print(f"  Project Name: {request_data.get('project', {}).get('name')}")
    print(f"  Training Run ID: {request_data.get('training_run_id')}")
    print(f"\n🔌 API Endpoint: {START_TRAINING_ENDPOINT}")

    # Start the training job
    result = start_training_job(request_data)

    if result and result.get("success"):
        job_uuid = result.get("job_uuid")
        print(f"\n✅ Training job started successfully!")
        print(f"🔗 Job UUID: {job_uuid}")
        print(f"🔍 Check status at: {JOB_STATUS_ENDPOINT}/{job_uuid}")

        # Optional: Check initial status
        print("\n📊 Checking initial job status...")
        status = check_job_status(job_uuid)
        if status:
            print(f"  Status: {status.get('status', 'unknown')}")
    else:
        print("\n❌ Failed to start training job")
        if result:
            print(f"  Error: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
