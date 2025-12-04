#!/usr/bin/env python3
"""
Script to start and monitor a training job via the InnoTrain API.
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
BASE_URL = os.getenv("BASE_URL")
START_TRAINING_ENDPOINT = f"{BASE_URL}/v1/training/start"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/v1/training/status"  # Adjust if needed
TRAIN_REQUEST_PATH = os.getenv("TRAIN_REQUEST_PATH")


with open(TRAIN_REQUEST_PATH, 'r') as f:
    request_data = json.load(f)

def start_training_job(request_data):
    """Start a training job and return the job ID."""
    print("🚀 Starting training job...")
    try:
        response = requests.post(
            START_TRAINING_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        print(response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ non api - Error starting training job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response ({e.response.status_code}): {e.response.text}")
        sys.exit(1)


def main():
    """Main function to start and monitor a training job."""
    result = start_training_job(request_data)
    if result and result.get('success'):
        job_uuid = result.get('job_uuid')
        print(f"✅ Training job started successfully!")
        print(f"🔗 Job UUID: {job_uuid}")
    else:
        print("❌ Failed to start training job")
        if result:
            print(f"Error: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()