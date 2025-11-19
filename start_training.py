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

# Configuration
BASE_URL = "http://localhost:8001/api"
START_TRAINING_ENDPOINT = f"{BASE_URL}/v1/training/start"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/v1/training/status"  # Adjust if needed
TRAIN_REQUEST_PATH = "app/api/train_request.json"

request_data = {
            "training_run_id": "7b1be4c4-084d-46d7-948d-12b04b26b049",
            "project": {
                "id": "7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd",
                "name": "spam local",
                "description": "testing spam on local",
                "task_type": "text_classification"
            },
            "prompt": {
                "id": "c2e62aa3-cb7f-43e1-ba57-5817909ef04a",
                "name": "short prompt",
                "content": "You are a helpful assistant that classifies text messages as spam or not spam. For each message, respond with 'spam' or 'not spam'"
            },
            "train_dataset": {
                "id": "996c0774-5676-41d9-a94b-b5b78d694828",
                "name": "spam train",
                "file_url": "http://localhost:9000/innotone-media/media/projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/train/spam_train_test.csv",
                "file_name": "projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/train/spam_train_test.csv",
                "file_format": "csv",
                "text_col": "text",
                "label_col": "label",
                "think_col": ""
            },
            "eval_dataset": {
                "id": "a11fe5b8-639a-4486-b215-f3e1b4e29216",
                "name": "spam test",
                "file_url": "http://localhost:9000/innotone-media/media/projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/eval/spam_cv_test.csv",
                "file_name": "projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/eval/spam_cv_test.csv",
                "file_format": "csv",
                "text_col": "text",
                "label_col": "label",
                "think_col": ""
            },
        }

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