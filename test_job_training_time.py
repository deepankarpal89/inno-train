#!/usr/bin/env python3
import asyncio
import sys
import argparse
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
BASE_URL = os.getenv("BASE_URL")

def get_job_training_time(job_uuid: str, base_url: str = BASE_URL):
    """
    Call the training time endpoint
    
    Args:
        job_uuid: UUID of the training job
        base_url: Base URL of the API server
    """
    endpoint = f"{base_url}/v1/training/jobs/{job_uuid}/time"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Success for job {job_uuid}:")
        print(json.dumps(data, indent=2))
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Get training time information for a job")
    parser.add_argument("job_uuid", help="UUID of the training job")
    parser.add_argument("--url", default=BASE_URL, 
                       help="Base URL of the API server (default: BASE_URL from .env)")
    
    args = parser.parse_args()
    
    print(f"🔍 Getting training time for job: {args.job_uuid}")
    print(f"🌐 API URL: {args.url}")
    print("-" * 50)
    
    result = get_job_training_time(args.job_uuid, args.url)
    
    if result:
        print("-" * 50)
        print("✅ Request completed successfully")
    else:
        print("-" * 50)
        print("❌ Request failed")
        sys.exit(1)

if __name__ == "__main__":
    main()