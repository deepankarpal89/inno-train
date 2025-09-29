# InnoTrain - EC2 Docker Training Simulator

A FastAPI application that simulates the process of setting up an EC2 server and running Docker hello-world containers. This simulator provides a realistic workflow without actually provisioning AWS resources.

## Features

- 🚀 **EC2 Instance Simulation**: Simulates creating and terminating EC2 instances
- 🐳 **Docker Hello-World**: Runs simulated Docker hello-world containers
- 📊 **Job Tracking**: Track multiple training jobs with unique IDs
- 🔄 **Async Processing**: Non-blocking job execution using asyncio
- 📝 **Comprehensive Logging**: Detailed logs for all operations
- 🌐 **REST API**: Full REST API with OpenAPI documentation

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at:

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### POST /train

Start a new training job that simulates EC2 setup and Docker execution.

**Request Body:**

```json
{
  "job_name": "my-training-job",
  "instance_type": "t2.micro",
  "region": "us-east-1"
}
```

**Response:**

```json
{
  "job_id": "job-abc12345",
  "status": "initializing",
  "message": "Training job my-training-job started successfully",
  "instance_id": "pending",
  "started_at": "2024-01-15T10:30:00"
}
```

### GET /status/{job_id}

Check the status of a specific training job.

**Response:**

```json
{
  "job_id": "job-abc12345",
  "status": "completed",
  "job_name": "my-training-job",
  "instance_type": "t2.micro",
  "region": "us-east-1",
  "instance_id": "i-def67890",
  "started_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:30:15",
  "docker_result": {
    "status": "success",
    "output": "Hello from Docker!...",
    "container_id": "container-xyz98765",
    "execution_time": "2.3s"
  }
}
```

### GET /jobs

List all training jobs.

### DELETE /jobs/{job_id}

Delete a completed job from memory.

## Job Workflow

Each training job follows this workflow:

1. **Initializing** - Job created and queued
2. **Creating Instance** - Simulating EC2 instance creation (2s)
3. **Instance Running** - EC2 instance is ready
4. **Running Docker** - Executing Docker hello-world (3s)
5. **Terminating Instance** - Cleaning up EC2 instance (1s)
6. **Completed** - Job finished successfully

## Job Status Values

- `initializing` - Job just started
- `creating_instance` - Setting up EC2 instance
- `instance_running` - EC2 instance is active
- `running_docker` - Executing Docker container
- `terminating_instance` - Cleaning up resources
- `completed` - Job finished successfully
- `failed` - Job encountered an error

## Example Usage

### Using curl

```bash
# Start a training job
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "test-job",
    "instance_type": "t2.small",
    "region": "us-west-2"
  }'

# Check job status
curl "http://localhost:8000/status/job-abc12345"

# List all jobs
curl "http://localhost:8000/jobs"
```

### Using Python requests

```python
import requests
import time

# Start training job
response = requests.post("http://localhost:8000/train", json={
    "job_name": "python-test",
    "instance_type": "t2.micro",
    "region": "us-east-1"
})

job_data = response.json()
job_id = job_data["job_id"]
print(f"Started job: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8000/status/{job_id}")
    status_data = status_response.json()

    print(f"Status: {status_data['status']}")

    if status_data["status"] in ["completed", "failed"]:
        print("Job finished!")
        print(f"Docker output: {status_data.get('docker_result', {}).get('output', 'N/A')}")
        break

    time.sleep(2)
```

## Simulation Details

### EC2 Simulation

- Generates realistic instance IDs (e.g., `i-abc12345`)
- Simulates startup time (2 seconds)
- Simulates termination time (1 second)
- Logs all operations with emojis for clarity

### Docker Simulation

- Simulates Docker pull and run operations (3 seconds)
- Returns authentic hello-world output
- Generates container IDs and execution metrics
- Includes realistic timing information

## Development

### Project Structure

```
innotone-training/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .venv/              # Virtual environment
```

### Adding Features

The application is designed to be easily extensible:

- **New simulators**: Add classes similar to `EC2Simulator` and `DockerSimulator`
- **Additional endpoints**: Add new FastAPI routes
- **Enhanced logging**: Modify the logging configuration
- **Persistent storage**: Replace in-memory storage with a database

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `main.py` or kill the existing process
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Virtual environment**: Make sure you've activated the virtual environment

### Logs

The application provides detailed logging. Look for these log patterns:

- 🚀 Instance creation
- ✅ Successful operations
- 🐳 Docker operations
- 🛑 Instance termination
- ❌ Errors
- 🎉 Job completion

## License

This project is for educational and simulation purposes.
