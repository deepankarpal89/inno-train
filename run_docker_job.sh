#!/bin/bash

# Configuration
# Detect processor architecture and set appropriate image tag
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_text_lambda"  # ARM image
elif [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_combined_amd64"   # AMD64 image
else
    IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_text_multi"   # Multi-arch image for other architectures
fi
OUTPUT_DIR="output"
LOG_FILE="${OUTPUT_DIR}/execution.log"
CONTAINER_NAME="innotrain"
DOCKER_USERNAME="deepankarpal89"
DOCKER_PASSWORD="yadadocker7"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "[$(date)] Starting Docker job..." | tee -a "${LOG_FILE}"

# Function to check GPU availability
check_gpu_availability() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[$(date)] Warning: nvidia-smi not found. GPU may not be available." | tee -a "${LOG_FILE}"
        return 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        echo "[$(date)] Warning: GPU not accessible. Running in CPU mode." | tee -a "${LOG_FILE}"
        return 1
    fi
    
    echo "[$(date)] GPU detected and accessible" | tee -a "${LOG_FILE}"
    return 0
}

# Function to handle Docker login
login_to_docker() {
    # Check if credentials are provided as environment variables
    if [[ -z "$DOCKER_USERNAME" || -z "$DOCKER_PASSWORD" ]]; then
        echo "Docker credentials not provided as environment variables."
        read -p "Docker username: " DOCKER_USERNAME
        read -s -p "Docker password: " DOCKER_PASSWORD
        echo ""
    fi
    
    echo "[$(date)] Logging into Docker Hub..." | tee -a "${LOG_FILE}"
    
    # Capture login output and exit code
    LOGIN_OUTPUT=$(mktemp)
    echo "$DOCKER_PASSWORD" | sudo docker login -u "$DOCKER_USERNAME" --password-stdin > "$LOGIN_OUTPUT" 2>&1
    LOGIN_EXIT_CODE=$?
    
    cat "$LOGIN_OUTPUT" | tee -a "${LOG_FILE}"
    rm -f "$LOGIN_OUTPUT"
    
    if [ $LOGIN_EXIT_CODE -ne 0 ]; then
        echo "[$(date)] Error: Docker login failed" | tee -a "${LOG_FILE}"
        return 1
    fi
    return 0
}

# Check GPU availability
check_gpu_availability
GPU_AVAILABLE=$?

# Login to Docker Hub
login_to_docker || exit 1

# Check if Docker image exists locally, otherwise pull from Docker Hub
echo "[$(date)] Checking for local Docker image: ${IMAGE_NAME}" | tee -a "${LOG_FILE}"
if ! sudo docker image inspect "${IMAGE_NAME}" &> /dev/null; then
    echo "[$(date)] Local image not found, pulling from Docker Hub: ${IMAGE_NAME}" | tee -a "${LOG_FILE}"
    
    # Capture pull output and exit code
    PULL_OUTPUT=$(mktemp)
    sudo docker pull "${IMAGE_NAME}" > "$PULL_OUTPUT" 2>&1
    PULL_EXIT_CODE=$?
    
    cat "$PULL_OUTPUT" | tee -a "${LOG_FILE}"
    rm -f "$PULL_OUTPUT"
    
    # Check if pull was successful
    if [ $PULL_EXIT_CODE -ne 0 ]; then
        echo "[$(date)] Error: Failed to pull Docker image from Docker Hub" | tee -a "${LOG_FILE}"
        echo "[$(date)] Please ensure the image ${IMAGE_NAME} exists on Docker Hub" | tee -a "${LOG_FILE}"
        exit 1
    fi
    echo "[$(date)] Successfully pulled Docker image from Docker Hub" | tee -a "${LOG_FILE}"
    sleep 5
else
    echo "[$(date)] Using local Docker image: ${IMAGE_NAME}" | tee -a "${LOG_FILE}"
fi

# Run the Docker container
echo "[$(date)] Starting container..." | tee -a "${LOG_FILE}"

# Build Docker run command based on GPU availability
DOCKER_CMD="docker run --name \"${CONTAINER_NAME}\""

if [ $GPU_AVAILABLE -eq 0 ]; then
    echo "[$(date)] Running with GPU support" | tee -a "${LOG_FILE}"
    DOCKER_CMD="$DOCKER_CMD --gpus all"
else
    echo "[$(date)] Running in CPU-only mode" | tee -a "${LOG_FILE}"
fi

DOCKER_CMD="$DOCKER_CMD -v \"$(pwd)/data:/app/data\" -v \"$(pwd)/projects_yaml:/app/projects_yaml\" -v \"$(pwd)/output:/app/output\" \"${IMAGE_NAME}\""

# Execute the Docker command and stream output
# Note: Not using --rm to ensure we can get exit code reliably
echo "[$(date)] Executing Docker container..." | tee -a "${LOG_FILE}"
echo "[$(date)] Command: $DOCKER_CMD" | tee -a "${LOG_FILE}"

# Run container and capture output while it runs
# Use set -o pipefail to ensure we get the docker command's exit code, not tee's
set -o pipefail
eval "$DOCKER_CMD" 2>&1 | tee -a "${LOG_FILE}"
DOCKER_EXIT_CODE=$?
set +o pipefail

# Check if container ran successfully
echo "[$(date)] Container finished with exit code: $DOCKER_EXIT_CODE" | tee -a "${LOG_FILE}"

if [ "$DOCKER_EXIT_CODE" -eq 0 ]; then
    echo "[$(date)] Container executed successfully" | tee -a "${LOG_FILE}"
else
    echo "[$(date)] Error: Container execution failed with exit code $DOCKER_EXIT_CODE" | tee -a "${LOG_FILE}"
    # Show last 50 lines of container logs for debugging
    echo "[$(date)] Last 50 lines of container logs:" | tee -a "${LOG_FILE}"
    docker logs --tail 50 "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}"
fi

# Clean up container
echo "[$(date)] Removing container..." | tee -a "${LOG_FILE}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Exit with container's exit code
if [ "$DOCKER_EXIT_CODE" -ne 0 ]; then
    echo "[$(date)] Script exiting with code $DOCKER_EXIT_CODE" | tee -a "${LOG_FILE}"
    exit 1
fi

# Logout from Docker Hub
echo "[$(date)] Logging out from Docker Hub..." | tee -a "${LOG_FILE}"
docker logout 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date)] Job completed successfully!" | tee -a "${LOG_FILE}"
echo "Logs saved to: ${LOG_FILE}"
