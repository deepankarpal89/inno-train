#!/bin/bash

# Configuration
# Detect processor architecture and set appropriate image tag
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_text_lambda"  # ARM image
elif [[ "$ARCH" == "x86_64" ]]; then
    IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_text_amd64"   # AMD64 image
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
    echo "$DOCKER_PASSWORD" | sudo docker login -u "$DOCKER_USERNAME" --password-stdin 2>&1 | tee -a "${LOG_FILE}"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
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
    sudo docker pull "${IMAGE_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    
    # Check if pull was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[$(date)] Error: Failed to pull Docker image from Docker Hub" | tee -a "${LOG_FILE}"
        echo "[$(date)] Please ensure the image ${IMAGE_NAME} exists on Docker Hub" | tee -a "${LOG_FILE}"
        exit 1
    fi
    echo "[$(date)] Successfully pulled Docker image from Docker Hub" | tee -a "${LOG_FILE}"
else
    echo "[$(date)] Using local Docker image: ${IMAGE_NAME}" | tee -a "${LOG_FILE}"
fi

# Run the Docker container
echo "[$(date)] Starting container..." | tee -a "${LOG_FILE}"

# Build Docker run command based on GPU availability
DOCKER_CMD="sudo docker run --name \"${CONTAINER_NAME}\" --rm"

if [ $GPU_AVAILABLE -eq 0 ]; then
    echo "[$(date)] Running with GPU support" | tee -a "${LOG_FILE}"
    DOCKER_CMD="$DOCKER_CMD --gpus all"
else
    echo "[$(date)] Running in CPU-only mode" | tee -a "${LOG_FILE}"
fi

DOCKER_CMD="$DOCKER_CMD -v \"$(pwd)/data:/app/data\" -v \"$(pwd)/projects_yaml:/app/projects_yaml\" -v \"$(pwd)/output:/app/output\" \"${IMAGE_NAME}\""

# Execute the Docker command
eval "$DOCKER_CMD" 2>&1 | tee -a "${LOG_FILE}"

# Check if container ran successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[$(date)] Container executed successfully" | tee -a "${LOG_FILE}" 
    # Copy any output files from container if needed
    # docker cp "${CONTAINER_NAME}:/path/in/container" "${OUTPUT_DIR}/"
    
    # Clean up
    sudo docker rm "${CONTAINER_NAME}" 2>/dev/null
    echo "[$(date)] Container cleaned up" | tee -a "${LOG_FILE}"
else
    echo "[$(date)] Error: Container execution failed" | tee -a "${LOG_FILE}"
    exit 1
fi

# Logout from Docker Hub
echo "[$(date)] Logging out from Docker Hub..." | tee -a "${LOG_FILE}"
sudo docker logout 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date)] Job completed successfully!" | tee -a "${LOG_FILE}"
echo "Logs saved to: ${LOG_FILE}"
