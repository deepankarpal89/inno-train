#!/bin/bash

# Configuration
IMAGE_NAME="deepankarpal89/innotone:ddp_rlhf_text_lambda"  # Default image, can be overridden with command line argument
OUTPUT_DIR="output"
LOG_FILE="${OUTPUT_DIR}/execution.log"
CONTAINER_NAME="innotrain"
DOCKER_USERNAME="deepankarpal89"
DOCKER_PASSWORD="yadadocker7"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "[$(date)] Starting Docker job..." | tee -a "${LOG_FILE}"

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

# Login to Docker Hub
login_to_docker || exit 1

# Pull the Docker image
echo "[$(date)] Pulling Docker image: ${IMAGE_NAME}" | tee -a "${LOG_FILE}"
sudo docker pull "${IMAGE_NAME}" 2>&1 | tee -a "${LOG_FILE}"

# Check if pull was successful
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[$(date)] Error: Failed to pull Docker image" | tee -a "${LOG_FILE}"
    exit 1
fi

# Run the Docker container
echo "[$(date)] Starting container..." | tee -a "${LOG_FILE}"
sudo docker run --name "${CONTAINER_NAME}" \
    --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/projects_yaml:/app/projects_yaml" \
    -v "$(pwd)/output:/app/output" \
    "${IMAGE_NAME}" 2>&1 | tee -a "${LOG_FILE}"

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
