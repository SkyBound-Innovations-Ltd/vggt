#!/bin/bash
#
# VGGT-P Service Docker Run Script
#
# Runs the VGGT depth estimation service container with proper GPU flags
# for Tesla T4 (15GB VRAM).
#
# Usage:
#   ./docker-run.sh          # Run with default settings
#   ./docker-run.sh --build  # Rebuild image before running
#
# Base URL: http://13.40.25.32:8011/vggt_p

set -e

IMAGE_NAME="vggt-t4:latest"
CONTAINER_NAME="vggt-service"

# Check for --build flag
if [ "$1" == "--build" ]; then
    echo "Building Docker image..."
    docker build -f Dockerfile.vggt-t4 -t ${IMAGE_NAME} .
fi

# Stop existing container if running
if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
    echo "Stopping existing container..."
    docker stop ${CONTAINER_NAME}
fi

# Remove existing container if exists
if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
    echo "Removing existing container..."
    docker rm ${CONTAINER_NAME}
fi

# Create directories if they don't exist
mkdir -p "$(pwd)/uploads"
mkdir -p "$(pwd)/outputs"

echo "Starting VGGT-P Service..."
echo "Base URL: http://13.40.25.32:8011/vggt_p"

# Add --dev flag for development mode (mounts static files for live reload)
DEV_MOUNT=""
if [ "$1" == "--dev" ] || [ "$2" == "--dev" ]; then
    echo "Development mode: mounting static files for live reload"
    DEV_MOUNT="-v $(pwd)/service/static:/workspace/service/static"
fi

docker run -d \
    --name ${CONTAINER_NAME} \
    --runtime=nvidia \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e SERVICE_PORT=8011 \
    --shm-size=8g \
    --health-cmd="curl -f http://localhost:8011/vggt_p/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-start-period=180s \
    --health-retries=3 \
    -v "$(pwd)/uploads:/workspace/uploads" \
    -v "$(pwd)/outputs:/workspace/outputs" \
    ${DEV_MOUNT} \
    ${IMAGE_NAME}

echo ""
echo "Container started. Waiting for service to be ready..."
echo "Note: Model loading takes 30-60 seconds on first startup."
echo ""
echo "Check status with:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "Access the service:"
echo "  Web UI:     http://100.102.61.99:8011/vggt_p/  (Tailscale)"
echo "  Web UI:     http://localhost:8011/vggt_p/      (Local)"
echo "  API Docs:   http://localhost:8011/docs"
echo "  Health:     http://localhost:8011/vggt_p/health"
echo ""
echo "API endpoints:"
echo "  POST /vggt_p/upload       - Upload video + flight log"
echo "  POST /vggt_p/process      - Start processing"
echo "  GET  /vggt_p/jobs/{id}    - Check job status"
echo "  GET  /vggt_p/download/... - Download results"
