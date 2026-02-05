#!/bin/bash
# Start VGGT-P Service

set -e

echo "========================================"
echo "Starting VGGT-P Service"
echo "========================================"

# Check if GPU is available
echo -e "\n1. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Check Python dependencies
echo -e "\n2. Checking Python dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from vggt.models.vggt import VGGT; print('VGGT: OK')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')"

# Start service
echo -e "\n3. Starting service on http://0.0.0.0:8000"
echo "   Press Ctrl+C to stop"
echo "========================================"
echo ""

uvicorn main_v3_service:app --host 0.0.0.0 --port 8000 --reload
