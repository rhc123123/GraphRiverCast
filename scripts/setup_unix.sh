#!/bin/bash
# ============================================================
# GraphRiverCast - Unified Setup Script for Linux / WSL
# ============================================================
# Automatically detects GPU and CUDA version, then installs
# the optimal PyTorch version.
#
# Usage:
#   ./setup_unix.sh          # Auto-detect (recommended)
#   ./setup_unix.sh cpu      # Force CPU-only
#   ./setup_unix.sh cuda     # Force CUDA (auto-detect version)
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  GraphRiverCast - Environment Setup${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================
# Function: Detect GPU and CUDA
# ============================================================
detect_gpu() {
    echo -e "${CYAN}[Auto-Detect]${NC} Checking for NVIDIA GPU..."

    # Check if nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}[INFO]${NC} nvidia-smi not found. No NVIDIA GPU detected."
        GPU_DETECTED=false
        return
    fi

    # Try to run nvidia-smi
    if ! nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}[INFO]${NC} nvidia-smi failed. GPU driver may not be installed."
        GPU_DETECTED=false
        return
    fi

    GPU_DETECTED=true

    # Get GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1)
    echo -e "${GREEN}[OK]${NC} GPU detected: ${GPU_NAME}"

    # Get driver CUDA version
    DRIVER_CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
    CUDA_VERSION_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -n1)
    echo -e "${GREEN}[OK]${NC} CUDA Version: ${CUDA_VERSION_FULL}"

    # Parse major.minor
    CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f2)

    # Get GPU compute capability
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
    echo -e "${GREEN}[OK]${NC} GPU Compute Capability: sm_${GPU_ARCH}"
}

# ============================================================
# Function: Select PyTorch version based on GPU
# ============================================================
select_pytorch_version() {
    if [ "$GPU_DETECTED" = false ]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
        PYTORCH_DESC="CPU-only"
        DEVICE_FLAG="cpu"
        return
    fi

    # Check for new architecture GPUs (Blackwell: sm_120+, Ada Lovelace: sm_89+)
    # These require newer CUDA versions

    if [ "$GPU_ARCH" -ge 120 ]; then
        # Blackwell architecture (RTX 50 series) - requires CUDA 12.8+
        echo -e "${YELLOW}[INFO]${NC} Detected Blackwell architecture GPU (sm_${GPU_ARCH})"
        echo -e "${YELLOW}[INFO]${NC} This requires PyTorch with CUDA 12.8+ support"
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
        PYTORCH_DESC="CUDA 12.8 (Blackwell)"
        DEVICE_FLAG="cuda"
    elif [ "$GPU_ARCH" -ge 89 ]; then
        # Ada Lovelace architecture (RTX 40 series)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
            PYTORCH_DESC="CUDA 12.4"
        else
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
            PYTORCH_DESC="CUDA 11.8"
        fi
        DEVICE_FLAG="cuda"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        # CUDA 12.x
        if [ "$CUDA_MINOR" -ge 4 ]; then
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
            PYTORCH_DESC="CUDA 12.4"
        else
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
            PYTORCH_DESC="CUDA 12.1"
        fi
        DEVICE_FLAG="cuda"
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        # CUDA 11.x
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
        PYTORCH_DESC="CUDA 11.8"
        DEVICE_FLAG="cuda"
    else
        # Older CUDA or unknown
        echo -e "${YELLOW}[WARNING]${NC} CUDA version ${CUDA_VERSION_FULL} may not be fully supported"
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
        PYTORCH_DESC="CUDA 11.8 (fallback)"
        DEVICE_FLAG="cuda"
    fi
}

# ============================================================
# Main Script
# ============================================================

# Parse argument
FORCE_MODE=${1:-auto}

echo -e "${YELLOW}[INFO]${NC} Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Detect GPU
detect_gpu

# Handle force modes
case $FORCE_MODE in
    auto)
        select_pytorch_version
        ;;
    cpu)
        echo -e "${YELLOW}[INFO]${NC} Forcing CPU-only mode"
        GPU_DETECTED=false
        PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
        PYTORCH_DESC="CPU-only (forced)"
        DEVICE_FLAG="cpu"
        ;;
    cuda)
        if [ "$GPU_DETECTED" = false ]; then
            echo -e "${RED}[ERROR]${NC} CUDA mode requested but no GPU detected!"
            exit 1
        fi
        select_pytorch_version
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Invalid option: $FORCE_MODE"
        echo "Usage: ./setup_unix.sh [auto|cpu|cuda]"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}[Config]${NC} PyTorch version: ${PYTORCH_DESC}"
echo -e "${CYAN}[Config]${NC} Device flag: ${DEVICE_FLAG}"
echo ""

# ============================================================
# Step 1: Install UV
# ============================================================
echo -e "${GREEN}[Step 1/4]${NC} Installing UV package manager..."

if command -v uv &> /dev/null; then
    echo -e "${YELLOW}[INFO]${NC} UV is already installed: $(uv --version)"
else
    echo -e "${YELLOW}[INFO]${NC} Downloading and installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    # Add to shell config
    SHELL_RC=""
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi

    if [ -n "$SHELL_RC" ] && ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> "$SHELL_RC"
        echo -e "${YELLOW}[INFO]${NC} Added UV to PATH in $SHELL_RC"
    fi

    echo -e "${GREEN}[OK]${NC} UV installed: $(uv --version)"
fi

# ============================================================
# Step 2: Create Virtual Environment
# ============================================================
echo ""
echo -e "${GREEN}[Step 2/4]${NC} Creating virtual environment..."

if [ -d ".venv" ]; then
    echo -e "${YELLOW}[INFO]${NC} Removing existing virtual environment..."
    rm -rf .venv
fi

uv venv .venv --python 3.11 --prompt grc
source .venv/bin/activate

echo -e "${GREEN}[OK]${NC} Virtual environment created"
echo -e "${YELLOW}[INFO]${NC} Python: $(python --version)"

# ============================================================
# Step 3: Install PyTorch
# ============================================================
echo ""
echo -e "${GREEN}[Step 3/4]${NC} Installing PyTorch (${PYTORCH_DESC})..."

uv pip install torch torchvision --index-url "$PYTORCH_INDEX"

echo -e "${GREEN}[OK]${NC} PyTorch installed"

# ============================================================
# Step 4: Install Dependencies
# ============================================================
echo ""
echo -e "${GREEN}[Step 4/4]${NC} Installing other dependencies..."

uv pip install torch-geometric numpy matplotlib scipy tqdm omegaconf

echo -e "${GREEN}[OK]${NC} All dependencies installed"

# ============================================================
# Verify Installation
# ============================================================
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Verifying Installation${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

python -c "
import torch
import torch_geometric
import numpy as np

print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU compute capability: {torch.cuda.get_device_capability(0)}')
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print('CUDA test: PASSED')
"

# ============================================================
# Done
# ============================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "  ${YELLOW}cd $PROJECT_DIR${NC}"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "To run inference:"
echo -e "  ${YELLOW}python src/inference.py --ckpt checkpoints/GRC_0.1deg.ckpt --data-dir data --group LamaH_CE06min_obs2000_2017 --device $DEVICE_FLAG${NC}"
echo ""
echo -e "To run fine-tuning:"
echo -e "  ${YELLOW}python src/finetune.py --ckpt checkpoints/GRC_0.1deg.ckpt --data-dir data --group LamaH_CE06min_obs2000_2017 --device $DEVICE_FLAG${NC}"
echo ""
