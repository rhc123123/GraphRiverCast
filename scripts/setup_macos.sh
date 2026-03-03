#!/bin/bash
# ============================================================
# GraphRiverCast - Setup Script for macOS
# ============================================================
# Automatically detects Apple Silicon (M1/M2/M3/M4) vs Intel,
# and installs the optimal PyTorch version with MPS support.
#
# Usage:
#   ./setup_macos.sh          # Auto-detect (recommended)
#   ./setup_macos.sh cpu      # Force CPU-only
#   ./setup_macos.sh mps      # Force MPS (Apple Silicon only)
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
echo -e "${BLUE}  GraphRiverCast - Environment Setup for macOS${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================
# Detect macOS Architecture
# ============================================================
detect_architecture() {
    echo -e "${CYAN}[Auto-Detect]${NC} Checking macOS architecture..."

    # Get architecture
    ARCH=$(uname -m)

    if [ "$ARCH" = "arm64" ]; then
        IS_APPLE_SILICON=true
        # Get specific chip info
        CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        echo -e "${GREEN}[OK]${NC} Apple Silicon detected: ${CHIP_INFO}"
        echo -e "${GREEN}[OK]${NC} MPS (Metal Performance Shaders) acceleration available"
    else
        IS_APPLE_SILICON=false
        CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Intel")
        echo -e "${GREEN}[OK]${NC} Intel Mac detected: ${CHIP_INFO}"
        echo -e "${YELLOW}[INFO]${NC} CPU-only mode will be used"
    fi
}

# ============================================================
# Main Script
# ============================================================

# Parse argument
FORCE_MODE=${1:-auto}

echo -e "${YELLOW}[INFO]${NC} Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Detect architecture
detect_architecture

# Handle force modes and select device
case $FORCE_MODE in
    auto)
        if [ "$IS_APPLE_SILICON" = true ]; then
            DEVICE_FLAG="mps"
            PYTORCH_DESC="Apple Silicon (MPS)"
        else
            DEVICE_FLAG="cpu"
            PYTORCH_DESC="CPU-only (Intel)"
        fi
        ;;
    cpu)
        echo -e "${YELLOW}[INFO]${NC} Forcing CPU-only mode"
        DEVICE_FLAG="cpu"
        PYTORCH_DESC="CPU-only (forced)"
        ;;
    mps)
        if [ "$IS_APPLE_SILICON" = false ]; then
            echo -e "${RED}[ERROR]${NC} MPS mode requested but Intel Mac detected!"
            echo "MPS is only available on Apple Silicon (M1/M2/M3/M4)"
            exit 1
        fi
        DEVICE_FLAG="mps"
        PYTORCH_DESC="Apple Silicon (MPS)"
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Invalid option: $FORCE_MODE"
        echo "Usage: ./setup_macos.sh [auto|cpu|mps]"
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
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
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

if [ "$IS_APPLE_SILICON" = true ] && [ "$DEVICE_FLAG" != "cpu" ]; then
    # Apple Silicon - use default PyTorch (includes MPS support)
    uv pip install torch torchvision
else
    # Intel Mac - use CPU-only version
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

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
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    # Quick test on MPS
    x = torch.randn(100, 100, device='mps')
    y = torch.matmul(x, x)
    print('MPS test: PASSED')
else:
    # Quick test on CPU
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)
    print('CPU test: PASSED')
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

if [ "$DEVICE_FLAG" = "mps" ]; then
    echo -e "${CYAN}[TIP]${NC} For best performance on Apple Silicon, use --device mps"
    echo -e "${CYAN}[TIP]${NC} If you encounter issues, try --device cpu as fallback"
    echo ""
fi
