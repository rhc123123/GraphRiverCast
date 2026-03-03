@echo off
REM ============================================================
REM GraphRiverCast - Setup Script for Windows
REM ============================================================
REM Automatically detects GPU and CUDA version, then installs
REM the optimal PyTorch version.
REM
REM Usage:
REM   setup_windows.bat          # Auto-detect (recommended)
REM   setup_windows.bat cpu      # Force CPU-only
REM   setup_windows.bat cuda     # Force CUDA (auto-detect version)
REM
REM Prerequisites:
REM   - Anaconda or Miniconda must be installed
REM   - Run this script from Anaconda Prompt
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo   GraphRiverCast - Environment Setup for Windows
echo ============================================================
echo.

REM Parse argument
set "FORCE_MODE=%~1"
if "%FORCE_MODE%"=="" set "FORCE_MODE=auto"

REM ============================================================
REM Check Conda
REM ============================================================
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda is not found!
    echo.
    echo Please install Miniconda or Anaconda first:
    echo   Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   Anaconda:  https://www.anaconda.com/download
    echo.
    echo After installation, run this script from Anaconda Prompt.
    pause
    exit /b 1
)

echo [INFO] Conda found:
call conda --version
echo.

REM ============================================================
REM Detect GPU and CUDA
REM ============================================================
echo [Auto-Detect] Checking for NVIDIA GPU...

set "GPU_DETECTED=false"
set "CUDA_VERSION="
set "GPU_NAME="
set "GPU_ARCH="

REM Check if nvidia-smi exists
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] nvidia-smi not found. No NVIDIA GPU detected.
    goto :select_pytorch
)

REM Try to get GPU info
for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do (
    set "GPU_NAME=%%a"
    set "GPU_DETECTED=true"
)

if "%GPU_DETECTED%"=="false" (
    echo [INFO] nvidia-smi failed. No GPU available.
    goto :select_pytorch
)

echo [OK] GPU detected: %GPU_NAME%

REM Get CUDA version from nvidia-smi output
for /f "tokens=*" %%a in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do (
    set "CUDA_LINE=%%a"
)

REM Extract CUDA version number
for /f "tokens=3" %%a in ("%CUDA_LINE:CUDA Version: =CUDA Version: %") do (
    set "CUDA_VERSION=%%a"
)

REM Clean up CUDA version (remove trailing characters)
for /f "tokens=1 delims= |" %%a in ("%CUDA_VERSION%") do set "CUDA_VERSION=%%a"

echo [OK] CUDA Version: %CUDA_VERSION%

REM Get compute capability
for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul') do (
    set "GPU_ARCH_RAW=%%a"
)

REM Remove decimal point from compute capability (e.g., 8.9 -> 89)
set "GPU_ARCH=%GPU_ARCH_RAW:.=%"
echo [OK] GPU Compute Capability: sm_%GPU_ARCH%

:select_pytorch
REM ============================================================
REM Select PyTorch Version
REM ============================================================

REM Handle force modes
if "%FORCE_MODE%"=="cpu" (
    echo [INFO] Forcing CPU-only mode
    set "GPU_DETECTED=false"
)

if "%FORCE_MODE%"=="cuda" (
    if "%GPU_DETECTED%"=="false" (
        echo [ERROR] CUDA mode requested but no GPU detected!
        pause
        exit /b 1
    )
)

if not "%FORCE_MODE%"=="auto" if not "%FORCE_MODE%"=="cpu" if not "%FORCE_MODE%"=="cuda" (
    echo [ERROR] Invalid option: %FORCE_MODE%
    echo Usage: setup_windows.bat [auto^|cpu^|cuda]
    pause
    exit /b 1
)

REM Select PyTorch index URL based on detection
if "%GPU_DETECTED%"=="false" (
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "PYTORCH_DESC=CPU-only"
    set "DEVICE_FLAG=cpu"
    goto :start_install
)

REM Parse CUDA major version
for /f "tokens=1 delims=." %%a in ("%CUDA_VERSION%") do set "CUDA_MAJOR=%%a"
for /f "tokens=2 delims=." %%a in ("%CUDA_VERSION%") do set "CUDA_MINOR=%%a"

REM Check for new architecture GPUs
REM Blackwell (RTX 50 series): sm_120+
REM Ada Lovelace (RTX 40 series): sm_89+

if %GPU_ARCH% GEQ 120 (
    echo [INFO] Detected Blackwell architecture GPU ^(sm_%GPU_ARCH%^)
    echo [INFO] This requires PyTorch with CUDA 12.8+ support
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu128"
    set "PYTORCH_DESC=CUDA 12.8 (Blackwell)"
    set "DEVICE_FLAG=cuda"
    goto :start_install
)

if %GPU_ARCH% GEQ 89 (
    if %CUDA_MAJOR% GEQ 12 (
        set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "PYTORCH_DESC=CUDA 12.4"
    ) else (
        set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
        set "PYTORCH_DESC=CUDA 11.8"
    )
    set "DEVICE_FLAG=cuda"
    goto :start_install
)

if %CUDA_MAJOR% GEQ 12 (
    if %CUDA_MINOR% GEQ 4 (
        set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "PYTORCH_DESC=CUDA 12.4"
    ) else (
        set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "PYTORCH_DESC=CUDA 12.1"
    )
    set "DEVICE_FLAG=cuda"
    goto :start_install
)

if %CUDA_MAJOR% GEQ 11 (
    set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
    set "PYTORCH_DESC=CUDA 11.8"
    set "DEVICE_FLAG=cuda"
    goto :start_install
)

REM Fallback
set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"
set "PYTORCH_DESC=CUDA 11.8 (fallback)"
set "DEVICE_FLAG=cuda"

:start_install
echo.
echo [Config] PyTorch version: %PYTORCH_DESC%
echo [Config] Device flag: %DEVICE_FLAG%
echo.

REM ============================================================
REM Step 1: Create Conda Environment
REM ============================================================
echo [Step 1/3] Creating Conda environment (Python 3.11)...

REM Remove existing environment if exists
call conda env remove -n grc -y 2>nul

call conda create -n grc python=3.11 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create environment
    pause
    exit /b 1
)
echo [OK] Environment created
echo.

REM ============================================================
REM Step 2: Install PyTorch
REM ============================================================
echo [Step 2/3] Installing PyTorch (%PYTORCH_DESC%)...

call conda run -n grc pip install torch torchvision --index-url %PYTORCH_INDEX%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

REM ============================================================
REM Step 3: Install Dependencies
REM ============================================================
echo [Step 3/3] Installing other dependencies...

call conda run -n grc pip install torch-geometric numpy matplotlib scipy tqdm omegaconf
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] All dependencies installed
echo.

REM ============================================================
REM Verify Installation
REM ============================================================
echo ============================================================
echo   Verifying Installation
echo ============================================================
echo.

call conda run -n grc python -c "import torch; import torch_geometric; import numpy as np; print(f'PyTorch: {torch.__version__}'); print(f'PyTorch Geometric: {torch_geometric.__version__}'); print(f'NumPy: {np.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); exec('if torch.cuda.is_available():\n    print(f\"CUDA version: {torch.version.cuda}\")\n    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n    x = torch.randn(100, 100, device=\"cuda\")\n    y = torch.matmul(x, x)\n    print(\"CUDA test: PASSED\")')"

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo To use the environment:
echo.
echo   conda activate grc
echo.
echo To run inference:
echo   python src/inference.py --ckpt checkpoints/GRC_0.1deg.ckpt --data-dir data --group LamaH_CE06min_obs2000_2017 --device %DEVICE_FLAG%
echo.
echo To run fine-tuning:
echo   python src/finetune.py --ckpt checkpoints/GRC_0.1deg.ckpt --data-dir data --group LamaH_CE06min_obs2000_2017 --device %DEVICE_FLAG%
echo.
pause
