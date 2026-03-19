#!/bin/bash
# ============================================================
# ScalForex — RunPod L40 Setup Script
# Run: bash scripts/setup_runpod.sh
# ============================================================
set -e

echo "============================================"
echo " ScalForex — RunPod L40 Setup"
echo "============================================"

# 1. Navigate to persistent workspace
cd /workspace
echo "[1/6] Working directory: $(pwd)"

# 2. Clone repo (skip if exists)
if [ ! -d "ScalForex" ]; then
    echo "[2/6] Cloning ScalForex repo..."
    git clone https://github.com/nguyenvu1892/Scalp4.git ScalForex
else
    echo "[2/6] ScalForex exists, pulling latest..."
    cd ScalForex && git pull && cd ..
fi

cd ScalForex

# 3. Install dependencies
echo "[3/6] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision torchaudio \
    polars \
    stable-baselines3 \
    wandb \
    pydantic \
    numpy \
    gymnasium \
    pyyaml \
    python-dotenv

# 4. GPU check
echo "[4/6] GPU Status:"
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    print(f'  GPU: {gpu} ({mem:.1f}GB)')
    print(f'  Compute Capability: {cap[0]}.{cap[1]}')
    print(f'  AMP (Mixed Precision): SUPPORTED')
    print(f'  torch.compile: SUPPORTED')
    print(f'  BF16: {\"YES\" if cap[0] >= 8 else \"NO\"}')
    # Check existing GPU usage
    used = torch.cuda.memory_allocated(0) / 1e9
    print(f'  Current VRAM used: {used:.2f}GB')
else:
    print('  NO GPU FOUND!')
    exit(1)
"

# 5. Check existing processes (Propfirm bot)
echo "[5/6] Existing GPU processes:"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo "  No GPU processes running"
echo ""
echo "  WARNING: Propfirm bot may be using ~13% GPU."
echo "  Our training is capped to ~80% VRAM max."

# 6. Create checkpoint directory
mkdir -p /workspace/checkpoints/transformer
echo "[6/6] Checkpoint dir ready: /workspace/checkpoints/transformer/"

echo ""
echo "============================================"
echo " Setup COMPLETE! Run training with:"
echo "   python scripts/train_runpod.py --wandb"
echo "============================================"
