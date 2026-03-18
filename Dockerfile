# ScalForex — Docker Deployment
# Dual Xeon 56t + 96GB RAM + GTX 750 Ti
# 24/7 operation for Crypto (BTCUSD, ETHUSD) xuyên cuối tuần

FROM python:3.12-slim

# System dependencies for MetaTrader5 (Wine on Linux) and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs checkpoints reports

# Health check — verify heartbeat file is fresh
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import time; t=float(open('logs/heartbeat.txt').read()); exit(0 if time.time()-t < 120 else 1)" || exit 1

# Default: run the main trading orchestrator
CMD ["python", "-m", "scripts.live_trade"]
