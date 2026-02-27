# FLUX.2 Deployment Guide

Complete instructions for deploying FLUX.2 Professional in various environments.

---

## Table of Contents

1. [Native Runtime Development](#native-runtime-development)
2. [Multi-Worker Local Deployment](#multi-worker-local-deployment)
3. [GPU Configuration](#gpu-configuration)
4. [Environment Variables](#environment-variables)
5. [Systemd Service (Linux)](#systemd-service-linux)
6. [Windows Service (NSSM)](#windows-service-nssm)
7. [Health Checks & Monitoring](#health-checks--monitoring)
8. [Deployment Hardening Profiles](#deployment-hardening-profiles)
9. [Troubleshooting](#troubleshooting)

---

## Native Runtime Development

### Prerequisites

- Python 3.10+ installed
- Virtual environment tool (venv, conda, etc.)
- GPU drivers (NVIDIA or AMD) or CPU-only mode
- 15-20 GB free disk space

### Setup (15 minutes)

#### Step 1: Clone Repository

```bash
git clone https://github.com/black-forest-labs/flux.git flux2
cd flux2
```

#### Step 2: Create Virtual Environment

**Using venv**:
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

**Using conda**:
```bash
conda create -n flux2 python=3.11
conda activate flux2
```

#### Step 3: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install torch==2.8.0 torchvision==0.23.0
pip install transformers==4.56.1 safetensors==0.4.5
pip install streamlit==1.35.0
pip install einops==0.8.1
pip install prometheus-client==0.21.0
```

#### Step 4: Set Environment Variables

```bash
# HuggingFace token (required for some models)
export HF_TOKEN=hf_your_token_here

# Model paths (optional, auto-detected)
export KLEIN_4B_MODEL_PATH=/path/to/flux-2-klein-4b.safetensors
export AE_MODEL_PATH=/path/to/ae.safetensors

# Performance tuning
export FLUX2_RESULT_CACHE_SIZE=100
export FLUX2_PROGRESS_PREVIEW_INTERVAL=4

# API integration
export OPENROUTER_API_KEY=sk_your_key
export OLLAMA_HOST=http://127.0.0.1:11434
```

**Windows PowerShell**:
```powershell
$env:HF_TOKEN = "hf_your_token_here"
$env:FLUX2_LOCAL_ONLY = "1"
```

#### Step 5: Launch Application

```bash
# Start Streamlit app
streamlit run ui_flux2_professional.py

# With custom port
streamlit run ui_flux2_professional.py --server.port=8501

# With custom host
streamlit run ui_flux2_professional.py --server.address=0.0.0.0
```

**Opens at**: `http://localhost:8501`

---

## Multi-Worker Local Deployment

Deploy multiple UI workers with load balancing for higher throughput.

### Architecture

```
Internet
    ↓
Reverse Proxy (Nginx)
    ├─→ UI Worker 1 (port 8501)
    ├─→ UI Worker 2 (port 8502)
    └─→ UI Worker 3 (port 8503)
         ↓
    Shared Model Weights
    Shared Cache (Redis optional)
```

### Step 1: Install Load Balancer (Nginx)

**Linux**:
```bash
sudo apt-get install nginx
```

**macOS** (Homebrew):
```bash
brew install nginx
```

**Windows**: Download from https://nginx.org/en/download.html

### Step 2: Configure Nginx Reverse Proxy

Create `nginx.conf`:

```nginx
http {
    upstream flux2_ui {
        # Round-robin load balancing
        server 127.0.0.1:8501;
        server 127.0.0.1:8502;
        server 127.0.0.1:8503;
    }
    
    server {
        listen 80;
        server_name localhost;
        client_max_body_size 100M;
        
        location / {
            proxy_pass http://flux2_ui;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Streamlit specific
            proxy_buffering off;
            proxy_request_buffering off;
        }
    }
}
```

### Step 3: Start Load Balancer

```bash
# Linux/macOS
nginx -c $(pwd)/nginx.conf

# Verify
curl http://localhost
```

### Step 4: Start UI Workers

Terminal 1:
```bash
export PYTHONUNBUFFERED=1
streamlit run ui_flux2_professional.py --server.port=8501
```

Terminal 2:
```bash
export PYTHONUNBUFFERED=1
streamlit run ui_flux2_professional.py --server.port=8502
```

Terminal 3:
```bash
export PYTHONUNBUFFERED=1
streamlit run ui_flux2_professional.py --server.port=8503
```

### Step 5: Test Load Balancing

```bash
# Send request
curl http://localhost

# Check which worker handled it (in logs)
# Should see rotation: 8501 → 8502 → 8503 → 8501
```

---

## GPU Configuration

### Single GPU Setup

**Auto-detection** (simplest):
```bash
streamlit run ui_flux2_professional.py
# Automatically uses GPU:0
```

**Explicit Selection**:
```bash
export CUDA_VISIBLE_DEVICES=0
streamlit run ui_flux2_professional.py
```

### Multi-GPU Setup

**Distribute Models Across GPUs**:

```bash
# GPU 0: Text encoder + VAE
export CUDA_VISIBLE_DEVICES=0
# GPU 1: Flow transformer
export FLUX2_GPU_ID=1

# Start workers on different GPUs
export CUDA_VISIBLE_DEVICES=0 streamlit run ui_flux2_professional.py --server.port=8501
export CUDA_VISIBLE_DEVICES=1 streamlit run ui_flux2_professional.py --server.port=8502
```

**Check Available GPUs**:

```bash
# PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0))"

# NVIDIA
nvidia-smi

# AMD
rocm-smi
```

### GPU Memory Optimization

#### CPU Offloading

```bash
export FLUX2_CPU_OFFLOAD=1
streamlit run ui_flux2_professional.py
```

#### Attention Slicing (Reduced VRAM)

```bash
export FLUX2_ATTN_SLICING=1
streamlit run ui_flux2_professional.py
```

#### Quantization (Advanced)

```python
# In streamlit UI, select:
# Settings → Data Type → int8
# (requires extra memory but faster)
```

---

## Environment Variables

Complete reference of all configurable environment variables.

### Model Configuration

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `KLEIN_4B_MODEL_PATH` | Auto-detect | Path to Klein 4B weights | `/weights/flux-2-klein-4b.safetensors` |
| `KLEIN_9B_MODEL_PATH` | Auto-detect | Path to Klein 9B weights | `/weights/flux-2-klein-9b.safetensors` |
| `AE_MODEL_PATH` | Auto-detect | Path to AutoEncoder | `/weights/ae.safetensors` |
| `FLUX2_DEFAULT_MODEL` | `flux.2-klein-4b` | Default on startup | `flux.2-klein-base-4b` |
| `HF_TOKEN` | (required) | HuggingFace API token | Get from huggingface.co/settings/tokens |

### Performance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUX2_RESULT_CACHE_SIZE` | `100` | Number of generations to cache in memory |
| `FLUX2_PROGRESS_PREVIEW_INTERVAL` | `4` | Update preview every N steps |
| `FLUX2_DEFAULT_STEPS` | `4` | Default inference steps |
| `FLUX2_DEFAULT_GUIDANCE` | `3.5` | Default guidance strength |
| `FLUX2_CPU_OFFLOAD` | `0` | Enable CPU offloading (0/1) |
| `FLUX2_ATTN_SLICING` | `0` | Enable attention slicing (0/1) |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (optional) | OpenRouter API key for upsampling |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama service URL |
| `UPSAMPLER_BACKEND` | `none` | Default upsampler: none, openrouter, ollama |

### Safety Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUX2_SAFETY_LEVEL` | `moderate` | Safety filtering: strict, moderate, permissive |
| `FLUX2_DISABLE_SAFETY` | `0` | Disable all safety checks (development only) |

### Logging & Debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `FLUX2_LOG_LEVEL` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `FLUX2_LOG_DIR` | `logs/` | Directory for log files |
| `PYTHONUNBUFFERED` | `0` | Unbuffered Python output (set to 1 for logs) |

### GPU Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | All | Visible GPU devices (e.g., "0,1") |
| `PYTORCH_CUDA_ALLOC_CONF` | (default) | CUDA memory allocation strategy |

---

## Systemd Service (Linux)

Run FLUX.2 as a background service that auto-starts on reboot.

### Step 1: Create Service File

```bash
sudo nano /etc/systemd/system/flux2.service
```

Paste:

```ini
[Unit]
Description=FLUX.2 Professional Image Generator
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/flux2
Environment="PATH=/home/username/flux2/venv/bin"
Environment="HF_TOKEN=hf_xxxxx"
Environment="PYTHONUNBUFFERED=1"

ExecStart=/home/username/flux2/venv/bin/streamlit run ui_flux2_professional.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --logger.level=info

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Replace:
- `username` - Your username
- `/home/username/flux2` - Your repo path
- `hf_xxxxx` - Your HF token

### Step 2: Enable & Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable flux2

# Start service
sudo systemctl start flux2

# Check status
sudo systemctl status flux2

# View logs
sudo journalctl -u flux2 -f  # Follow logs
sudo journalctl -u flux2 --since "1 hour ago"  # Last hour
```

### Step 3: Manage Service

```bash
# Stop
sudo systemctl stop flux2

# Restart
sudo systemctl restart flux2

# Disable auto-start
sudo systemctl disable flux2
```

---

## Windows Service (NSSM)

Run FLUX.2 as a Windows Service using NSSM (Non-Sucking Service Manager).

### Step 1: Download NSSM

```powershell
# Download from: https://nssm.cc/download
# Or use Chocolatey
choco install nssm
```

### Step 2: Create Service

```powershell
# Run as Administrator
$venvPath = "C:\flux2\venv\Scripts\streamlit.exe"
$workDir = "C:\flux2"

nssm install flux2 $venvPath `
    run ui_flux2_professional.py `
    --server.port=8501 `
    --server.address=0.0.0.0

# Or use the GUI
nssm install flux2
# Fill in: Path (streamlit.exe), Arguments, Working Directory
```

### Step 3: Configure Service

```powershell
# Set environment variables
nssm set flux2 AppEnvironmentExtra HF_TOKEN=hf_xxxxx
nssm set flux2 AppEnvironmentExtra PYTHONUNBUFFERED=1

# Set startup type
nssm set flux2 Start SERVICE_AUTO_START

# Set restart behavior
nssm set flux2 AppRestartDelay 10000  # 10 seconds
```

### Step 4: Start Service

```powershell
# Start
Start-Service flux2

# Check status
Get-Service flux2

# View logs
Get-EventLog -LogName Application -Source nssm
```

### Step 5: Manage Service

```powershell
# Stop
Stop-Service flux2

# Restart
Restart-Service flux2

# Remove
nssm remove flux2 confirm
```

---

## Health Checks & Monitoring

### Manual Health Check

```bash
# Check if app is running
curl http://localhost:8501

# Should return HTML page
# Status 200 = Running
```

### Automated Health Check Script

```bash
#!/bin/bash
# health_check.sh

URL="http://localhost:8501"
TIMEOUT=5
MAX_RETRIES=3

for i in {1..MAX_RETRIES}; do
    if curl -s -m $TIMEOUT "$URL" > /dev/null; then
        echo "✅ FLUX.2 is running"
        exit 0
    fi
    echo "Attempt $i/$MAX_RETRIES failed, retrying..."
    sleep 2
done

echo "❌ FLUX.2 is down!"
exit 1
```

Run periodically:
```bash
# Every 5 minutes via cron
*/5 * * * * /home/user/flux2/health_check.sh >> /tmp/health_check.log 2>&1
```

### Prometheus Monitoring

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'flux2'
    static_configs:
      - targets: ['127.0.0.1:8501']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Access metrics at: `http://localhost:8501/metrics`

### Grafana Dashboard

Create dashboard showing:
- Generation duration
- Cache hit rate
- GPU memory usage
- Queue length
- Error rate

---

## Deployment Hardening Profiles

Use one of these repeatable profiles to standardize rollout and rollback behavior.

### 1) Local Pilot

- Health endpoints: `/health`, `/health/live`, `/health/ready`
- Graceful degradation: fallback to CPU, disable advanced modules, reduce queue concurrency
- Rollback: restore previous Python environment lock and restart app service
- Version pinning: pinned dependencies in `requirements.txt`/`pyproject.toml`

### 2) Team Server

- Health endpoints: `/health`, `/ready`
- Graceful degradation: enable saturation fallback, pause workshop lanes, prioritize P0/P1 queues
- Rollback: switch to last known good service revision and replay config snapshot
- Version pinning: container/runtime tag plus pinned Python dependencies

### 3) Enterprise Managed

- Health endpoints: `/health`, `/health/live`, `/health/ready`
- Graceful degradation: feature-flag rollback, regional failover, read-only governance mode
- Rollback: deployment ring rollback with signed release manifest verification
- Version pinning: immutable artifact digest and infrastructure template pinning

### Validation Checklist

Before promotion, validate:

1. Startup health checks return expected results (`/health`, readiness probes).
2. Graceful degradation actions are documented and tested for the selected profile.
3. Rollback runbook steps are executable without ad-hoc manual edits.
4. Runtime and dependency versions are pinned and reproducible.

## Troubleshooting

### ❌ "Port 8501 already in use"

```bash
# Find process on port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill process (if safe)
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
streamlit run ui_flux2_professional.py --server.port=8502
```

### ❌ "CUDA out of memory"

```bash
# Enable CPU offloading
export FLUX2_CPU_OFFLOAD=1

# Or reduce resolution in UI
```

### ❌ "Model weights not found"

```bash
# Pre-download models
python -c "from flux2.util import load_flow_model; \
           load_flow_model('flux-2-klein-4b')"

# Or set model path explicitly
export KLEIN_4B_MODEL_PATH=/path/to/weights
```

### ❌ "Nginx connection refused"

```bash
# Check nginx is running
ps aux | grep nginx

# Restart nginx
nginx -s reload

# Check upstream servers responding
curl http://127.0.0.1:8501
```

### ❌ Application Crashes

```bash
# Run with verbose logging
FLUX2_LOG_LEVEL=DEBUG streamlit run ui_flux2_professional.py

# Check system resources
free -h  # Memory
df -h    # Disk space
nvidia-smi  # GPU
```

---

## Production Checklist

Before deploying to production:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] HF_TOKEN set (if needed)
- [ ] Models pre-downloaded
- [ ] Environment variables configured
- [ ] Load balancer tested (if multi-worker)
- [ ] Health checks working
- [ ] Logs configured
- [ ] Monitoring set up
- [ ] Scaling strategy defined
- [ ] Backup plan for failures
- [ ] Documentation updated
- [ ] Security review passed
- [ ] Performance benchmarked

---

## See Also

- [Getting Started](GETTING_STARTED.md)
- [Performance Tuning](PERFORMANCE.md)
- [Architecture](ARCHITECTURE.md)
- [Error Handling](ERROR_HANDLING.md)
