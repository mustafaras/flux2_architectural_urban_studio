# FLUX.2 Phase 10: Production Deployment Guide

Complete setup guide for native (non-Docker) production deployment with load balancing, monitoring, and auto-scaling.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Linux Deployment (systemd)](#linux-deployment-systemd)
4. [Windows Deployment (NSSM)](#windows-deployment-nssm)
5. [Load Balancer Setup](#load-balancer-setup)
6. [Monitoring & Observability](#monitoring--observability)
7. [Health Checks & Auto-Recovery](#health-checks--auto-recovery)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+ / macOS 10.15+
- **CPU**: 16+ cores (Xeon or similar)
- **RAM**: 64 GB minimum
- **GPU**: 3+ NVIDIA GPUs (40GB+ VRAM each recommended)
- **Storage**: 200+ GB SSD (models + outputs)
- **Network**: 1Gbps+ connectivity

### Required Software

```bash
# Linux/macOS
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3-pip \
    git curl wget \
    nginx redis-server rabbitmq-server \
    prometheus grafana-server \
    supervisor \
    nvidia-driver-latest nvidia-utils \
    docker.io docker-compose  # Optional

# Windows (PowerShell as Administrator)
choco install python nginx redis rabbitmq nssm
# Download NSSM: https://nssm.cc/download
```

### Python Environment

```bash
cd /opt/flux2

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install supervisor prometheus-client redis pika
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Nginx (Port 80)                         │
│         Load Balancer + Reverse Proxy                      │
└────────┬─────────────────┬────────────────┬────────────────┘
         │                 │                │
    ┌────▼────┐        ┌───▼────┐      ┌───▼────┐
    │UI Worker│        │UI Work │      │UI Work │
    │(8501)   │        │ (8502) │      │ (8503) │
    └────┬────┘        └───┬────┘      └───┬────┘
         │                 │                │
    ┌────▼─────┬───────────┼───────────┬───▼───────┐
    │           │           │           │           │
┌───▼────┐  ┌──▼──┐   ┌────▼───┐  ┌───▼──┐   ┌───▼──┐
│ Redis  │  │Model│   │Service │  │ Node │   │ Prom │
│Cache   │  │Work │   │  Disc  │  │ Expr │   │etheus│
│        │  │(8600)   │        │  │      │   │      │
└────────┘  └──────┘  └────────┘  └──────┘   └──────┘
                                    ↓
                            ┌───────────────┐
                            │  Grafana UI   │
                            │  (port 3000)  │
                            └───────────────┘
```

### Component Breakdown

| Component | Port | Role | Count |
|-----------|------|------|-------|
| Nginx | 80 | Load balancer | 1 |
| Streamlit UI | 8501-8503 | Frontend | 3 |
| Model Worker 4B | 8600 | Inference | 1 |
| Model Worker 9B | 8601 | Inference | 1 |
| Redis | 6379 | Cache layer | 1 |
| RabbitMQ | 5672 | Message queue | 1 |
| Prometheus | 9090 | Metrics | 1 |
| Grafana | 3000 | Dashboards | 1 |

---

## Linux Deployment (systemd)

### 1. Create System User

```bash
sudo useradd -m -s /bin/bash flux2
sudo usermod -aG docker flux2  # If using Docker
```

### 2. Setup Directory Structure

```bash
sudo mkdir -p /opt/flux2
sudo cp -r . /opt/flux2
sudo chown -R flux2:flux2 /opt/flux2

# Create logging directories
sudo mkdir -p /var/log/supervisor /var/log/nginx /var/log/redis /var/log/rabbitmq
sudo chown -R flux2:flux2 /var/log/supervisor
```

### 3. Install systemd Services

```bash
# Copy service files
sudo cp deployment/systemd/*.service /etc/systemd/system/

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable flux2.service
sudo systemctl enable flux2-nginx.service
sudo systemctl enable flux2-prometheus.service
sudo systemctl enable flux2-grafana.service

# Start supervisord (manages UI/model workers)
sudo systemctl start flux2.service

# Wait for startup
sleep 10

# Check status
sudo systemctl status flux2.service
sudo systemctl status flux2-nginx.service
```

### 4. Verify Deployment

```bash
# Check all services
sudo systemctl list-units --type=service | grep flux2

# Test UI workers
curl http://127.0.0.1:8501/health
curl http://127.0.0.1:8502/health
curl http://127.0.0.1:8503/health

# Test model workers
curl http://127.0.0.1:8600/health
curl http://127.0.0.1:8601/health

# Test load balancer
curl http://127.0.0.1/health

# Test monitoring
curl http://127.0.0.1:9090/-/healthy
curl http://127.0.0.1:3000/api/health
```

### 5. Monitor Logs

```bash
# Supervisord
sudo tail -f /var/log/supervisor/flux2-ui-worker-1.log
sudo tail -f /var/log/supervisor/flux2-model-worker-4b.log

# Nginx
sudo tail -f /var/log/nginx/flux2_error.log
sudo tail -f /var/log/nginx/flux2_access.log

# Systemd
sudo journalctl -u flux2.service -f
```

---

## Windows Deployment (NSSM)

### 1. Install NSSM

```powershell
# Download NSSM from https://nssm.cc/download
# Extract to C:\Program Files\nssm

# Or use Chocolatey
choco install nssm
```

### 2. Run Installation Script

```powershell
# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Run installation (as Administrator)
.\deployment\windows-nssm\install_services.ps1

# Or use batch script
cmd /c .\deployment\windows-nssm\install_services.bat
```

### 3. Verify Services

```powershell
# List deployed services
Get-Service | Where-Object {$_.Name -like 'FLUX2-*'}

# Check service status
net start | findstr FLUX2
nssm status FLUX2-UI-Worker-1

# View service logs
Get-Content "$env:TEMP\FLUX2-UI-Worker-1.log" -Tail 50
```

### 4. Manage Services

```powershell
# Start all services
net start FLUX2-UI-Worker-1
net start FLUX2-UI-Worker-2
net start FLUX2-UI-Worker-3
net start FLUX2-Model-Worker-4B
net start FLUX2-Model-Worker-9B

# Stop service
net stop FLUX2-UI-Worker-1

# Restart service
nssm restart FLUX2-UI-Worker-1

# Remove service
nssm remove FLUX2-UI-Worker-1 confirm
```

### 5. Uninstall

```powershell
# Run uninstall script (as Administrator)
.\deployment\windows-nssm\uninstall_services.ps1
```

---

## Load Balancer Setup

### Nginx Configuration

Nginx acts as reverse proxy with round-robin load balancing and circuit breaker pattern.

**Key Features:**
- ✅ Round-robin distribution across 3 UI workers
- ✅ Circuit breaker: Removes unhealthy backends automatically
- ✅ Caching: Static files cached for 30 days
- ✅ Rate limiting: 30 req/s per IP for UI, 5 req/s for models
- ✅ Timeout handling: 300s timeout for long inference operations
- ✅ Health checks: `GET /health` endpoints monitored

**Check Load Balancer Status:**

```bash
# Test load balancing
for i in {1..30}; do curl http://127.0.0.1 2>&1 | grep -o "8501\|8502\|8503"; done

# View Nginx metrics
curl http://127.0.0.1/metrics -s | head -20

# Check upstream status
sudo nginx -T | grep upstream
```

**Reload Configuration (zero-downtime):**

```bash
sudo nginx -t  # Validate config
sudo systemctl reload nginx
```

---

## Monitoring & Observability

### Prometheus

Scrapes metrics from:
- UI workers (Streamlit)
- Model workers (Custom endpoints)
- Redis cache
- RabbitMQ queue
- Node exporter (system metrics)
- NVIDIA GPU metrics

**Prometheus Dashboard:**
```
http://127.0.0.1:9090
```

**Common Queries:**

```promql
# Generation latency (P95)
histogram_quantile(0.95, rate(flux2_generation_duration_seconds_bucket[5m]))

# Queue length
flux2_queue_length

# GPU memory usage
flux2_gpu_memory_bytes / 1024 / 1024 / 1024

# Error rate
rate(flux2_api_errors_total[5m])

# Cache hit rate
rate(flux2_model_cache_hits_total[1h]) / (rate(flux2_model_cache_hits_total[1h]) + rate(flux2_model_cache_misses_total[1h]))
```

### Grafana

Pre-built dashboard: `flux2-dashboard.json`

**Access:**
```
http://127.0.0.1:3000
Username: admin
Password: admin (change on first login!)
```

**Add Data Source:**
1. Settings → Data Sources
2. Choose Prometheus
3. URL: `http://127.0.0.1:9090`

**Import Dashboard:**
1. Dashboards → New → Import
2. Upload: `config/grafana/flux2-dashboard.json`

---

## Health Checks & Auto-Recovery

### Supervisord (Linux)

Automatically restarts failed processes:

```bash
# Check process status
sudo supervisorctl -c /opt/flux2/deployment/supervisord/flux2.conf status

# Restart failed process
sudo supervisorctl -c /opt/flux2/deployment/supervisord/flux2.conf restart flux2-ui-worker-1

# Restart all
sudo supervisorctl -c /opt/flux2/deployment/supervisord/flux2.conf restart all

# View logs
sudo tail -f /var/log/supervisor/flux2-ui-worker-1.log
```

### Systemd (Linux)

Services auto-restart on failure:

```bash
# Manual restart
sudo systemctl restart flux2-ui-worker-1

# Check restart history
sudo journalctl -u flux2-ui-worker-1.service -n 50

# Increase restart limit if needed
# Edit: /etc/systemd/system/flux2.service
# [Service]
# StartLimitInterval=600
# StartLimitBurst=3
```

### NSSM (Windows)

Services auto-restart with configurable delays:

```powershell
# View restart count
nssm get FLUX2-UI-Worker-1 AppExit

# Set restart delay (milliseconds)
nssm set FLUX2-UI-Worker-1 AppThrottle 1000

# View all settings
nssm dump FLUX2-UI-Worker-1
```

### Health Check Endpoints

Each service exposes health checks:

```bash
# Liveness check (is service running?)
GET /health/live → 200 OK

# Readiness check (is service ready to serve?)
GET /health/ready → 200 OK (all checks pass) | 503 (some checks fail)

# Full status
GET /health → { status, version, uptime_seconds, checks }
```

**Kubernetes-style Probes:**

```yaml
# liveness probe - restart if fails
livenessProbe:
  httpGet:
    path: /health/live
    port: 8501
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

# readiness probe - remove from LB if fails  
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8501
  initialDelaySeconds: 40
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 2
```

---

## Troubleshooting

### Common Issues

#### 1. UI Workers Not Starting

```bash
# Check logs
sudo tail -f /var/log/supervisor/flux2-ui-worker-1-error.log

# Verify streamlit installation
python3 -c "import streamlit; print(streamlit.__version__)"

# Test manually
cd /opt/flux2
streamlit run ui_flux2_professional.py --server.port=8501
```

#### 2. Model Worker Crashes

```bash
# Check CUDA availability
nvidia-smi

# Check GPU memory
nvidia-smi -q -d MEMORY

# Verify model weights
ls -lh /opt/flux2/weights/

# Test manually
python3 scripts/model_worker.py
```

#### 3. High Latency

```bash
# Check load on Nginx
watch curl http://127.0.0.1/metrics

# Monitor queue depth
curl http://127.0.0.1:9090/api/query?query=flux2_queue_length

# Check Prometheus scrape health
curl http://127.0.0.1:9090/api/query_range?query=up
```

#### 4. Redis Connection Failed

```bash
# Test Redis
redis-cli ping

# Check port
netstat -an | grep 6379

# Verify config
redis-cli CONFIG GET "*"

# Clear cache if needed
redis-cli FLUSHALL
```

#### 5. RabbitMQ Queue Backlog

```bash
# Check RabbitMQ status
sudo rabbitmqctl status

# List queues
sudo rabbitmqctl list_queues name messages consumers

# Purge queue (CAUTION!)
sudo rabbitmqctl purge_queue flux2.generation.requests
```

### Performance Tuning

**Increase file descriptors (Linux):**

```bash
# /etc/security/limits.conf
flux2 soft nofile 65535
flux2 hard nofile 65535
```

**Optimize Nginx worker processes:**

```bash
# config/nginx/flux2.conf
worker_processes auto;  # Match CPU cores
worker_connections 10240;  # Increase if needed
```

**Scale Redis memory:**

```bash
# config/redis/redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
```

---

## Next Steps

1. ✅ Run full deployment
2. ✅ Verify all health checks pass
3. ✅ Load test with `ab` or `wrk`
4. ✅ Monitor metrics in Grafana
5. ✅ Set up alerting in Prometheus
6. ✅ Configure backup & disaster recovery
7. ✅ Document runbooks for operations team

---

**Last Updated**: February 2026  
**Tested On**: Ubuntu 22.04 LTS, CentOS 8.5, Windows Server 2019
