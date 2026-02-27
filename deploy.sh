#!/bin/bash
# FLUX.2 Phase 10 Quick Start Deployment Script
# Automated setup for Linux/macOS with systemd

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FLUX2_ROOT="${FLUX2_ROOT:-/opt/flux2}"
FLUX2_USER="${FLUX2_USER:-flux2}"
PYTHON_VERSION="python3.10"

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      FLUX.2 Phase 10 Production Setup          ║${NC}"
echo -e "${BLUE}║     Native Deployment with systemd             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root${NC}"
   echo "Run: sudo ./deploy.sh"
   exit 1
fi

# ============================================================
# 1. SYSTEM DEPENDENCIES
# ============================================================

echo -e "${YELLOW}[1/8] Installing system dependencies...${NC}"

if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    apt-get update
    apt-get install -y \
        build-essential \
        python3.10 python3.10-venv python3-pip \
        git curl wget \
        nginx redis-server rabbitmq-server \
        prometheus grafana-server \
        supervisor \
        nvidia-driver-latest nvidia-utils \
        jq
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    yum install -y \
        gcc gcc-c++ make \
        python310 python310-pip \
        git curl wget \
        nginx redis \
        python3-prometheus-client \
        supervisor \
        python3-jinja2
else
    echo -e "${RED}Unsupported OS. Install dependencies manually.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo

# ============================================================
# 2. CREATE SYSTEM USER
# ============================================================

echo -e "${YELLOW}[2/8] Setting up system user...${NC}"

if ! id -u "$FLUX2_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$FLUX2_USER"
    echo -e "${GREEN}✓ User '$FLUX2_USER' created${NC}"
else
    echo -e "${GREEN}✓ User '$FLUX2_USER' already exists${NC}"
fi

# Add to docker group if Docker is installed
if command -v docker &> /dev/null; then
    usermod -aG docker "$FLUX2_USER"
fi

echo

# ============================================================
# 3. SETUP DIRECTORIES
# ============================================================

echo -e "${YELLOW}[3/8] Creating directory structure...${NC}"

mkdir -p \
    "$FLUX2_ROOT" \
    /var/log/supervisor \
    /var/log/nginx \
    /var/log/redis \
    /var/lib/flux2/redis \
    /var/lib/flux2/prometheus \
    /var/lib/flux2/grafana \
    /var/lib/flux2/rabbitmq \
    /var/run/flux2 \
    /var/cache/nginx/ui \
    /var/cache/nginx/models

# Set permissions
chown -R "$FLUX2_USER:$FLUX2_USER" \
    "$FLUX2_ROOT" \
    /var/log/supervisor \
    /var/lib/flux2 \
    /var/run/flux2

chown -R "$FLUX2_USER:$FLUX2_USER" /var/cache/nginx

echo -e "${GREEN}✓ Directories created${NC}"
echo

# ============================================================
# 4. PYTHON ENVIRONMENT
# ============================================================

echo -e "${YELLOW}[4/8] Setting up Python environment...${NC}"

cd "$FLUX2_ROOT"

if [ ! -d .venv ]; then
    $PYTHON_VERSION -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install supervisor prometheus-client redis pika

# Copy to flux2 user if needed
if [ "$USER" != "$FLUX2_USER" ]; then
    chown -R "$FLUX2_USER:$FLUX2_USER" "$FLUX2_ROOT/.venv"
fi

echo -e "${GREEN}✓ Python environment ready${NC}"
echo

# ============================================================
# 5. SYSTEMD SERVICES
# ============================================================

echo -e "${YELLOW}[5/8] Installing systemd services...${NC}"

cp "$FLUX2_ROOT/deployment/systemd"/*.service /etc/systemd/system/

# Update paths in service files
sed -i "s|/opt/flux2|$FLUX2_ROOT|g" /etc/systemd/system/flux2*.service
sed -i "s|User=flux2|User=$FLUX2_USER|g" /etc/systemd/system/flux2*.service

systemctl daemon-reload
systemctl enable flux2.service
systemctl enable flux2-nginx.service
systemctl enable flux2-prometheus.service
systemctl enable flux2-grafana.service
systemctl enable redis-server.service
systemctl enable rabbitmq-server.service

echo -e "${GREEN}✓ Systemd services installed${NC}"
echo

# ============================================================
# 6. CONFIGURATION FILES
# ============================================================

echo -e "${YELLOW}[6/8] Configuring services...${NC}"

# Nginx
mkdir -p /etc/nginx/sites-enabled /var/cache/nginx/ui /var/cache/nginx/models
cp "$FLUX2_ROOT/config/nginx/flux2.conf" /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-enabled/flux2.conf /etc/nginx/sites-enabled/default || true

# Redis
cp "$FLUX2_ROOT/config/redis/redis.conf" /etc/redis/
chown redis:redis /etc/redis/redis.conf

# RabbitMQ
cp "$FLUX2_ROOT/config/rabbitmq/definitions.json" /etc/rabbitmq/
chown rabbitmq:rabbitmq /etc/rabbitmq/definitions.json

# Prometheus
mkdir -p /etc/prometheus
cp "$FLUX2_ROOT/config/prometheus"/*.yml /etc/prometheus/
chown -R prometheus:prometheus /etc/prometheus /var/lib/flux2/prometheus

# Grafana
mkdir -p /etc/grafana/provisioning/dashboards
cp "$FLUX2_ROOT/config/grafana"/*.ini /etc/grafana/
cp "$FLUX2_ROOT/config/grafana"/*.json /etc/grafana/provisioning/dashboards/

# Supervisord
mkdir -p /etc/supervisor/conf.d
cp "$FLUX2_ROOT/deployment/supervisord/flux2.conf" /etc/supervisor/conf.d/

echo -e "${GREEN}✓ Services configured${NC}"
echo

# ============================================================
# 7. START SERVICES
# ============================================================

echo -e "${YELLOW}[7/8] Starting services...${NC}"

# Start infrastructure first
systemctl start redis-server
systemctl start rabbitmq-server
sleep 5

# Start monitoring
systemctl start prometheus.service
systemctl start grafana-server.service
sleep 5

# Start load balancer
systemctl start nginx.service

# Start FLUX.2 services (Supervisord manages workers)
systemctl start flux2.service
sleep 10

echo -e "${GREEN}✓ Services started${NC}"
echo

# ============================================================
# 8. VERIFICATION
# ============================================================

echo -e "${YELLOW}[8/8] Verifying deployment...${NC}"
echo

CHECKS_PASSED=0
CHECKS_TOTAL=9

# Check UI workers
echo -n "  UI Worker 1 (8501): "
if curl -s http://127.0.0.1:8501/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

echo -n "  UI Worker 2 (8502): "
if curl -s http://127.0.0.1:8502/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

echo -n "  UI Worker 3 (8503): "
if curl -s http://127.0.0.1:8503/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

# Check model workers
echo -n "  Model Worker 4B (8600): "
if curl -s http://127.0.0.1:8600/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

echo -n "  Model Worker 9B (8601): "
if curl -s http://127.0.0.1:8601/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

# Check load balancer
echo -n "  Load Balancer (Nginx): "
if curl -s http://127.0.0.1/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

# Check Redis
echo -n "  Redis Cache: "
if redis-cli ping >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

# Check Prometheus
echo -n "  Prometheus: "
if curl -s http://127.0.0.1:9090/-/healthy >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

# Check Grafana
echo -n "  Grafana: "
if curl -s http://127.0.0.1:3000/api/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
    ((CHECKS_PASSED++))
else
    echo -e "${RED}✗${NC}"
fi

echo

# ============================================================
# FINAL RESULTS
# ============================================================

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Deployment Successful!                     ║${NC}"
    echo -e "${GREEN}║  All $CHECKS_TOTAL checks passed              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠ Deployment Partial                         ║${NC}"
    echo -e "${YELLOW}║  $CHECKS_PASSED/$CHECKS_TOTAL checks passed${NC}"
    echo -e "${YELLOW}║  See logs: systemctl status flux2.service     ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
fi

echo
echo -e "${BLUE}📊 Service URLs:${NC}"
echo "  • UI: http://127.0.0.1 (port 80, load balanced)"
echo "  • Grafana: http://127.0.0.1:3000 (user: admin, pass: admin)"
echo "  • Prometheus: http://127.0.0.1:9090"
echo "  • RabbitMQ: http://127.0.0.1:15672 (user: guest, pass: guest)"
echo
echo -e "${BLUE}📝 Useful Commands:${NC}"
echo "  • Status: systemctl status flux2.service"
echo "  • Logs: sudo tail -f /var/log/supervisor/flux2-ui-worker-1.log"
echo "  • Workers: sudo supervisorctl -c $FLUX2_ROOT/deployment/supervisord/flux2.conf status"
echo "  • Stop: sudo systemctl stop flux2.service"
echo "  • Restart: sudo systemctl restart flux2.service"
echo
echo -e "${BLUE}📖 Documentation:${NC}"
echo "  • Read: $FLUX2_ROOT/docs/PHASE10_DEPLOYMENT.md"
echo
