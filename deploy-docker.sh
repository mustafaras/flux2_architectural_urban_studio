#!/bin/bash
# FLUX.2 Phase 10 Quick Start - Docker Compose
# Fast deployment with docker-compose up

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      FLUX.2 Phase 10 - Docker Deployment      ║${NC}"
echo -e "${BLUE}║     Fast Setup with docker-compose            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo

# Configuration
FLUX2_ROOT="${FLUX2_ROOT:-.}"
COMPOSE_FILE="$FLUX2_ROOT/docker-compose.yml"

# ============================================================
# 1. PREREQUISITES CHECK
# ============================================================

echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    echo "Install: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker: $(docker --version)${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose not installed${NC}"
    echo "Install: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose: $(docker-compose --version)${NC}"

# Check NVIDIA Docker (optional but recommended)
if command -v nvidia-docker &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Docker: Available${NC}"
    echo -e "${YELLOW}  → GPU support enabled${NC}"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo -e "${YELLOW}⚠ NVIDIA Docker not installed${NC}"
    echo "  → GPU support via env variable: $CUDA_VISIBLE_DEVICES"
else
    echo -e "${YELLOW}⚠ NVIDIA Docker not installed${NC}"
    echo "  → GPU acceleration disabled"
    echo "  → Install: https://github.com/NVIDIA/nvidia-docker"
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo

# ============================================================
# 2. CONFIGURATION VALIDATION
# ============================================================

echo -e "${YELLOW}[2/4] Validating configuration files...${NC}"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Error: docker-compose.yml not found at $FLUX2_ROOT${NC}"
    exit 1
fi

docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ docker-compose.yml is valid${NC}"
else
    echo -e "${RED}Error: Invalid docker-compose.yml${NC}"
    exit 1
fi

echo

# ============================================================
# 3. PRE-DEPLOYMENT SETUP
# ============================================================

echo -e "${YELLOW}[3/4] Setting up environment...${NC}"

# Create required directories
mkdir -p \
    "$FLUX2_ROOT/outputs" \
    "$FLUX2_ROOT/weights" \
    "$FLUX2_ROOT/projects" \
    "$FLUX2_ROOT/logs/analytics" \
    "$FLUX2_ROOT/logs/crashes" \
    "$FLUX2_ROOT/logs/performance"

echo -e "${GREEN}✓ Directories prepared${NC}"

# Check weights
if [ ! -f "$FLUX2_ROOT/weights/model_registry.json" ]; then
    echo -e "${YELLOW}⚠ Warning: Model weights not found${NC}"
    echo "  → Add weight files to: $FLUX2_ROOT/weights/"
    echo "  → Example: flux-2-klein-4b.safetensors"
else
    echo -e "${GREEN}✓ Model weights found${NC}"
fi

echo

# ============================================================
# 4. DEPLOYMENT
# ============================================================

echo -e "${YELLOW}[4/4] Starting services...${NC}"
echo

cd "$FLUX2_ROOT"

# Build images (if needed)
echo "Building Docker images..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

# Start services
echo -e "\n${BLUE}Starting FLUX.2 services...${NC}\n"
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services
echo -e "\n${BLUE}Waiting for services to become ready...${NC}\n"
sleep 15

# Health check
echo -e "${YELLOW}Checking service health...${NC}\n"

HEALTHY_SERVICES=0
TOTAL_SERVICES=10

check_service() {
    local service=$1
    local port=$2
    local endpoint=${3:-"/health"}
    
    echo -n "  $service: "
    
    if docker-compose -f "$COMPOSE_FILE" ps $service | grep -q "Up"; then
        echo -e "${GREEN}Running${NC}"
        ((HEALTHY_SERVICES++))
        return 0
    else
        echo -e "${RED}Down${NC}"
        return 1
    fi
}

# Check all services
check_service "flux2-ui-worker-1" "8501"
check_service "flux2-ui-worker-2" "8502"
check_service "flux2-ui-worker-3" "8503"
check_service "flux2-model-worker-4b" "8600"
check_service "flux2-model-worker-9b" "8601"
check_service "flux2-nginx" "80"
check_service "flux2-redis" "6379"
check_service "flux2-rabbitmq" "5672"
check_service "flux2-prometheus" "9090"
check_service "flux2-grafana" "3000"

echo

# ============================================================
# DEPLOYMENT RESULTS
# ============================================================

if [ $HEALTHY_SERVICES -eq $TOTAL_SERVICES ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Deployment Successful!                     ║${NC}"
    echo -e "${GREEN}║  All $TOTAL_SERVICES services running           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠ Deployment Partial                         ║${NC}"
    echo -e "${YELLOW}║  $HEALTHY_SERVICES/$TOTAL_SERVICES services running${NC}"
    echo -e "${YELLOW}║  Check logs: docker-compose logs -f            ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
fi

echo
echo -e "${BLUE}📊 Service URLs:${NC}"
echo "  • Web UI: http://localhost:80"
echo "  • UI Worker 1: http://localhost:8501"
echo "  • Model Worker 4B: http://localhost:8600"
echo "  • Model Worker 9B: http://localhost:8601"
echo "  • Grafana: http://localhost:3000 (user: admin, pass: admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • RabbitMQ: http://localhost:15672 (user: guest, pass: guest)"
echo

echo -e "${BLUE}📝 Useful Commands:${NC}"
echo "  • View logs: docker-compose logs -f [service]"
echo "  • Status: docker-compose ps"
echo "  • Restart: docker-compose restart [service]"
echo "  • Stop: docker-compose down"
echo "  • Stop + Remove volumes: docker-compose down -v"
echo "  • View container: docker exec -it [container] bash"
echo

echo -e "${BLUE}📈 Monitoring:${NC}"
echo "  • Logs: docker-compose logs -f"
echo "  • Performance: docker stats"
echo "  • Events: docker events --filter type=container"
echo

echo -e "${YELLOW}⚡ First time setup:${NC}"
echo "  1. Login to Grafana: http://localhost:3000"
echo "  2. Default credentials: admin / admin"
echo "  3. Change password immediately!"
echo "  4. Import dashboard from provisioning folder"
echo "  5. Configure alerts in Grafana"
echo

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "   Run: docker-compose logs -f"
echo "   to monitor real-time service activity"
echo
