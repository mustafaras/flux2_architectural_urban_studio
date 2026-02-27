# FLUX.2 Phase 10 Quick Start Deployment Script (Windows PowerShell)
# Automated setup for Windows with NSSM service manager

param(
    [string]$FluxRoot = "C:\flux2",
    [string]$PythonPath = "C:\Python310",
    [switch]$SkipDependencies
)

# ============================================================
# CONFIGURATION & COLORS
# ============================================================

$ErrorActionPreference = "Stop"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Error_ { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Section { Write-Host "`n╔════════════════════════════════════════════════╗" -ForegroundColor Blue; Write-Host "║ $args" -ForegroundColor Blue; Write-Host "╚════════════════════════════════════════════════╝`n" -ForegroundColor Blue }

# Admin check
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error_ "This script must be run as Administrator!"
    Write-Error_ "Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

Write-Section "FLUX.2 Phase 10 Production Setup - Windows"
Write-Section "Native Deployment with NSSM"

# ============================================================
# 1. SYSTEM DEPENDENCIES (Optional)
# ============================================================

if (-not $SkipDependencies) {
    Write-Section "[1/7] Checking system dependencies..."
    
    # Check Python
    if (-not (Test-Path $PythonPath)) {
        Write-Error_ "Python not found at $PythonPath"
        Write-Warn "Download Python 3.10+ from https://www.python.org/downloads/"
        Write-Warn "Install with 'Add Python to PATH' option enabled"
        exit 1
    } else {
        Write-Info "Python found at $PythonPath"
    }
    
    # Check NSSM
    $NSSmPath = "C:\Program Files\nssm\win64\nssm.exe"
    if (-not (Test-Path $NSSmPath)) {
        Write-Error_ "NSSM not found at $NSSmPath"
        Write-Warn "Download NSSM from: https://nssm.cc/download"
        Write-Warn "Extract to: C:\Program Files\nssm\"
        exit 1
    } else {
        Write-Info "NSSM found at $NSSmPath"
    }
    
    # Check Redis
    $RedisPath = Get-Command redis-server -ErrorAction SilentlyContinue
    if (-not $RedisPath) {
        Write-Warn "Redis not found in PATH"
        Write-Info "Download from: https://github.com/microsoftarchive/redis/releases"
        Write-Info "Or use: choco install redis"
    } else {
        Write-Info "Redis found: $($RedisPath.Source)"
    }
}

Write-Info "Dependencies check complete`n"

# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

Write-Section "[2/7] Creating directory structure..."

$dirs = @(
    $FluxRoot,
    "$FluxRoot\logs",
    "$FluxRoot\logs\ui",
    "$FluxRoot\logs\models",
    "$FluxRoot\config",
    "$FluxRoot\cache",
    "$FluxRoot\.venv"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Info "Created: $dir"
    } else {
        Write-Info "Exists: $dir"
    }
}

Write-Info "Directory structure ready`n"

# ============================================================
# 3. PYTHON ENVIRONMENT
# ============================================================

Write-Section "[3/7] Setting up Python environment..."

$venvPath = "$FluxRoot\.venv"
$pythonExe = "$venvPath\Scripts\python.exe"
$pipExe = "$venvPath\Scripts\pip.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Info "Creating virtual environment..."
    & "$PythonPath\python.exe" -m venv $venvPath
    Write-Info "Virtual environment created"
} else {
    Write-Info "Virtual environment already exists"
}

# Upgrade pip
Write-Info "Upgrading pip, setuptools, wheel..."
& $pipExe install --upgrade pip setuptools wheel

# Install dependencies
Write-Info "Installing Flask, Pytorch, Streamlit..."
& $pipExe install -r "$FluxRoot\requirements.txt" 2>&1 | Select-Object -Last 5
& $pipExe install redis pika prometheus-client 2>&1 | Select-Object -Last 3

Write-Info "Python environment ready`n"

# ============================================================
# 4. NSSM SERVICES
# ============================================================

Write-Section "[4/7] Installing NSSM services..."

$NSSmPath = "C:\Program Files\nssm\win64\nssm.exe"
$WorkingDir = "$FluxRoot"
$StreamlitExe = "$pythonExe"
$UIScript = "$FluxRoot\ui_flux2_professional.py"
$ModelWorkerScript = "$FluxRoot\scripts\model_worker.py"

function Install-NSSmService {
    param(
        [string]$ServiceName,
        [string]$ProgName,
        [string]$Port,
        [string]$GPU_ID,
        [string]$ScriptPath
    )
    
    # Stop if exists
    $existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Info "Stopping existing service: $ServiceName"
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        
        & $NSSmPath remove $ServiceName confirm | Out-Null
        Write-Info "Removed old service definition"
    }
    
    # Install new service
    Write-Info "Installing: $ServiceName"
    
    $args = @(
        "install",
        $ServiceName,
        "$pythonExe",
        "`"$ScriptPath`""
    )
    
    & $NSSmPath $args | Out-Null
    
    # Configure service
    & $NSSmPath set $ServiceName AppDirectory "$WorkingDir" | Out-Null
    & $NSSmPath set $ServiceName AppRotateFiles 1 | Out-Null
    & $NSSmPath set $ServiceName AppRotateSeconds 3600 | Out-Null
    
    # Environment variables
    if ($Port) {
        & $NSSmPath set $ServiceName AppEnvironmentExtra "STREAMLIT_SERVER_PORT=$Port" | Out-Null
    }
    if ($GPU_ID) {
        & $NSSmPath set $ServiceName AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=$GPU_ID" | Out-Null
    }
    
    # Set startup type
    Set-Service -Name $ServiceName -StartupType Automatic
    
    Write-Info "  ✓ Service configured: $ServiceName (Port: $Port, GPU: $GPU_ID)"
}

# UI Workers
Install-NSSmService "FLUX2-UI-Worker-1" "Streamlit" "8501" "0" $UIScript
Install-NSSmService "FLUX2-UI-Worker-2" "Streamlit" "8502" "0" $UIScript
Install-NSSmService "FLUX2-UI-Worker-3" "Streamlit" "8503" "0" $UIScript

# Model Workers
Install-NSSmService "FLUX2-Model-Worker-4B" "Model Server" "8600" "1" $ModelWorkerScript
Install-NSSmService "FLUX2-Model-Worker-9B" "Model Server" "8601" "2" $ModelWorkerScript

Write-Info "All NSSM services installed`n"

# ============================================================
# 5. AUXILIARY SERVICES
# ============================================================

Write-Section "[5/7] Starting auxiliary services..."

# Check and start Redis
Write-Info "Redis: $(if ((Get-Service redis-server -ErrorAction SilentlyContinue).Status -eq 'Running') { 'Running' } else { 'Starting...' })"
Start-Service -Name redis-server -ErrorAction SilentlyContinue

# Check and start RabbitMQ (if available)
if (Get-Service RabbitMQ -ErrorAction SilentlyContinue) {
    Write-Info "RabbitMQ: Starting..."
    Start-Service -Name RabbitMQ -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
    Write-Info "  ✓ RabbitMQ started"
}

# Start Prometheus (if available)
if (Get-Service Prometheus -ErrorAction SilentlyContinue) {
    Write-Info "Prometheus: Starting..."
    Start-Service -Name Prometheus -ErrorAction SilentlyContinue
}

# Start Grafana (if available)
if (Get-Service Grafana -ErrorAction SilentlyContinue) {
    Write-Info "Grafana: Starting..."
    Start-Service -Name Grafana -ErrorAction SilentlyContinue
}

Write-Info "Auxiliary services configured`n"

# ============================================================
# 6. START FLUX2 SERVICES
# ============================================================

Write-Section "[6/7] Starting FLUX.2 services..."

$services = @(
    "FLUX2-UI-Worker-1",
    "FLUX2-UI-Worker-2",
    "FLUX2-UI-Worker-3",
    "FLUX2-Model-Worker-4B",
    "FLUX2-Model-Worker-9B"
)

foreach ($svc in $services) {
    Write-Info "Starting: $svc"
    Start-Service -Name $svc -ErrorAction SilentlyContinue
}

# Wait for startup
Write-Info "Waiting 15 seconds for services to start..."
Start-Sleep -Seconds 15

# ============================================================
# 7. HEALTH CHECKS
# ============================================================

Write-Section "[7/7] Verifying deployment..."

$checksPassed = 0
$checksTotal = 5

function Check-Service {
    param([string]$Name, [int]$Port)
    
    $status = Get-Service -Name $Name -ErrorAction SilentlyContinue
    
    if ($status -and $status.Status -eq 'Running') {
        Write-Host "  ✓ $Name" -ForegroundColor Green
        return $true
    } elseif ($status) {
        Write-Host "  ✗ $Name (Status: $($status.Status))" -ForegroundColor Red
        return $false
    } else {
        Write-Host "  ✗ $Name (Not found)" -ForegroundColor Red
        return $false
    }
}

Write-Host "`nService Status:"
if (Check-Service "FLUX2-UI-Worker-1" 8501) { $checksPassed++ }
if (Check-Service "FLUX2-UI-Worker-2" 8502) { $checksPassed++ }
if (Check-Service "FLUX2-UI-Worker-3" 8503) { $checksPassed++ }
if (Check-Service "FLUX2-Model-Worker-4B" 8600) { $checksPassed++ }
if (Check-Service "FLUX2-Model-Worker-9B" 8601) { $checksPassed++ }

# ============================================================
# FINAL RESULTS
# ============================================================

Write-Host ""

if ($checksPassed -eq 5) {
    Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║  ✓ Deployment Successful!                     ║" -ForegroundColor Green
    Write-Host "║  All 5 services running                       ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Green
} else {
    Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║  ⚠ Deployment Partial                         ║" -ForegroundColor Yellow
    Write-Host "║  $checksPassed/5 services running${' ' * (25 - $checksPassed.ToString().Length)} ║" -ForegroundColor Yellow
    Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Yellow
}

Write-Host "`n📊 Service URLs:" -ForegroundColor Blue
Write-Host "  • Web UI: http://localhost:8501 (UI Worker 1)"
Write-Host "  • Model API: http://localhost:8600 (Model Worker 4B)"
Write-Host ""

Write-Host "📝 Useful Commands:" -ForegroundColor Blue
Write-Host "  • View services: Get-Service FLUX2-*"
Write-Host "  • Restart all: Get-Service FLUX2-* | Restart-Service"
Write-Host "  • View logs: Get-Content '$FluxRoot\logs\ui\*.log'"
Write-Host "  • Remove all: Get-Service FLUX2-* | Stop-Service; nssm remove FLUX2-UI-Worker-1 confirm"
Write-Host ""

Write-Host "📖 Documentation:" -ForegroundColor Blue
Write-Host "  • Read: $FluxRoot\docs\PHASE10_DEPLOYMENT.md"
Write-Host "  • Troubleshooting: $FluxRoot\docs\PHASE10_DEPLOYMENT.md#troubleshooting"
Write-Host ""

Write-Host "✅ Setup complete! Check service status with: Get-Service FLUX2-*" -ForegroundColor Green
