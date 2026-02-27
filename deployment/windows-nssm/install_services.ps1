# FLUX.2 Windows Deployment Script (PowerShell)
# Requires: Administrator privileges
# Usage: Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .\install_services.ps1

param(
    [string]$FluxRoot = "C:\ai\flux2",
    [string]$NssmPath = "C:\Program Files\nssm\win64\nssm.exe"
)

# Check if running as Administrator
$CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($CurrentUser)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

# Validate paths
if (-not (Test-Path $FluxRoot)) {
    Write-Error "FLUX2 root directory not found: $FluxRoot"
    exit 1
}

if (-not (Test-Path $NssmPath)) {
    Write-Error "NSSM not found at: $NssmPath"
    Write-Host "Please download NSSM from: https://nssm.cc/download"
    exit 1
}

$VenvPath = Join-Path $FluxRoot ".venv"

Write-Host "============================================================"
Write-Host "FLUX.2 Windows Service Deployment (PowerShell)"
Write-Host "============================================================"

# Define services configuration
$Services = @(
    @{
        Name = "FLUX2-UI-Worker-1"
        Program = Join-Path $VenvPath "Scripts\streamlit.exe"
        Args = "run ui_flux2_professional.py --server.port=8501 --server.address=127.0.0.1"
        GPU = "0"
    },
    @{
        Name = "FLUX2-UI-Worker-2"
        Program = Join-Path $VenvPath "Scripts\streamlit.exe"
        Args = "run ui_flux2_professional.py --server.port=8502 --server.address=127.0.0.1"
        GPU = "0"
    },
    @{
        Name = "FLUX2-UI-Worker-3"
        Program = Join-Path $VenvPath "Scripts\streamlit.exe"
        Args = "run ui_flux2_professional.py --server.port=8503 --server.address=127.0.0.1"
        GPU = "0"
    },
    @{
        Name = "FLUX2-Model-Worker-4B"
        Program = Join-Path $VenvPath "Scripts\python.exe"
        Args = "scripts\model_worker.py"
        GPU = "1"
        Env = @{
            MODEL_WORKER_HOST = "127.0.0.1"
            MODEL_WORKER_PORT = "8600"
            MODEL_SERVICE_NAME = "klein-4b-service"
            MODEL_KEY = "flux.2-klein-4b"
        }
    },
    @{
        Name = "FLUX2-Model-Worker-9B"
        Program = Join-Path $VenvPath "Scripts\python.exe"
        Args = "scripts\model_worker.py"
        GPU = "2"
        Env = @{
            MODEL_WORKER_HOST = "127.0.0.1"
            MODEL_WORKER_PORT = "8601"
            MODEL_SERVICE_NAME = "klein-9b-service"
            MODEL_KEY = "flux.2-klein-9b"
        }
    }
)

$Counter = 1
foreach ($Service in $Services) {
    Write-Host ""
    Write-Host "[$Counter/$($Services.Count)] Installing $($Service.Name)..."
    
    # Check if service exists and remove it
    $ExistingService = Get-Service -Name $Service.Name -ErrorAction SilentlyContinue
    if ($ExistingService) {
        Write-Host "  Removing existing service..."
        & $NssmPath remove $Service.Name confirm
        Start-Sleep -Seconds 2
    }
    
    # Install service
    & $NssmPath install $Service.Name $Service.Program
    & $NssmPath set $Service.Name AppDirectory $FluxRoot
    & $NssmPath set $Service.Name AppParameters $Service.Args
    & $NssmPath set $Service.Name AppEnvironmentExtra "CUDA_VISIBLE_DEVICES=$($Service.GPU)"
    & $NssmPath set $Service.Name AppNoConsole 0
    & $NssmPath set $Service.Name AppPriority HIGH
    & $NssmPath set $Service.Name AppExit Default Restart
    
    # Add custom environment variables if specified
    if ($Service.PSObject.Properties['Env']) {
        foreach ($Key in $Service.Env.Keys) {
            & $NssmPath set $Service.Name AppEnvironmentExtra "$Key=$($Service.Env[$Key])"
        }
    }
    
    # Start service
    & $NssmPath start $Service.Name
    Write-Host "  ✓ Installed and started: $($Service.Name)"
    
    $Counter++
}

Write-Host ""
Write-Host "============================================================"
Write-Host "✓ All services installed successfully!"
Write-Host "============================================================"
Write-Host ""
Write-Host "Installed Services:"
Write-Host "  • FLUX2-UI-Worker-1    (port 8501)"
Write-Host "  • FLUX2-UI-Worker-2    (port 8502)"
Write-Host "  • FLUX2-UI-Worker-3    (port 8503)"
Write-Host "  • FLUX2-Model-Worker-4B  (port 8600)"
Write-Host "  • FLUX2-Model-Worker-9B  (port 8601)"
Write-Host ""
Write-Host "Management Commands:"
Write-Host "  Get-Service | Where-Object {$_.Name -like 'FLUX2-*'}"
Write-Host "  net start FLUX2-UI-Worker-1"
Write-Host "  net stop FLUX2-UI-Worker-1"
Write-Host "  & '$NssmPath' status FLUX2-UI-Worker-1"
Write-Host ""
