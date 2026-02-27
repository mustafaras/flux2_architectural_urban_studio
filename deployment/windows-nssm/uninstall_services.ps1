# FLUX.2 Windows Service Uninstall Script (PowerShell)
# Usage: Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process; .\uninstall_services.ps1

param(
    [string]$NssmPath = "C:\Program Files\nssm\win64\nssm.exe"
)

# Check if running as Administrator
$CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($CurrentUser)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

if (-not (Test-Path $NssmPath)) {
    Write-Error "NSSM not found at: $NssmPath"
    exit 1
}

$ServiceNames = @(
    "FLUX2-UI-Worker-1",
    "FLUX2-UI-Worker-2",
    "FLUX2-UI-Worker-3",
    "FLUX2-Model-Worker-4B",
    "FLUX2-Model-Worker-9B"
)

Write-Host "============================================================"
Write-Host "FLUX.2 Windows Service Uninstall"
Write-Host "============================================================"
Write-Host ""

$Confirm = Read-Host "Do you want to remove all FLUX.2 services? (y/n)"
if ($Confirm -ne 'y') {
    Write-Host "Cancelled."
    exit 0
}

Write-Host ""
foreach ($ServiceName in $ServiceNames) {
    Write-Host "Removing $ServiceName..."
    
    # Stop service
    $Service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($Service) {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        & $NssmPath remove $ServiceName confirm
        Write-Host "  ✓ Removed: $ServiceName"
    } else {
        Write-Host "  - Not found: $ServiceName"
    }
}

Write-Host ""
Write-Host "✓ Uninstall complete"
Write-Host ""
