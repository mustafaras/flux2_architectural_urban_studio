from __future__ import annotations

from pathlib import Path
from typing import Any


_DEPLOYMENT_PROFILES: dict[str, dict[str, Any]] = {
    "local_pilot": {
        "name": "Local Pilot",
        "health_endpoints": ["/health", "/health/live", "/health/ready"],
        "graceful_degradation": ["fallback to CPU", "disable advanced modules", "reduce queue concurrency"],
        "rollback": ["restore previous .venv lock", "revert app code", "restart streamlit service"],
        "version_pinning": "requirements.txt pinned + pyproject lock",
    },
    "team_server": {
        "name": "Team Server",
        "health_endpoints": ["/health", "/ready"],
        "graceful_degradation": ["saturation fallback profile", "pause workshop lanes", "prioritize P0/P1 lanes"],
        "rollback": ["switch service symlink", "restore last known good image", "replay config snapshot"],
        "version_pinning": "container tag + requirements-dev pins",
    },
    "enterprise_managed": {
        "name": "Enterprise Managed",
        "health_endpoints": ["/health", "/health/live", "/health/ready"],
        "graceful_degradation": ["regional failover", "feature flag rollback", "read-only governance mode"],
        "rollback": ["rollback deployment ring", "restore signed release manifest", "execute audited rollback runbook"],
        "version_pinning": "signed artifact digest + immutable infrastructure templates",
    },
}


def list_profiles() -> list[str]:
    return list(_DEPLOYMENT_PROFILES.keys())


def get_profile(profile_key: str) -> dict[str, Any]:
    return dict(_DEPLOYMENT_PROFILES.get(profile_key, _DEPLOYMENT_PROFILES["local_pilot"]))


def validate_startup_health_prereqs() -> dict[str, bool]:
    return {
        "ui_entrypoint": Path("ui_flux2_professional.py").exists(),
        "health_module": Path("src/flux2/health_check.py").exists(),
        "deploy_script_linux": Path("deploy.sh").exists(),
        "deploy_script_windows": Path("deploy-windows.ps1").exists(),
    }
