"""
GPU monitoring utilities for real-time performance tracking.

Provides zero-latency GPU metrics: VRAM usage, temperature, power consumption,
and utilization percentage using pynvml.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

try:
    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetPowerState,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
        NVML_TEMPERATURE_GPU,
        NVMLError,
    )

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger("flux2_gpu_monitor")


@dataclass(slots=True)
class GPUMetrics:
    """GPU metrics snapshot."""

    device_id: int
    device_name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    gpu_util_percent: float
    memory_util_percent: float
    temperature_c: float | None
    power_w: float | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    """
    Real-time GPU monitoring using pynvml (nvidia-ml-py).

    Provides zero-latency metrics:
    - VRAM utilization (MB, %)
    - GPU utilization (%)
    - Temperature (°C)
    - Power consumption (W)
    """

    def __init__(self, device_id: int = 0) -> None:
        """
        Initialize GPU monitor.

        Args:
            device_id: CUDA device index to monitor (usually 0)
        """
        self.device_id = device_id
        self._initialized = False
        self._device_handle = None
        self._device_name = "unknown"

        if PYNVML_AVAILABLE:
            self._init_pynvml()

    def _init_pynvml(self) -> None:
        """Initialize pynvml library."""
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()

            if self.device_id >= device_count:
                logger.warning(
                    f"Device {self.device_id} not available; found {device_count} devices"
                )
                self.device_id = 0

            self._device_handle = nvmlDeviceGetHandleByIndex(self.device_id)
            self._device_name = _get_device_name(self._device_handle)
            self._initialized = True
            logger.info(f"GPU monitor initialized for device {self.device_id}: {self._device_name}")
        except Exception as e:
            logger.warning(f"pynvml initialization failed: {e}")
            self._initialized = False

    def snapshot(self) -> GPUMetrics | None:
        """
        Get current GPU metrics snapshot.

        Returns:
            GPUMetrics if available, else None
        """
        import time

        if not self._initialized or not PYNVML_AVAILABLE:
            return None

        try:
            # Memory metrics
            mem_info = nvmlDeviceGetMemoryInfo(self._device_handle)
            vram_used_mb = float(mem_info.used) / (1024 ** 2)
            vram_total_mb = float(mem_info.total) / (1024 ** 2)
            vram_percent = 100.0 * vram_used_mb / vram_total_mb if vram_total_mb > 0 else 0.0

            # Utilization metrics
            try:
                util_rates = nvmlDeviceGetUtilizationRates(self._device_handle)
                gpu_util_percent = float(util_rates.gpu)
                memory_util_percent = float(util_rates.memory)
            except NVMLError:
                gpu_util_percent = 0.0
                memory_util_percent = vram_percent

            # Temperature metrics
            temp_c = None
            try:
                temp_c = float(nvmlDeviceGetTemperature(self._device_handle, NVML_TEMPERATURE_GPU))
            except NVMLError:
                pass

            # Power metrics
            power_w = None
            try:
                power_mw = nvmlDeviceGetPowerState(self._device_handle)
                power_w = float(power_mw) / 1000.0
            except (NVMLError, TypeError):
                # Try alternative method
                power_w = _get_power_consumption_fallback()

            return GPUMetrics(
                device_id=self.device_id,
                device_name=self._device_name,
                vram_used_mb=vram_used_mb,
                vram_total_mb=vram_total_mb,
                vram_percent=vram_percent,
                gpu_util_percent=gpu_util_percent,
                memory_util_percent=memory_util_percent,
                temperature_c=temp_c,
                power_w=power_w,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.warning(f"GPU metrics snapshot failed: {e}")
            return None

    def shutdown(self) -> None:
        """Shutdown GPU monitor."""
        if self._initialized and PYNVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


def _get_device_name(device_handle: Any) -> str:
    """Get device name from handle."""
    try:
        from pynvml import nvmlDeviceGetName

        name = nvmlDeviceGetName(device_handle)
        if isinstance(name, bytes):
            return name.decode("utf-8")
        return str(name)
    except Exception:
        return "unknown"


def _get_power_consumption_fallback() -> float | None:
    """
    Fallback: get power consumption via nvidia-smi.

    Slower but works when pynvml method fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def get_gpu_monitor() -> GPUMonitor:
    """Get singleton GPU monitor instance."""
    if not hasattr(get_gpu_monitor, "_instance"):
        get_gpu_monitor._instance = GPUMonitor(device_id=0)
    return get_gpu_monitor._instance
