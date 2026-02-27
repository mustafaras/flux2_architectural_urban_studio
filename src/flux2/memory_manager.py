"""GPU memory management and model offloading helpers for FLUX.2."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger("flux2_memory")


@dataclass(slots=True)
class MemorySnapshot:
    cuda_available: bool
    total_mb: float
    used_mb: float
    free_mb: float
    allocated_mb: float
    reserved_mb: float

    @property
    def used_ratio(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb / self.total_mb


class MemoryManager:
    """Encapsulates VRAM observation, pooling and offload policy."""

    def __init__(
        self,
        target_peak_ratio: float = 0.80,
        reserve_pool_ratio: float = 0.05,
    ) -> None:
        self.target_peak_ratio = float(target_peak_ratio)
        self.reserve_pool_ratio = float(reserve_pool_ratio)
        self._pool: torch.Tensor | None = None

    def snapshot(self) -> MemorySnapshot:
        if not torch.cuda.is_available():
            return MemorySnapshot(
                cuda_available=False,
                total_mb=0,
                used_mb=0,
                free_mb=0,
                allocated_mb=0,
                reserved_mb=0,
            )

        total_mb = 0.0
        free_mb = 0.0
        used_mb = 0.0
        allocated_mb = 0.0
        reserved_mb = 0.0

        try:
            allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        except Exception:  # noqa: BLE001
            allocated_mb = 0.0

        try:
            reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
        except Exception:  # noqa: BLE001
            reserved_mb = 0.0

        try:
            free, total = torch.cuda.mem_get_info(0)
            total_mb = total / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            used_mb = total_mb - free_mb
        except Exception as exc:  # noqa: BLE001
            logger.debug("mem_get_info unavailable; falling back to allocated/reserved metrics: %s", exc)
            try:
                props = torch.cuda.get_device_properties(0)
                total_mb = float(props.total_memory) / (1024 * 1024)
                used_mb = max(allocated_mb, reserved_mb)
                free_mb = max(0.0, total_mb - used_mb)
            except Exception:  # noqa: BLE001
                total_mb = 0.0
                free_mb = 0.0
                used_mb = max(allocated_mb, reserved_mb)

        return MemorySnapshot(
            cuda_available=True,
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def reserve_pool(self) -> None:
        """Pre-allocate a small VRAM chunk to reduce allocator fragmentation."""
        if not torch.cuda.is_available():
            return
        if self._pool is not None:
            return
        snap = self.snapshot()
        if snap.total_mb <= 0:
            return
        reserve_mb = max(16.0, snap.total_mb * self.reserve_pool_ratio)
        reserve_elems = int((reserve_mb * 1024 * 1024) // 2)
        try:
            self._pool = torch.empty(reserve_elems, dtype=torch.float16, device="cuda")
        except Exception as exc:  # noqa: BLE001
            logger.debug("VRAM reserve pool allocation skipped: %s", exc)
            self._pool = None

    def release_pool(self) -> None:
        self._pool = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def clear_unused(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def should_offload(self, required_mb: float = 0.0) -> bool:
        snap = self.snapshot()
        if not snap.cuda_available:
            return False
        projected_ratio = (snap.used_mb + required_mb) / max(snap.total_mb, 1.0)
        return projected_ratio >= self.target_peak_ratio

    def maybe_offload_models(self, *models: Any) -> bool:
        """Move provided models to CPU if VRAM pressure exceeds threshold."""
        if not self.should_offload():
            return False

        did_offload = False
        for model in models:
            if model is None:
                continue
            try:
                if hasattr(model, "cpu"):
                    model.cpu()
                    did_offload = True
            except Exception:
                continue

        if did_offload and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        return did_offload

    def maybe_reload_to_gpu(self, device: torch.device, *models: Any) -> None:
        if not torch.cuda.is_available():
            return
        for model in models:
            if model is None:
                continue
            try:
                if hasattr(model, "to"):
                    model.to(device)
            except Exception:
                continue

    def apply_quantization(self, model: Any, mode: str = "none") -> Any:
        """Best-effort quantization support for int8/int4 configurations."""
        normalized = (mode or "none").lower()
        if normalized in {"none", "bf16", "fp16", "fp32"}:
            return model

        if normalized == "int8":
            return self._dynamic_int8_quantize(model)

        if normalized == "int4":
            logger.warning("int4 quantization requested; falling back to int8 dynamic quantization")
            return self._dynamic_int8_quantize(model)

        return model

    @staticmethod
    def _dynamic_int8_quantize(model: Any) -> Any:
        try:
            import torch.nn as nn

            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return quantized
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dynamic int8 quantization unavailable; using original model: %s", exc)
            return model
