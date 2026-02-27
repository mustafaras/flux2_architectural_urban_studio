"""Model lifecycle and Streamlit resource caching for FLUX.2."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from flux2.performance_metrics import get_performance_collector
from flux2.util import load_ae, load_flow_model, load_text_encoder


logger = logging.getLogger("flux2_model_cache")


@dataclass(slots=True)
class LoadedBundle:
    model_name: str
    dtype_str: str
    cpu_offloading: bool
    attn_slicing: bool
    flow_model: Any
    text_encoder: Any
    ae: Any
    loaded_at: float


def _streamlit_cache_resource(fn):
    if st is None:
        return fn
    return st.cache_resource(show_spinner=False)(fn)


@_streamlit_cache_resource
def _load_bundle_cached(
    model_name: str,
    dtype_str: str,
    cpu_offloading: bool,
    attn_slicing: bool,
    device_str: str,
) -> LoadedBundle:
    load_device = "cpu" if cpu_offloading else device_str

    flow_model = load_flow_model(model_name, device=load_device)
    flow_model.eval()

    text_encoder = load_text_encoder(model_name, device=load_device)
    text_encoder.eval()

    ae = load_ae(model_name, device=load_device)
    ae.eval()

    return LoadedBundle(
        model_name=model_name,
        dtype_str=dtype_str,
        cpu_offloading=cpu_offloading,
        attn_slicing=attn_slicing,
        flow_model=flow_model,
        text_encoder=text_encoder,
        ae=ae,
        loaded_at=time.time(),
    )


class ModelLifecycleManager:
    """Coordinates cached model loads, lazy loads, and startup warming."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._warm_thread: threading.Thread | None = None
        self._collector = get_performance_collector()
        self._active_cache_signature: tuple[str, str, bool, bool, str] | None = None

    def _prepare_single_active_cache(self, signature: tuple[str, str, bool, bool, str]) -> None:
        if self._active_cache_signature is None:
            self._active_cache_signature = signature
            return
        if self._active_cache_signature == signature:
            return

        if st is not None:
            try:
                _load_bundle_cached.clear()  # type: ignore[attr-defined]
                logger.info("Cleared model cache before switching active model bundle.")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to clear Streamlit model cache during switch: %s", exc)

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        self._active_cache_signature = signature

    def load_bundle(
        self,
        model_name: str,
        dtype_str: str,
        cpu_offloading: bool,
        attn_slicing: bool,
        device: torch.device,
    ) -> LoadedBundle:
        signature = (model_name, dtype_str, bool(cpu_offloading), bool(attn_slicing), str(device))
        with self._lock:
            self._prepare_single_active_cache(signature)

        started = time.perf_counter()
        bundle = _load_bundle_cached(
            model_name=model_name,
            dtype_str=dtype_str,
            cpu_offloading=cpu_offloading,
            attn_slicing=attn_slicing,
            device_str=str(device),
        )
        elapsed = time.perf_counter() - started
        self._collector.record_phase("model_load", elapsed)
        self._collector.increment("loads", 1)
        return bundle

    def warm_models_async(
        self,
        model_names: list[str],
        dtype_str: str,
        cpu_offloading: bool,
        attn_slicing: bool,
        device: torch.device,
        wait: bool = False,
        wait_timeout_s: float | None = None,
    ) -> None:
        if not model_names:
            return

        with self._lock:
            if self._warm_thread and self._warm_thread.is_alive():
                return

            def worker() -> None:
                for model_name in model_names:
                    try:
                        signature = (model_name, dtype_str, bool(cpu_offloading), bool(attn_slicing), str(device))
                        with self._lock:
                            self._prepare_single_active_cache(signature)
                        _load_bundle_cached(
                            model_name=model_name,
                            dtype_str=dtype_str,
                            cpu_offloading=cpu_offloading,
                            attn_slicing=attn_slicing,
                            device_str=str(device),
                        )
                    except Exception:
                        continue

            self._warm_thread = threading.Thread(target=worker, daemon=True, name="flux2-model-warmer")
            self._warm_thread.start()

        if wait:
            self.wait_for_warmup(wait_timeout_s)

    def wait_for_warmup(self, timeout_s: float | None = None) -> None:
        thread: threading.Thread | None
        with self._lock:
            thread = self._warm_thread
        if thread is None:
            return
        thread.join(timeout=timeout_s)

    def clear_all(self) -> None:
        if st is not None:
            try:
                _load_bundle_cached.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._active_cache_signature = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_GLOBAL_MANAGER: ModelLifecycleManager | None = None


def get_model_lifecycle_manager() -> ModelLifecycleManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ModelLifecycleManager()
    return _GLOBAL_MANAGER
