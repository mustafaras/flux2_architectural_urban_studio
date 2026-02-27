"""
Streamlit backend adapter for FLUX.2 Professional UI.

Wraps the CLI generation logic (scripts/cli.py) behind a clean, cacheable
interface that Streamlit pages can use without importing diffusers or
duplicating model-loading code.
"""

from __future__ import annotations

import io
import hashlib
import logging
import os
import time
import json
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from urllib import error as urllib_error
from urllib import request as urllib_request

import streamlit as st
import torch
from einops import rearrange
from PIL import Image, ImageDraw

from .memory_manager import MemoryManager
from .model_cache import get_model_lifecycle_manager
from .model_registry import get_model_registry, ModelType
from .model_health import get_health_monitor
from .openrouter_api_client import DEFAULT_SAMPLING_PARAMS, OpenRouterAPIClient
from .performance_metrics import RequestQueueMonitor, get_performance_collector
from .observability import (
    inc_model_cache_hits,
    observe_generation_duration,
    set_queue_length,
    set_gpu_memory_bytes,
    log_generation_completed,
)
from .sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from .system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I
from .util import FLUX2_MODEL_INFO
from .safety_pipeline import get_safety_pipeline, SafetyLevel, SafetyResult
from .error_types import ErrorContext, ErrorCategory, Severity, classify_exception
from .progress_tracker import get_progress_tracker
from .gpu_monitor import get_gpu_monitor
from .gif_generator import get_gif_generator

logger = logging.getLogger("flux2_adapter")


# ─── Adapter class ────────────────────────────────────────────────────────────

class Flux2Adapter:
    """
    Stateful adapter that holds loaded model components and exposes
    `generate()`, `edit()`, and `upsample_prompt()` methods.

    Instances are cached by Streamlit via `get_adapter()` so the heavy
    model weights are only loaded once per session / config change.
    """

    def __init__(self) -> None:
        self._model_name: str | None = None
        self._dtype_str: str | None = None
        self._cpu_offloading: bool | None = None
        self._attn_slicing: bool | None = None
        self._quantization_mode: str = "none"

        self._flow_model = None
        self._text_encoder = None
        self._upsampling_model = None   # optional auxiliary encoder (lazy-loaded)
        self._ae = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lifecycle_manager = get_model_lifecycle_manager()
        self._memory_manager = MemoryManager()
        self._metrics = get_performance_collector()
        self._queue_monitor = RequestQueueMonitor(self._metrics)
        
        # Phase 5: Model Registry & Health Monitoring
        self._model_registry = get_model_registry()
        self._health_monitor = get_health_monitor(registry=self._model_registry, enable_auto_recovery=True)
        
        self._openrouter_client: OpenRouterAPIClient | None = None
        self._openrouter_model: str | None = None
        self._openrouter_sampling_params: dict | None = None
        self._ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self._cache_key = "_flux2_generation_cache"
        self._cache_meta_key = "_flux2_generation_cache_meta"
        self._cache_stats_key = "_flux2_generation_cache_stats"
        self._cache_limit = int(os.environ.get("FLUX2_RESULT_CACHE_SIZE", "100"))
        self._progress_key = "_flux2_progress_state"
        self._network_key = "_flux2_network_metrics"
        self._progress_preview_interval = max(1, int(os.environ.get("FLUX2_PROGRESS_PREVIEW_INTERVAL", "4")))
        self._progress_decode_preview_enabled = os.environ.get("FLUX2_PROGRESS_DECODE_PREVIEW", "0").strip() == "1"
        
        # Phase 4: Real-time Progress Tracking & GPU Monitoring
        self._progress_tracker = get_progress_tracker()
        self._gpu_monitor = get_gpu_monitor()
        self._gif_generator = get_gif_generator()
        
        # Phase 6: Safety Pipeline Integration
        self._safety_pipeline = get_safety_pipeline(SafetyLevel.MODERATE)
        self._safety_enabled = True
        self._safety_check_start_time = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        model_name: str,
        dtype_str: str = "bf16",
        cpu_offloading: bool = False,
        attn_slicing: bool = False,
    ) -> None:
        """
        Load (or hot-swap) model components for *model_name*.

        If the same model is already loaded with the same settings this is a
        no-op. Changing the model unloads the previous weights first.
        """
        normalized_dtype = (dtype_str or "bf16").lower()
        quantization_mode = normalized_dtype if normalized_dtype in {"int8", "int4"} else "none"
        runtime_dtype = "bf16" if quantization_mode != "none" else normalized_dtype

        # Check if anything actually changed
        if (
            self._model_name == model_name
            and self._dtype_str == runtime_dtype
            and self._cpu_offloading == cpu_offloading
            and self._quantization_mode == quantization_mode
        ):
            logger.debug("Model already loaded — skipping reload.")
            return

        logger.info(
            "Loading model: %s  dtype=%s  offload=%s  quant=%s",
            model_name,
            runtime_dtype,
            cpu_offloading,
            quantization_mode,
        )

        if any(component is not None for component in (self._flow_model, self._text_encoder, self._ae, self._upsampling_model)):
            logger.info("Releasing current model components before loading %s", model_name)
            self._unload()

        self._memory_manager.reserve_pool()
        try:
            bundle = self._lifecycle_manager.load_bundle(
                model_name=model_name,
                dtype_str=runtime_dtype,
                cpu_offloading=cpu_offloading,
                attn_slicing=attn_slicing,
                device=self._device,
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_runtime_failure(exc, operation="load")
            raise

        self._flow_model = bundle.flow_model
        self._text_encoder = bundle.text_encoder
        self._ae = bundle.ae

        if quantization_mode in {"int8", "int4"}:
            self._flow_model = self._memory_manager.apply_quantization(self._flow_model, quantization_mode)
            self._text_encoder = self._memory_manager.apply_quantization(self._text_encoder, quantization_mode)

        # Keep auxiliary upsampling/safety model lazy to avoid OOM.
        # For non-Klein models, reuse primary encoder immediately.
        if "klein" in model_name.lower():
            self._upsampling_model = None
        else:
            self._upsampling_model = self._text_encoder

        self._model_name = model_name
        self._dtype_str = runtime_dtype
        self._cpu_offloading = cpu_offloading
        self._attn_slicing = attn_slicing
        self._quantization_mode = quantization_mode

        self._metrics.sample_runtime()

        logger.info("Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        num_steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        true_cfg: float = 1.0,
    ) -> Image.Image:
        """Run text-to-image generation and return a PIL Image."""
        logger.info(f"generate() called with prompt_len={len(prompt)}, num_steps={num_steps}, guidance={guidance}")
        self._assert_loaded()
        self._init_progress(mode="t2i", total_steps=num_steps)
        project_scope_hash, option_scope_hash, reproducibility_bundle = self._build_cache_scope_context()
        cache_key = self._make_generation_cache_key(
            mode="t2i",
            prompt=prompt,
            num_steps=num_steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
            reference_images=[],
            true_cfg=true_cfg,
            project_scope_hash=project_scope_hash,
            option_scope_hash=option_scope_hash,
            reproducibility_bundle=reproducibility_bundle,
        )

        with self._queue_monitor.request():
            self._metrics.increment("requests", 1)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.info("Cache hit! Returning cached image")
                self._metrics.increment("cache_hits", 1)
                inc_model_cache_hits()
                self._metrics.record_phase("generation_cached", 0.0)
                self._finalize_progress(status="completed", error_message=None)
                return cached

            logger.info("Cache miss - proceeding with generation")
            self._metrics.increment("cache_misses", 1)
            try:
                logger.info("Calling _run_generation()")
                with self._metrics.track_phase("generation_total"):
                    img = self._run_generation(
                        prompt=prompt,
                        num_steps=num_steps,
                        guidance=guidance,
                        width=width,
                        height=height,
                        seed=seed,
                        reference_images=[],
                        true_cfg=true_cfg,
                    )
                logger.info("_run_generation() completed successfully, caching result")
                self._cache_put(
                    cache_key,
                    img,
                    project_scope_hash=project_scope_hash,
                    option_scope_hash=option_scope_hash,
                    reproducibility_bundle=reproducibility_bundle,
                )
                self._metrics.sample_runtime()
                observe_generation_duration(float(self._metrics.snapshot().get("summary", {}).get("avg_phase_seconds", {}).get("generation_total", 0.0) or 0.0))
                set_queue_length(int(self._metrics.snapshot().get("queue", {}).get("queued", 0)))
                memory = self._metrics.snapshot().get("runtime_samples", [])
                if memory:
                    latest = memory[-1]
                    used_mb = float(latest.get("gpu_memory_used_mb") or 0.0)
                    set_gpu_memory_bytes(int(used_mb * 1024 * 1024))
                log_generation_completed(
                    logger=logger,
                    duration_ms=int(self._metrics.snapshot().get("summary", {}).get("avg_phase_seconds", {}).get("generation_total", 0.0) * 1000),
                    model=str(self._model_name or "unknown"),
                    prompt_length=len(prompt),
                    user_session=str(st.session_state.get("session_id", "sess_xxxxx")),
                )
                logger.info("generate() completed successfully")
                self._finalize_progress(status="completed", error_message=None)
                return img
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"generate() failed with exception: {exc}")
                self._metrics.increment("errors", 1)
                self._handle_runtime_failure(exc, operation="generate")
                self._finalize_progress(status="failed", error_message=str(exc))
                raise

    def edit(
        self,
        prompt: str,
        reference_images: list[Image.Image],
        num_steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        true_cfg: float = 1.0,
    ) -> Image.Image:
        """Run image-to-image generation with reference image(s) and return a PIL Image."""
        self._assert_loaded()
        self._init_progress(mode="i2i", total_steps=num_steps)
        if not reference_images:
            raise ValueError("At least one reference image is required for edit().")
        project_scope_hash, option_scope_hash, reproducibility_bundle = self._build_cache_scope_context()
        cache_key = self._make_generation_cache_key(
            mode="i2i",
            prompt=prompt,
            num_steps=num_steps,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
            reference_images=reference_images,
            true_cfg=true_cfg,
            project_scope_hash=project_scope_hash,
            option_scope_hash=option_scope_hash,
            reproducibility_bundle=reproducibility_bundle,
        )

        with self._queue_monitor.request():
            self._metrics.increment("requests", 1)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._metrics.increment("cache_hits", 1)
                inc_model_cache_hits()
                self._metrics.record_phase("edit_cached", 0.0)
                self._finalize_progress(status="completed", error_message=None)
                return cached

            self._metrics.increment("cache_misses", 1)
            try:
                with self._metrics.track_phase("edit_total"):
                    img = self._run_generation(
                        prompt=prompt,
                        num_steps=num_steps,
                        guidance=guidance,
                        width=width,
                        height=height,
                        seed=seed,
                        reference_images=reference_images,
                        true_cfg=true_cfg,
                    )
                self._cache_put(
                    cache_key,
                    img,
                    project_scope_hash=project_scope_hash,
                    option_scope_hash=option_scope_hash,
                    reproducibility_bundle=reproducibility_bundle,
                )
                self._metrics.sample_runtime()
                observe_generation_duration(float(self._metrics.snapshot().get("summary", {}).get("avg_phase_seconds", {}).get("edit_total", 0.0) or 0.0))
                set_queue_length(int(self._metrics.snapshot().get("queue", {}).get("queued", 0)))
                self._finalize_progress(status="completed", error_message=None)
                return img
            except Exception as exc:  # noqa: BLE001
                self._metrics.increment("errors", 1)
                self._handle_runtime_failure(exc, operation="edit")
                self._finalize_progress(status="failed", error_message=str(exc))
                raise

    def upsample_prompt(
        self,
        prompt: str,
        backend: str = "none",
        ollama_model: str = "qwen3:30b",
        ollama_temperature: float = 0.15,
        openrouter_model: str = "mistralai/pixtral-large-2411",
        api_key: str = "",
        reference_images: list[Image.Image] | None = None,
    ) -> str:
        """
        Expand *prompt* using the requested *backend*.

        Args:
            prompt:            Original short prompt.
            backend:           ``"none"`` | ``"local"`` | ``"openrouter"``
            openrouter_model:  Model identifier on OpenRouter (used when backend=="openrouter").
            api_key:           OpenRouter API key (used when backend=="openrouter").
            reference_images:  If provided, uses i2i upsampling system message.

        Returns:
            The expanded prompt string (falls back to the original on any error).
        """
        if backend == "none" or not prompt.strip():
            return prompt

        imgs = reference_images or []

        try:
            if backend == "local":
                return self._upsample_prompt_ollama(
                    prompt=prompt,
                    model=ollama_model,
                    temperature=ollama_temperature,
                    reference_images=imgs,
                )

            elif backend == "openrouter":
                if not api_key:
                    api_key = os.environ.get("OPENROUTER_API_KEY", "")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY is not set.")
                os.environ["OPENROUTER_API_KEY"] = api_key

                sampling_params = DEFAULT_SAMPLING_PARAMS.get(openrouter_model, {})
                if (
                    self._openrouter_client is None
                    or self._openrouter_model != openrouter_model
                    or self._openrouter_sampling_params != sampling_params
                ):
                    self._openrouter_client = OpenRouterAPIClient(
                        sampling_params=sampling_params,
                        model=openrouter_model,
                    )
                    self._openrouter_model = openrouter_model
                    self._openrouter_sampling_params = sampling_params

                # Retry transient network/API errors with backoff
                last_error: Exception | None = None
                for attempt in range(3):
                    started = time.perf_counter()
                    try:
                        results = self._openrouter_client.upsample_prompt(
                            [prompt],
                            img=[imgs] if imgs else None,
                        )
                        self._record_api_call(
                            provider="openrouter",
                            latency_ms=1000.0 * (time.perf_counter() - started),
                            ok=True,
                        )
                        return results[0] if results else prompt
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        self._record_api_call(
                            provider="openrouter",
                            latency_ms=1000.0 * (time.perf_counter() - started),
                            ok=False,
                            error_message=str(exc),
                            rate_limited=self._is_rate_limited_error(exc),
                        )
                        if attempt < 2:
                            time.sleep(0.8 * (2**attempt))
                            continue
                if last_error is not None:
                    raise last_error

        except Exception as exc:  # noqa: BLE001
            logger.warning("Prompt upsampling failed (%s); returning original. Error: %s", backend, exc)
            return prompt

        return prompt

    def list_ollama_models(self) -> list[str]:
        url = f"{self._ollama_host}/api/tags"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            names = [m.get("name", "") for m in models if m.get("name")]
            return names
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

    def warm_startup_models(self, model_names: list[str]) -> None:
        wait_for_warmup = os.environ.get("FLUX2_BLOCKING_WARMUP", "1").strip() == "1"
        wait_timeout_s = float(os.environ.get("FLUX2_BLOCKING_WARMUP_TIMEOUT", "120"))
        self._lifecycle_manager.warm_models_async(
            model_names=model_names,
            dtype_str=self._dtype_str or "bf16",
            cpu_offloading=bool(self._cpu_offloading) if self._cpu_offloading is not None else False,
            attn_slicing=bool(self._attn_slicing) if self._attn_slicing is not None else True,
            device=self._device,
            wait=wait_for_warmup,
            wait_timeout_s=wait_timeout_s,
        )

    def get_performance_snapshot(self) -> dict:
        snapshot = self._metrics.snapshot()
        try:
            current_memory = asdict(self._memory_manager.snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory snapshot unavailable after runtime failure: %s", exc)
            current_memory = {
                "cuda_available": bool(torch.cuda.is_available()),
                "total_mb": 0.0,
                "used_mb": 0.0,
                "free_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "used_ratio": 0.0,
            }
        snapshot["memory"] = {
            "current": current_memory,
            "target_peak_ratio": self._memory_manager.target_peak_ratio,
        }
        return snapshot

    def clear_generation_cache(self) -> None:
        if self._cache_key in st.session_state:
            del st.session_state[self._cache_key]

    def get_progress_snapshot(self) -> dict:
        snapshot = self._progress_tracker.snapshot()
        # Add network metrics
        snapshot["network"] = self._get_network_metrics()
        return snapshot

    def _get_network_metrics(self) -> dict:
        if self._network_key not in st.session_state:
            st.session_state[self._network_key] = {
                "api_latency_ms": [],
                "request_count": 0,
                "rate_limit_status": "unknown",
                "last_provider": None,
                "last_error": None,
            }
        return dict(st.session_state[self._network_key])

    def _record_api_call(
        self,
        provider: str,
        latency_ms: float,
        ok: bool,
        error_message: str | None = None,
        rate_limited: bool = False,
    ) -> None:
        payload = self._get_network_metrics()
        latencies = list(payload.get("api_latency_ms", []))
        latencies.append(float(latency_ms))
        latencies = latencies[-128:]

        payload["api_latency_ms"] = latencies
        payload["request_count"] = int(payload.get("request_count", 0)) + 1
        payload["last_provider"] = provider
        payload["last_error"] = error_message
        if rate_limited:
            payload["rate_limit_status"] = "rate_limited"
        elif ok:
            payload["rate_limit_status"] = "ok"

        st.session_state[self._network_key] = payload

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "rate limit" in message or "429" in message or "too many requests" in message

    def _init_progress(self, mode: str, total_steps: int) -> None:
        self._progress_tracker.begin(mode=mode, total_steps=total_steps)
        
        # Set frame capture function for intermediate preview
        def capture_frame(step_idx: int, total_steps: int, latent: torch.Tensor, latent_ids: torch.Tensor) -> bytes | None:
            return self._make_intermediate_preview_frame(
                latent=latent,
                latent_ids=latent_ids,
                step_idx=step_idx,
                total_steps=total_steps,
                progress_percent=100.0 * step_idx / max(1, total_steps),
            )
        
        self._progress_tracker.set_frame_capture_fn(capture_frame)

    def _update_progress(
        self,
        step_idx: int,
        total_steps: int,
        t_curr: float,
        t_prev: float,
        latent: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> None:
        self._progress_tracker.update(
            step_idx=step_idx,
            total_steps=total_steps,
            t_curr=t_curr,
            t_prev=t_prev,
            latent=latent,
            latent_ids=latent_ids,
        )
        
        # Sample GPU metrics periodically
        if step_idx % 4 == 0:
            self._metrics.sample_runtime()
            gpu_metrics = self._gpu_monitor.snapshot()
            if gpu_metrics:
                # Log GPU metrics for monitoring
                logger.debug(
                    f"GPU: {gpu_metrics.vram_used_mb:.0f}MB/{gpu_metrics.vram_total_mb:.0f}MB, "
                    f"Util: {gpu_metrics.gpu_util_percent:.1f}%, "
                    f"Temp: {gpu_metrics.temperature_c:.1f}°C"
                )

    @staticmethod
    def _make_progress_preview_frame(step_idx: int, total_steps: int, progress_percent: float) -> bytes:
        image = Image.new("RGB", (320, 64), color=(24, 26, 32))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 36, 310, 52), outline=(70, 80, 90), width=1)
        bar_end = int(10 + (300 * (max(0.0, min(100.0, progress_percent)) / 100.0)))
        draw.rectangle((10, 36, bar_end, 52), fill=(64, 170, 255))
        draw.text((10, 10), f"Step {step_idx}/{max(1, total_steps)}", fill=(220, 230, 240))

        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def _make_intermediate_preview_frame(
        self,
        latent: torch.Tensor,
        latent_ids: torch.Tensor,
        step_idx: int,
        total_steps: int,
        progress_percent: float,
    ) -> bytes:
        if not self._progress_decode_preview_enabled:
            return self._make_progress_preview_frame(step_idx, total_steps, progress_percent)

        try:
            with torch.no_grad():
                if self._ae is None:
                    raise RuntimeError("Autoencoder is not available")

                decoded_latent = torch.cat(scatter_ids(latent, latent_ids)).squeeze(2)
                decoded = self._ae.decode(decoded_latent).float().clamp(-1, 1)
                arr = rearrange(decoded[0], "c h w -> h w c")
                image = Image.fromarray((127.5 * (arr + 1.0)).cpu().byte().numpy())
                image.thumbnail((320, 320))

                out = io.BytesIO()
                image.save(out, format="PNG")
                return out.getvalue()
        except Exception:
            return self._make_progress_preview_frame(step_idx, total_steps, progress_percent)

    def _finalize_progress(self, status: str, error_message: str | None) -> None:
        self._progress_tracker.finalize(status=status, error_message=error_message)

    def _get_cache_dict(self) -> dict[str, bytes]:
        if self._cache_key not in st.session_state:
            st.session_state[self._cache_key] = {}
        cache = st.session_state[self._cache_key]
        if not isinstance(cache, dict):
            cache = {}
            st.session_state[self._cache_key] = cache
        return cache

    def _get_cache_meta_dict(self) -> dict[str, dict[str, object]]:
        if self._cache_meta_key not in st.session_state:
            st.session_state[self._cache_meta_key] = {}
        meta = st.session_state[self._cache_meta_key]
        if not isinstance(meta, dict):
            meta = {}
            st.session_state[self._cache_meta_key] = meta
        return meta

    def _get_cache_stats(self) -> dict[str, object]:
        if self._cache_stats_key not in st.session_state or not isinstance(st.session_state.get(self._cache_stats_key), dict):
            st.session_state[self._cache_stats_key] = {
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "puts": 0,
                "evictions": 0,
                "invalidations": 0,
                "invalidated_entries": 0,
                "projects": {},
            }
        return st.session_state[self._cache_stats_key]

    def _record_cache_stat(self, event: str, project_scope_hash: str = "") -> None:
        stats = self._get_cache_stats()
        event_key = {
            "request": "requests",
            "hit": "hits",
            "miss": "misses",
            "put": "puts",
            "evict": "evictions",
            "invalidate": "invalidations",
        }.get(event)
        if event_key:
            stats[event_key] = int(stats.get(event_key, 0)) + 1

        projects = stats.get("projects", {})
        if not isinstance(projects, dict):
            projects = {}
        project_key = project_scope_hash or "global"
        project_bucket = projects.get(project_key, {})
        if not isinstance(project_bucket, dict):
            project_bucket = {}
        if event_key:
            project_bucket[event_key] = int(project_bucket.get(event_key, 0)) + 1
        projects[project_key] = project_bucket
        stats["projects"] = projects
        st.session_state[self._cache_stats_key] = stats

    def get_generation_cache_stats(self) -> dict[str, object]:
        return copy.deepcopy(self._get_cache_stats())

    def invalidate_generation_cache(self, *, project_id: str | None = None, reason: str = "manual") -> int:
        cache = self._get_cache_dict()
        meta = self._get_cache_meta_dict()
        removed = 0
        target_project_hash = hashlib.sha256(str(project_id).encode("utf-8")).hexdigest() if project_id else ""

        if target_project_hash:
            keys = [
                k
                for k, m in meta.items()
                if isinstance(m, dict) and str(m.get("project_scope_hash", "")) == target_project_hash
            ]
        else:
            keys = list(cache.keys())

        for key in keys:
            if key in cache:
                cache.pop(key, None)
                removed += 1
            meta.pop(key, None)

        stats = self._get_cache_stats()
        stats["invalidated_entries"] = int(stats.get("invalidated_entries", 0)) + removed
        st.session_state[self._cache_stats_key] = stats
        self._record_cache_stat("invalidate", project_scope_hash=target_project_hash)
        logger.info("Cache invalidation completed: removed=%s reason=%s project_id=%s", removed, reason, project_id or "*")

        st.session_state[self._cache_key] = cache
        st.session_state[self._cache_meta_key] = meta
        return removed

    def _build_cache_scope_context(self) -> tuple[str, str, dict[str, object]]:
        active_project_id = str(st.session_state.get("active_project_id", "")).strip()
        project_scope_hash = hashlib.sha256(active_project_id.encode("utf-8")).hexdigest() if active_project_id else ""

        phase2_payload = st.session_state.get("phase2_controls_payload", {})
        if not isinstance(phase2_payload, dict):
            phase2_payload = {}
        prompt_payload = st.session_state.get("prompt_taxonomy_payload", {})
        if not isinstance(prompt_payload, dict):
            prompt_payload = {}

        option_payload = {
            "option_id": str(st.session_state.get("selected_option_id", "")).strip(),
            "seed_group_id": str(phase2_payload.get("seed_group_id", "")).strip(),
            "scenario_id": str(st.session_state.get("selected_scenario_id", "")).strip(),
        }
        option_scope_hash = hashlib.sha256(
            json.dumps(option_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        reproducibility_bundle = {
            "cache_schema_version": "x1-v1",
            "template_version": str(prompt_payload.get("template_version", "")),
            "template_name": str(prompt_payload.get("template_name", "")),
            "massing_preset": str(phase2_payload.get("massing_preset", "")),
            "facade_style_pack": str(phase2_payload.get("facade_style_pack", "")),
            "continuity_profile": str(phase2_payload.get("continuity_profile", "")),
            "time_of_day": str(phase2_payload.get("time_of_day", "")),
            "seasonal_profile": str(phase2_payload.get("seasonal_profile", "")),
            "climate_mood": str(phase2_payload.get("climate_mood", "")),
            "project_scope_hash": project_scope_hash,
            "option_scope_hash": option_scope_hash,
        }
        return project_scope_hash, option_scope_hash, reproducibility_bundle

    def _cache_get(self, key: str) -> Image.Image | None:
        cache = self._get_cache_dict()
        meta = self._get_cache_meta_dict()
        payload = cache.get(key)
        project_scope_hash = ""
        record = meta.get(key)
        if isinstance(record, dict):
            project_scope_hash = str(record.get("project_scope_hash", ""))
        self._record_cache_stat("request", project_scope_hash=project_scope_hash)
        if payload is None:
            self._record_cache_stat("miss", project_scope_hash=project_scope_hash)
            return None
        cache.pop(key, None)
        cache[key] = payload
        if isinstance(record, dict):
            meta[key] = {
                **record,
                "last_hit_ts": time.time(),
                "hits": int(record.get("hits", 0)) + 1,
            }
        st.session_state[self._cache_key] = cache
        st.session_state[self._cache_meta_key] = meta
        self._record_cache_stat("hit", project_scope_hash=project_scope_hash)
        return Image.open(io.BytesIO(payload)).convert("RGB")

    def _cache_put(
        self,
        key: str,
        img: Image.Image,
        *,
        project_scope_hash: str = "",
        option_scope_hash: str = "",
        reproducibility_bundle: dict[str, object] | None = None,
    ) -> None:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()

        cache = self._get_cache_dict()
        meta = self._get_cache_meta_dict()
        if key in cache:
            cache.pop(key, None)
        cache[key] = payload
        bundle = reproducibility_bundle if isinstance(reproducibility_bundle, dict) else {}
        now_ts = time.time()
        meta[key] = {
            "project_scope_hash": project_scope_hash,
            "option_scope_hash": option_scope_hash,
            "reproducibility_hash": hashlib.sha256(
                json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if bundle
            else "",
            "created_ts": now_ts,
            "last_hit_ts": now_ts,
            "hits": 0,
        }
        self._record_cache_stat("put", project_scope_hash=project_scope_hash)

        while len(cache) > self._cache_limit:
            oldest_key = next(iter(cache.keys()))
            cache.pop(oldest_key, None)
            meta.pop(oldest_key, None)
            self._record_cache_stat("evict", project_scope_hash=project_scope_hash)

        st.session_state[self._cache_key] = cache
        st.session_state[self._cache_meta_key] = meta

    def _make_generation_cache_key(
        self,
        mode: str,
        prompt: str,
        num_steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        reference_images: list[Image.Image],
        true_cfg: float = 1.0,
        project_scope_hash: str = "",
        option_scope_hash: str = "",
        reproducibility_bundle: dict[str, object] | None = None,
    ) -> str:
        ref_hashes: list[str] = []
        for img in reference_images:
            b = io.BytesIO()
            img.save(b, format="PNG")
            ref_hashes.append(hashlib.sha256(b.getvalue()).hexdigest())

        bundle = reproducibility_bundle if isinstance(reproducibility_bundle, dict) else {}
        reproducibility_hash = hashlib.sha256(
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        key_payload = {
            "schema": "x1-cache-key-v1",
            "mode": mode,
            "model": self._model_name,
            "dtype": self._dtype_str,
            "offload": self._cpu_offloading,
            "attn_slicing": self._attn_slicing,
            "quantization": self._quantization_mode,
            "prompt": prompt,
            "steps": int(num_steps),
            "guidance": float(guidance),
            "true_cfg": float(true_cfg),
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "reference_hashes": ref_hashes,
            "project_scope_hash": project_scope_hash,
            "option_scope_hash": option_scope_hash,
            "reproducibility_hash": reproducibility_hash,
        }
        raw = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _upsample_prompt_ollama(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.15,
        reference_images: list[Image.Image] | None = None,
    ) -> str:
        system_msg = SYSTEM_MESSAGE_UPSAMPLING_I2I if reference_images else SYSTEM_MESSAGE_UPSAMPLING_T2I
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_msg,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if reference_images:
            # Ollama multimodal models may optionally consume images as base64 strings.
            from flux2.util import image_to_base64

            payload["images"] = [image_to_base64(img) for img in reference_images]

        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self._ollama_host}/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib_request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = (result.get("response") or "").strip()
            self._record_api_call(
                provider="ollama",
                latency_ms=1000.0 * (time.perf_counter() - started),
                ok=True,
            )
            return text or prompt
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            self._record_api_call(
                provider="ollama",
                latency_ms=1000.0 * (time.perf_counter() - started),
                ok=False,
                error_message=detail,
                rate_limited=(exc.code == 429),
            )
            raise RuntimeError(f"Ollama HTTP error: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            self._record_api_call(
                provider="ollama",
                latency_ms=1000.0 * (time.perf_counter() - started),
                ok=False,
                error_message=str(exc),
                rate_limited=self._is_rate_limited_error(exc),
            )
            raise RuntimeError(f"Ollama upsampling failed: {exc}") from exc

    def check_prompt_safety(self, prompt: str) -> bool:
        """Check prompt safety using Phase 6 SafetyPipeline.
        
        Returns: True if safe, False if flagged by safety checks
        Logs: Violation details and timing metrics
        """
        if not self._safety_enabled or self._safety_pipeline is None:
            return True
        
        start_time = time.time()
        try:
            result: SafetyResult = self._safety_pipeline.check_prompt(prompt)
            latency_ms = (time.time() - start_time) * 1000
            
            if not result.is_safe:
                logger.warning(
                    f"Prompt blocked by safety: {[v.type.value for v in result.violations]} "
                    f"(latency: {latency_ms:.2f}ms)"
                )
                return False
            
            logger.debug(f"Prompt passed safety (latency: {latency_ms:.2f}ms)")
            return True
            
        except Exception as exc:
            logger.error(f"Safety check error: {exc}")
            return True

    def check_image_safety(self, img: Image.Image) -> bool:
        """Check generated image safety using Phase 6 SafetyPipeline.
        
        Returns: True if safe, False if flagged by safety checks
        Logs: Violation details with region heatmaps
        """
        if not self._safety_enabled or self._safety_pipeline is None:
            return True
        
        start_time = time.time()
        try:
            result: SafetyResult = self._safety_pipeline.check_image(img)
            latency_ms = (time.time() - start_time) * 1000
            
            if not result.is_safe:
                logger.warning(
                    f"Image blocked by safety: {[v.type.value for v in result.violations]} "
                    f"(latency: {latency_ms:.2f}ms)"
                )
                return False
            
            logger.debug(f"Image passed safety (latency: {latency_ms:.2f}ms)")
            return True
            
        except Exception as exc:
            logger.error(f"Image safety check error: {exc}")
            return True
    
    def get_safety_result_detail(self, prompt: str) -> SafetyResult | None:
        """Get detailed safety check result for UI display."""
        if not self._safety_enabled or self._safety_pipeline is None:
            return None
        
        try:
            return self._safety_pipeline.check_prompt(prompt)
        except Exception as exc:
            logger.error(f"Failed to get safety details: {exc}")
            return None
    
    def set_safety_level(self, level: SafetyLevel) -> None:
        """Update safety level dynamically."""
        if self._safety_pipeline is not None:
            self._safety_pipeline.set_safety_level(level)
            logger.info(f"Safety level updated to: {level.value}")
    
    def reload_safety_config(self) -> bool:
        """Reload safety configuration from disk."""
        if self._safety_pipeline is not None:
            return self._safety_pipeline.reload_config()
        return False
    
    def disable_safety_checks(self) -> None:
        """Disable safety checks (for testing/advanced users)."""
        self._safety_enabled = False
        logger.warning("Safety checks DISABLED - use with caution!")
    
    def enable_safety_checks(self) -> None:
        """Re-enable safety checks."""
        self._safety_enabled = True
        logger.info("Safety checks enabled")

    def _ensure_aux_text_encoder(self) -> bool:
        """Ensure optional auxiliary text encoder exists when needed."""
        if self._upsampling_model is not None:
            return True

        if self._model_name is None:
            return False

        if "klein" not in self._model_name.lower():
            self._upsampling_model = self._text_encoder
            return self._upsampling_model is not None

        # For Klein, load heavy Dev encoder lazily on CPU to avoid VRAM spikes.
        try:
            logger.info("Lazy-loading auxiliary encoder (flux.2-dev) on CPU.")
            self._upsampling_model = load_text_encoder("flux.2-dev", device="cpu")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auxiliary encoder unavailable; safety/upsampling disabled. Error: %s", exc)
            self._upsampling_model = None
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if self._flow_model is None or self._text_encoder is None or self._ae is None:
            raise RuntimeError(
                "Model components are not loaded. Call adapter.load() before generating."
            )

    def _unload(self) -> None:
        """Release previously loaded model references and free GPU memory."""
        self._flow_model = None
        self._text_encoder = None
        self._upsampling_model = None
        self._ae = None
        self._model_name = None
        self._dtype_str = None
        self._cpu_offloading = None
        self._attn_slicing = None
        self._quantization_mode = "none"
        self._memory_manager.release_pool()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _run_generation(
        self,
        prompt: str,
        num_steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: int,
        reference_images: list[Image.Image],
        true_cfg: float = 1.0,
    ) -> Image.Image:
        model_info = FLUX2_MODEL_INFO[self._model_name.lower()]
        guidance_distilled: bool = model_info["guidance_distilled"]
        cpu_offloading = self._cpu_offloading

        if not cpu_offloading:
            self._memory_manager.maybe_reload_to_gpu(
                self._device,
                self._flow_model,
                self._text_encoder,
                self._upsampling_model,
            )

        with torch.no_grad():
            with self._metrics.track_phase("preprocessing"):
                ref_tokens, ref_ids = encode_image_refs(self._ae, reference_images)

                if guidance_distilled:
                    ctx = self._text_encoder([prompt]).to(torch.bfloat16)
                else:
                    ctx_empty = self._text_encoder([""]).to(torch.bfloat16)
                    ctx_prompt = self._text_encoder([prompt]).to(torch.bfloat16)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

                if cpu_offloading:
                    self._text_encoder = self._text_encoder.cpu()
                    if self._upsampling_model is not None and self._upsampling_model is not self._text_encoder:
                        self._upsampling_model = self._upsampling_model.cpu()
                    torch.cuda.empty_cache()
                    self._flow_model = self._flow_model.to(self._device)

                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device=str(self._device)).manual_seed(seed)
                randn = torch.randn(
                    shape,
                    generator=generator,
                    dtype=torch.bfloat16,
                    device=self._device,
                )
                x, x_ids = batched_prc_img(randn)

            with self._metrics.track_phase("inference"):
                logger.info(f"Starting inference phase: guidance_distilled={guidance_distilled}, num_steps={num_steps}")
                timesteps = get_schedule(num_steps, x.shape[1])
                logger.info(f"Timesteps generated: {len(timesteps)} steps")
                
                # Set up progress callback for real-time tracking
                progress_callback = lambda step_idx, total_steps, t_curr, t_prev, latent: self._update_progress(
                    step_idx=step_idx,
                    total_steps=total_steps,
                    t_curr=t_curr,
                    t_prev=t_prev,
                    latent_ids=x_ids,
                    latent=latent,
                )
                
                try:
                    if guidance_distilled:
                        logger.info("Calling denoise() for guidance_distilled model")
                        x = denoise(
                            self._flow_model,
                            x,
                            x_ids,
                            ctx,
                            ctx_ids,
                            timesteps=timesteps,
                            guidance=guidance,
                            img_cond_seq=ref_tokens,
                            img_cond_seq_ids=ref_ids,
                            progress_callback=progress_callback,
                        )
                        logger.info("denoise() completed successfully")
                    else:
                        logger.info("Calling denoise_cfg() for non-distilled model")
                        x = denoise_cfg(
                            self._flow_model,
                            x,
                            x_ids,
                            ctx,
                            ctx_ids,
                            timesteps=timesteps,
                            guidance=guidance,
                            img_cond_seq=ref_tokens,
                            img_cond_seq_ids=ref_ids,
                            progress_callback=progress_callback,
                        )
                        logger.info("denoise_cfg() completed successfully")
                except Exception as e:
                    logger.exception(f"Inference failed: {e}")
                    raise

            with self._metrics.track_phase("postprocessing"):
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                try:
                    x = self._ae.decode(x).float()
                except Exception as exc:  # noqa: BLE001
                    if self._is_cuda_oom(exc):
                        logger.warning("OOM during decode; retrying decode on CPU.")
                        x = self._decode_on_cpu_fallback(x)
                    else:
                        raise

                if cpu_offloading:
                    self._flow_model = self._flow_model.cpu()
                    torch.cuda.empty_cache()
                    self._text_encoder = self._text_encoder.to(self._device)
                    if self._upsampling_model is not None and self._upsampling_model is not self._text_encoder:
                        self._upsampling_model = self._upsampling_model.to(self._device)

            if cpu_offloading:
                self._memory_manager.maybe_offload_models(self._flow_model)

        # ── Convert to PIL Image ───────────────────────────────────────────
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img

    def _decode_on_cpu_fallback(self, latent: torch.Tensor) -> torch.Tensor:
        if self._ae is None:
            raise RuntimeError("Autoencoder is not loaded")

        ae = self._ae
        restore_to_gpu = bool(torch.cuda.is_available() and self._device.type == "cuda" and not self._cpu_offloading)

        ae = ae.to("cpu")
        self._ae = ae
        self._memory_manager.clear_unused()

        latent_cpu = latent.detach().to("cpu")
        decoded = ae.decode(latent_cpu).float()

        if restore_to_gpu:
            try:
                self._ae = self._ae.to(self._device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to restore autoencoder to GPU after CPU decode fallback: %s", exc)
        return decoded

    # ── Phase 5: Model Management & Health Monitoring ──────────────────────

    def get_available_models(self) -> list:
        """Get list of available models from registry."""
        return self._model_registry.list_available()

    def get_model_info(self, model_id: str) -> dict | None:
        """Get detailed model information."""
        metadata = self._model_registry.get_model(model_id)
        if metadata:
            return {
                "name": metadata.name,
                "version": metadata.version,
                "status": metadata.status.value,
                "type": metadata.model_type.value,
                "parameter_count": metadata.parameter_count,
                "quantization": metadata.quantization,
                "vram_gb": metadata.vram_requirement_gb,
                "avg_inference_time_s": metadata.avg_inference_time_s,
                "performance_score": metadata.performance_score,
                "description": metadata.description,
                "tags": metadata.tags
            }
        return None

    def check_model_health(self, model_id: str) -> dict:
        """Check health status of a model."""
        result = self._health_monitor.check_model_health(model_id)
        return {
            "status": result.status.value,
            "file_exists": result.file_exists,
            "hash_valid": result.hash_valid,
            "loadable": result.loadable,
            "vram_available_mb": result.vram_available_mb,
            "vram_required_mb": result.vram_required_mb,
            "device": result.device,
            "summary": result.summary()
        }

    def verify_startup_models(self) -> dict:
        """Run comprehensive startup health check on all models."""
        results = self._health_monitor.verify_startup()
        return {
            model_id: {
                "status": result.status.value,
                "healthy": result.is_healthy(),
                "summary": result.summary()
            }
            for model_id, result in results.items()
        }

    def get_model_registry_stats(self) -> dict:
        """Get registry statistics."""
        return self._model_registry.get_statistics()

    def compare_models(self, model_a_id: str, model_b_id: str) -> dict:
        """Compare two models for A/B testing."""
        return self._model_registry.compare_models(model_a_id, model_b_id)

    def record_generation_metrics(self, model_id: str, inference_time_s: float, 
                                 vram_used_mb: float, quality_score: float = 0.0) -> bool:
        """Record generation metrics for a model."""
        return self._model_registry.record_inference_time(model_id, inference_time_s, vram_used_mb)

    def add_custom_model(self, name: str, file_path: str | Path, 
                        model_type: str = "custom", **kwargs) -> str | None:
        """Register a custom or LoRA model."""
        from pathlib import Path
        from flux2.model_registry import ModelType
        
        file_path = Path(file_path)
        
        # Map string type to ModelType enum
        type_map = {
            "custom": ModelType.CUSTOM,
            "lora": ModelType.LORA,
            "text_encoder": ModelType.TEXT_ENCODER,
            "autoencoder": ModelType.AUTOENCODER
        }
        
        model_type_enum = type_map.get(model_type.lower(), ModelType.CUSTOM)
        
        return self._model_registry.add_custom_model(
            name=name,
            file_path=file_path,
            model_type=model_type_enum,
            **kwargs
        )

    def get_device_compatible_models(self) -> list:
        """Get models compatible with current device."""
        available = self._model_registry.list_available()
        
        if torch.cuda.is_available():
            cuda_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return [m for m in available if m.vram_requirement_gb <= cuda_vram_gb]
        
        return available

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        message = str(exc).lower()
        return "out of memory" in message and ("cuda" in message or "accelerator" in message)

    def _handle_runtime_failure(self, exc: Exception, operation: str) -> None:
        if self._is_cuda_oom(exc):
            logger.warning("OOM detected during %s; releasing model references and clearing CUDA cache.", operation)
            self._unload()
            self._memory_manager.clear_unused()
            return

        message = str(exc).lower()
        if "cuda" in message:
            logger.warning("CUDA runtime error detected during %s; attempting cache cleanup.", operation)
            self._memory_manager.clear_unused()


# ─── Streamlit-cached singleton ───────────────────────────────────────────────

@st.cache_resource
def get_adapter() -> Flux2Adapter:
    """
    Return the session-scoped Flux2Adapter singleton.

    Streamlit's ``@st.cache_resource`` decorator ensures this is constructed
    once and reused across reruns, so model weights stay in memory.
    """
    adapter = Flux2Adapter()
    adapter.warm_startup_models(["flux.2-klein-4b", "flux.2-klein-9b"])
    return adapter
