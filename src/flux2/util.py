import base64
import io
import os
import sys
from pathlib import Path
import shutil

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weights_download_dir(repo_id: str) -> Path:
    safe_repo = repo_id.replace("/", "--")
    path = _repo_root() / "weights" / "downloads" / safe_repo
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_to_repo_weights(repo_id: str, filename: str, token: str | None) -> str:
    target_dir = _weights_download_dir(repo_id)

    try:
        local_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        return str(Path(local_path).resolve())
    except TypeError:
        cached_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
        )
        out_path = target_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, out_path)
        return str(out_path.resolve())


def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()

    token_files = [
        Path.cwd() / "hf_token.txt",
        Path.cwd().parent / "hf_token.txt",
    ]
    for token_file in token_files:
        if token_file.exists():
            candidate = token_file.read_text(encoding="utf-8").strip()
            if candidate:
                return candidate
    return None


def _local_only_mode() -> bool:
    return os.environ.get("FLUX2_LOCAL_ONLY", "1").strip() == "1"


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    hf_token = _resolve_hf_token()

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        weight_path = None
        if config["model_path"] in os.environ:
            candidate = os.environ[config["model_path"]]
            if os.path.exists(candidate):
                weight_path = candidate

        if weight_path is None:
            default_weight_path = _repo_root() / "weights" / config["filename"]
            if default_weight_path.exists():
                weight_path = str(default_weight_path)

        if weight_path is not None:
            pass
        else:
            if _local_only_mode():
                raise RuntimeError(
                    f"Local-only mode is active and `{config['model_path']}` is not set to a valid file. "
                    f"Download/copy `{config['filename']}` to `weights/{config['filename']}` "
                    "or set the corresponding path in Settings > Custom Model Paths."
                )
            # download from huggingface
            try:
                weight_path = _download_to_repo_weights(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    token=hf_token,
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    if model_name.lower() == "flux.2-dev" and _local_only_mode():
        raise RuntimeError(
            "Local-only mode is active and FLUX.2-Dev text encoder is not configured as a local bundle. "
            "Use Klein models for fully local inference or disable local-only mode."
        )
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    hf_token = _resolve_hf_token()

    weight_path = None
    if "AE_MODEL_PATH" in os.environ:
        candidate = os.environ["AE_MODEL_PATH"]
        if os.path.exists(candidate):
            weight_path = candidate

    if weight_path is None:
        default_ae_path = _repo_root() / "weights" / "ae.safetensors"
        if default_ae_path.exists():
            weight_path = str(default_ae_path)

    if weight_path is not None:
        pass
    else:
        if _local_only_mode():
            raise RuntimeError(
                "Local-only mode is active and `AE_MODEL_PATH` is not set to a valid file. "
                "Place FLUX.2 ae.safetensors in `weights/ae.safetensors` or set AE_MODEL_PATH in Settings > Custom Model Paths."
            )

        ae_filename = config["filename_ae"]
        ae_repo_candidates = [
            config["repo_id"],
            "black-forest-labs/FLUX.2-dev",
        ]

        weight_path = None
        for repo_id in ae_repo_candidates:
            try:
                weight_path = _download_to_repo_weights(
                    repo_id=repo_id,
                    filename=ae_filename,
                    token=hf_token,
                )
                break
            except (huggingface_hub.errors.EntryNotFoundError, huggingface_hub.errors.RepositoryNotFoundError):
                continue
            except huggingface_hub.errors.GatedRepoError:
                continue

        if weight_path is None:
            raise RuntimeError(
                "Failed to access FLUX.2 autoencoder weights. "
                "Remote download to repo/weights failed from model repo / FLUX.2-dev. "
                "Set AE_MODEL_PATH to a local FLUX.2 ae.safetensors, or request access to black-forest-labs/FLUX.2-dev."
            )

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
