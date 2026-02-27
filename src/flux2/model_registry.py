"""
Model Registry System for FLUX.2

Manages multiple model versions, integrity verification, and health monitoring.
Supports custom models, LoRA fine-tuning, and A/B testing capabilities.
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model verification and health status"""
    VERIFIED = "verified"
    PENDING = "pending"
    CORRUPTED = "corrupted"
    DOWNLOADING = "downloading"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Model architecture types"""
    FLUX_KLEIN = "flux_klein"
    FLUX_STANDARD = "flux_standard"
    TEXT_ENCODER = "text_encoder"
    AUTOENCODER = "autoencoder"
    LORA = "lora"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    name: str
    version: str
    status: ModelStatus
    model_type: ModelType
    file_path: Optional[Path] = None
    file_size_mb: float = 0.0
    sha256_hash: Optional[str] = None
    performance_score: float = 0.0
    last_updated: str = ""
    last_verified: str = ""
    download_url: Optional[str] = None
    parameter_count: Optional[str] = None
    quantization: Optional[str] = None  # e.g., "bf16", "int8", "int4"
    vram_requirement_gb: float = 0.0
    avg_inference_time_s: float = 0.0
    description: str = ""
    tags: List[str] = None
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}
        if isinstance(self.status, str):
            self.status = ModelStatus(self.status)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        if self.file_path and isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['model_type'] = self.model_type.value
        data['file_path'] = str(self.file_path) if self.file_path else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)


class ModelRegistry:
    """
    Central registry for managing FLUX.2 models and variants.
    
    Features:
    - Multi-version model support
    - Integrity verification (SHA256)
    - Health monitoring and benchmarking
    - Custom/LoRA model support
    - A/B testing capabilities
    """
    
    def __init__(self, weights_dir: Path = None, cache_dir: Path = None):
        self.weights_dir = weights_dir or Path("weights")
        self.cache_dir = cache_dir or Path("weights/imported_cache")
        self.registry: Dict[str, ModelMetadata] = {}
        self.registry_file = self.weights_dir / "model_registry.json"
        
        # Initialize with default models
        self._initialize_default_models()
        
        # Load persisted registry
        self._load_registry()
        
        # Verify existing models
        self._verify_all_models()
    
    def _initialize_default_models(self):
        """Initialize registry with known FLUX.2 models"""
        defaults = {
            "flux.2-klein-4b": ModelMetadata(
                name="FLUX.2 Klein 4B",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_KLEIN,
                file_path=self.weights_dir / "flux-2-klein-4b.safetensors",
                parameter_count="4B",
                quantization="bf16",
                vram_requirement_gb=8.5,
                avg_inference_time_s=3.2,
                performance_score=9.2,
                description="Compact 4B parameter model optimized for speed",
                tags=["fast", "compact", "recommended"],
                last_updated="2026-02-15"
            ),
            "flux.2-klein-base-4b": ModelMetadata(
                name="FLUX.2 Klein Base 4B",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_KLEIN,
                file_path=self.weights_dir / "flux-2-klein-base-4b.safetensors",
                parameter_count="4B",
                quantization="bf16",
                vram_requirement_gb=8.5,
                avg_inference_time_s=3.5,
                performance_score=8.8,
                description="Base Klein model without distillation",
                tags=["base", "compact"],
                last_updated="2026-02-10"
            ),
            "flux.2-klein-9b": ModelMetadata(
                name="FLUX.2 Klein 9B",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_KLEIN,
                file_path=self.weights_dir / "flux-2-klein-9b.safetensors",
                parameter_count="9B",
                quantization="bf16",
                vram_requirement_gb=16.0,
                avg_inference_time_s=4.0,
                performance_score=9.5,
                description="Higher-capacity distilled Klein model balancing speed and quality",
                tags=["fast", "high-detail"],
                last_updated="2026-02-15"
            ),
            "flux.2-klein-base-9b": ModelMetadata(
                name="FLUX.2 Klein Base 9B",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_KLEIN,
                file_path=self.weights_dir / "flux-2-klein-base-9b.safetensors",
                parameter_count="9B",
                quantization="bf16",
                vram_requirement_gb=16.0,
                avg_inference_time_s=4.8,
                performance_score=9.6,
                description="Non-distilled 9B Klein model for highest Klein quality",
                tags=["base", "high-quality"],
                last_updated="2026-02-15"
            ),
            "flux.2-dev": ModelMetadata(
                name="FLUX.2 Dev",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_STANDARD,
                file_path=self.weights_dir / "flux2-dev.safetensors",
                parameter_count="32B",
                quantization="bf16",
                vram_requirement_gb=20.0,
                avg_inference_time_s=9.0,
                performance_score=9.9,
                description="Full FLUX.2 development model optimized for top quality",
                tags=["high-quality", "slow", "large"],
                last_updated="2025-11-25"
            ),
            "flux.1-dev": ModelMetadata(
                name="FLUX.1 Dev",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_STANDARD,
                file_path=self.cache_dir / "models--black-forest-labs--FLUX.1-dev",
                parameter_count="12B",
                quantization="bf16",
                vram_requirement_gb=24.0,
                avg_inference_time_s=8.5,
                performance_score=9.8,
                description="High-quality FLUX.1 Development model",
                tags=["high-quality", "slow", "large"],
                last_updated="2025-08-01"
            ),
            "flux.1-schnell": ModelMetadata(
                name="FLUX.1 Schnell",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.FLUX_STANDARD,
                file_path=self.cache_dir / "models--black-forest-labs--FLUX.1-schnell",
                parameter_count="12B",
                quantization="bf16",
                vram_requirement_gb=24.0,
                avg_inference_time_s=2.8,
                performance_score=8.5,
                description="Fast inference variant (4 steps)",
                tags=["fast", "large", "distilled"],
                last_updated="2025-08-01"
            ),
            "autoencoder": ModelMetadata(
                name="FLUX AutoEncoder VAE",
                version="latest",
                status=ModelStatus.PENDING,
                model_type=ModelType.AUTOENCODER,
                file_path=self.weights_dir / "ae.safetensors",
                parameter_count="<1B",
                vram_requirement_gb=2.0,
                performance_score=9.5,
                description="Latent space encoder/decoder",
                tags=["vae", "required"],
                last_updated="2025-08-01"
            ),
        }
        
        for key, metadata in defaults.items():
            if key not in self.registry:
                self.registry[key] = metadata
    
    def register_model(self, key: str, metadata: ModelMetadata) -> bool:
        """
        Register a new model or update existing.
        
        Args:
            key: Unique model identifier
            metadata: Model metadata
            
        Returns:
            True if successfully registered
        """
        try:
            self.registry[key] = metadata
            self._save_registry()
            logger.info(f"Registered model: {key} ({metadata.name})")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {key}: {e}")
            return False
    
    def get_model(self, name: str, version: str = "latest") -> Optional[ModelMetadata]:
        """
        Retrieve specific model version.
        
        Args:
            name: Model identifier
            version: Version string (default: "latest")
            
        Returns:
            ModelMetadata if found, None otherwise
        """
        # Direct lookup
        if name in self.registry:
            model = self.registry[name]
            if version == "latest" or model.version == version:
                return model
        
        # Search by name field
        for key, model in self.registry.items():
            if model.name == name and (version == "latest" or model.version == version):
                return model
        
        logger.warning(f"Model not found: {name} (version: {version})")
        return None
    
    def list_available(self, model_type: Optional[ModelType] = None, 
                      status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """
        List all available models with optional filtering.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            
        Returns:
            List of model metadata
        """
        models = list(self.registry.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by performance score descending
        models.sort(key=lambda m: m.performance_score, reverse=True)
        
        return models
    
    def verify_integrity(self, model_path: Path, expected_hash: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify model file integrity using SHA256.
        
        Args:
            model_path: Path to model file
            expected_hash: Expected SHA256 hash (None to just compute)
            
        Returns:
            (is_valid, computed_hash)
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False, None
        
        try:
            sha256 = hashlib.sha256()
            
            # Read in chunks for large files
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192 * 1024), b""):
                    sha256.update(chunk)
            
            computed_hash = sha256.hexdigest()
            
            if expected_hash:
                is_valid = computed_hash == expected_hash
                if not is_valid:
                    logger.error(f"Hash mismatch for {model_path.name}")
                    logger.error(f"  Expected: {expected_hash}")
                    logger.error(f"  Got:      {computed_hash}")
                return is_valid, computed_hash
            
            return True, computed_hash
            
        except Exception as e:
            logger.error(f"Error verifying {model_path}: {e}")
            return False, None
    
    def check_health(self, model_key: str) -> Dict[str, Any]:
        """
        Perform comprehensive health check on model.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Health check results dictionary
        """
        model = self.registry.get(model_key)
        if not model:
            return {"status": "not_found", "healthy": False}
        
        results = {
            "model_key": model_key,
            "name": model.name,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "healthy": True
        }
        
        # Check 1: File exists
        file_exists = model.file_path and model.file_path.exists()
        results["checks"]["file_exists"] = file_exists
        if not file_exists:
            results["healthy"] = False
            results["checks"]["file_exists_message"] = f"File not found: {model.file_path}"
        
        # Check 2: File size reasonable
        if file_exists:
            file_size_mb = model.file_path.stat().st_size / (1024 * 1024)
            results["checks"]["file_size_mb"] = round(file_size_mb, 2)
            
            # Sanity check: only large built-in models should be > 100MB
            if file_size_mb < 100 and model.model_type not in {ModelType.LORA, ModelType.CUSTOM}:
                results["healthy"] = False
                results["checks"]["file_size_message"] = f"Suspiciously small: {file_size_mb:.1f}MB"
        
        # Check 3: Hash verification (if stored)
        if file_exists and model.sha256_hash:
            is_valid, _ = self.verify_integrity(model.file_path, model.sha256_hash)
            results["checks"]["integrity_valid"] = is_valid
            if not is_valid:
                results["healthy"] = False
                results["checks"]["integrity_message"] = "SHA256 mismatch detected"
        
        # Check 4: Last verification age
        if model.last_verified:
            try:
                last_check = datetime.fromisoformat(model.last_verified)
                age_days = (datetime.now() - last_check).days
                results["checks"]["last_verified_days_ago"] = age_days
                
                if age_days > 30:
                    results["checks"]["verification_message"] = f"Not verified in {age_days} days"
            except Exception:
                pass
        
        # Update status based on health
        if results["healthy"]:
            model.status = ModelStatus.VERIFIED
        else:
            model.status = ModelStatus.CORRUPTED
        
        model.last_verified = datetime.now().isoformat()
        self._save_registry()
        
        return results
    
    def _verify_all_models(self):
        """Run health checks on all registered models"""
        logger.info("Verifying all models in registry...")
        
        for key in list(self.registry.keys()):
            health = self.check_health(key)
            status = "✓" if health["healthy"] else "✗"
            logger.info(f"  {status} {key}: {health['checks']}")
    
    def record_inference_time(self, model_key: str, inference_time_s: float, vram_used_mb: float) -> bool:
        """
        Record inference time and VRAM usage for performance tracking.
        
        Args:
            model_key: Model identifier
            inference_time_s: Inference duration in seconds
            vram_used_mb: VRAM used during inference in MB
            
        Returns:
            True if recorded successfully
        """
        model = self.registry.get(model_key)
        if not model:
            logger.warning(f"Model not found for timing record: {model_key}")
            return False
        
        # Update average inference time
        if model.avg_inference_time_s == 0:
            model.avg_inference_time_s = inference_time_s
        else:
            # Running average (simple approach)
            model.avg_inference_time_s = (model.avg_inference_time_s + inference_time_s) / 2
        
        # Update last used timestamp
        model.last_updated = datetime.now().isoformat()
        
        self._save_registry()
        return True
    
    def benchmark_model(self, model_key: str, test_prompt: str = "test image") -> Dict[str, float]:
        """
        Benchmark model performance (placeholder - actual implementation needs inference).
        
        Args:
            model_key: Model identifier
            test_prompt: Test prompt for generation
            
        Returns:
            Performance metrics
        """
        model = self.registry.get(model_key)
        if not model:
            return {"error": "Model not found"}
        
        # This would integrate with actual inference in production
        # For now, return stored metrics
        return {
            "avg_inference_time_s": model.avg_inference_time_s,
            "performance_score": model.performance_score,
            "vram_requirement_gb": model.vram_requirement_gb
        }
    
    def add_custom_model(self, 
                        name: str,
                        file_path: Path,
                        model_type: ModelType = ModelType.CUSTOM,
                        **kwargs) -> Optional[str]:
        """
        Register a custom or LoRA model.
        
        Args:
            name: Human-readable model name
            file_path: Path to model weights
            model_type: Model type (default: CUSTOM)
            **kwargs: Additional metadata fields
            
        Returns:
            Model key if successful, None otherwise
        """
        if not file_path.exists():
            logger.error(f"Custom model file not found: {file_path}")
            return None
        
        # Generate unique key
        key = name.lower().replace(" ", "-")
        counter = 1
        original_key = key
        while key in self.registry:
            key = f"{original_key}-{counter}"
            counter += 1
        
        # Compute file hash
        is_valid, file_hash = self.verify_integrity(file_path)
        if not is_valid:
            logger.error(f"Failed to compute hash for {file_path}")
            return None
        
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=kwargs.get('version', '1.0'),
            status=ModelStatus.VERIFIED,
            model_type=model_type,
            file_path=file_path,
            file_size_mb=round(file_size_mb, 2),
            sha256_hash=file_hash,
            last_updated=datetime.now().isoformat(),
            last_verified=datetime.now().isoformat(),
            tags=kwargs.get('tags', ['custom']),
            description=kwargs.get('description', 'Custom uploaded model'),
            **{k: v for k, v in kwargs.items() if k not in ['version', 'tags', 'description']}
        )
        
        self.register_model(key, metadata)
        logger.info(f"Custom model registered: {key} ({name})")
        
        return key
    
    def remove_model(self, model_key: str, delete_file: bool = False) -> bool:
        """
        Remove model from registry.
        
        Args:
            model_key: Model identifier
            delete_file: Also delete model file from disk
            
        Returns:
            True if successful
        """
        if model_key not in self.registry:
            logger.warning(f"Model not found: {model_key}")
            return False
        
        model = self.registry[model_key]
        
        if delete_file and model.file_path and model.file_path.exists():
            try:
                model.file_path.unlink()
                logger.info(f"Deleted model file: {model.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {model.file_path}: {e}")
                return False
        
        del self.registry[model_key]
        self._save_registry()
        logger.info(f"Removed model from registry: {model_key}")
        
        return True
    
    def compare_models(self, model_a_key: str, model_b_key: str) -> Dict[str, Any]:
        """
        Compare two models for A/B testing.
        
        Args:
            model_a_key: First model identifier
            model_b_key: Second model identifier
            
        Returns:
            Comparison metrics
        """
        model_a = self.registry.get(model_a_key)
        model_b = self.registry.get(model_b_key)
        
        if not model_a or not model_b:
            return {"error": "One or both models not found"}
        
        comparison = {
            "model_a": {
                "key": model_a_key,
                "name": model_a.name,
                "performance_score": model_a.performance_score,
                "inference_time_s": model_a.avg_inference_time_s,
                "vram_gb": model_a.vram_requirement_gb,
                "parameters": model_a.parameter_count
            },
            "model_b": {
                "key": model_b_key,
                "name": model_b.name,
                "performance_score": model_b.performance_score,
                "inference_time_s": model_b.avg_inference_time_s,
                "vram_gb": model_b.vram_requirement_gb,
                "parameters": model_b.parameter_count
            },
            "differences": {
                "performance_delta": model_b.performance_score - model_a.performance_score,
                "speed_ratio": model_a.avg_inference_time_s / model_b.avg_inference_time_s if model_b.avg_inference_time_s > 0 else 0,
                "vram_delta_gb": model_b.vram_requirement_gb - model_a.vram_requirement_gb
            }
        }
        
        return comparison
    
    def _save_registry(self):
        """Persist registry to JSON file"""
        try:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            
            registry_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "models": {
                    key: metadata.to_dict()
                    for key, metadata in self.registry.items()
                }
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Registry saved: {self.registry_file}")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_registry(self):
        """Load registry from JSON file"""
        if not self.registry_file.exists():
            logger.info("No existing registry found, using defaults")
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            for key, model_dict in data.get('models', {}).items():
                metadata = ModelMetadata.from_dict(model_dict)
                self.registry[key] = metadata
            
            logger.info(f"Loaded {len(self.registry)} models from registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = list(self.registry.values())
        
        stats = {
            "total_models": len(models),
            "by_status": {},
            "by_type": {},
            "total_size_gb": 0.0,
            "avg_performance_score": 0.0,
            "verified_models": 0
        }
        
        for model in models:
            # Count by status
            status_key = model.status.value
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1
            
            # Count by type
            type_key = model.model_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1
            
            # Accumulate size
            stats["total_size_gb"] += model.file_size_mb / 1024
            
            # Performance
            if model.performance_score > 0:
                stats["avg_performance_score"] += model.performance_score
            
            if model.status == ModelStatus.VERIFIED:
                stats["verified_models"] += 1
        
        if len(models) > 0:
            stats["avg_performance_score"] /= len(models)
        
        stats["total_size_gb"] = round(stats["total_size_gb"], 2)
        stats["avg_performance_score"] = round(stats["avg_performance_score"], 2)
        
        return stats


# Global registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry(weights_dir: Path = None) -> ModelRegistry:
    """Get or create global model registry singleton"""
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = ModelRegistry(weights_dir=weights_dir)
    
    return _registry_instance
