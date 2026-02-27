"""FLUX.2 image generation and inference library."""

from __future__ import annotations

import sys

__version__ = "0.1.0"
__author__ = "Black Forest Labs"

sys.modules.setdefault("flux2", sys.modules[__name__])

# Make key modules available at package level for convenience
try:
    from .model import Flux2Model
    from .sampling import sample
    from .text_encoder import TextEncoder
    from .autoencoder import AutoEncoder
except ImportError:
    # Allow partial imports if some modules are not yet initialized
    pass

__all__ = [
    "Flux2Model",
    "sample",
    "TextEncoder",
    "AutoEncoder",
]
