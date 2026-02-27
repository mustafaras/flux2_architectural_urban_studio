"""
GIF animation generator for inference preview.

Converts intermediate frame sequence to animated GIF, optimized for <2 second
generation time through efficient frame selection and compression.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

from PIL import Image

logger = logging.getLogger("flux2_gif_generator")


class GIFGenerator:
    """
    Generates optimized animated GIFs from frame sequence.

    Features:
    - Fast generation (<2s for 12 frames)
    - Variable frame selection (sample every N frames)
    - Quality/speed optimization
    - Memory-efficient buffering
    """

    def __init__(
        self,
        frame_duration_ms: int = 100,
        loop: int = 0,
        optimize: bool = True,
        quality: int = 85,
    ) -> None:
        """
        Initialize GIF generator.

        Args:
            frame_duration_ms: Duration per frame in milliseconds
            loop: Loop count (0 = infinite)
            optimize: Optimize GIF (slower but smaller)
            quality: Quality 1-95 (higher = better)
        """
        self.frame_duration_ms = max(50, frame_duration_ms)
        self.loop = loop
        self.optimize = optimize
        self.quality = max(1, min(95, quality))

    def generate(
        self,
        frames: list[bytes | Image.Image],
        selected_indices: list[int] | None = None,
    ) -> bytes | None:
        """
        Generate animated GIF from frames.

        Args:
            frames: List of frame data (bytes PNG or PIL Image)
            selected_indices: Indices of frames to include (None = all)

        Returns:
            GIF bytes or None if generation failed
        """
        if not frames:
            logger.warning("No frames provided for GIF generation")
            return None

        try:
            # Select frames
            if selected_indices:
                frames = [frames[i] for i in selected_indices if i < len(frames)]
            else:
                frames = list(frames)

            if len(frames) < 2:
                logger.warning(f"Need at least 2 frames for GIF; got {len(frames)}")
                return None

            # Convert frames to PIL Images
            pil_frames = []
            for i, frame in enumerate(frames):
                try:
                    if isinstance(frame, bytes):
                        img = Image.open(io.BytesIO(frame)).convert("RGB")
                    elif isinstance(frame, Image.Image):
                        img = frame.convert("RGB")
                    else:
                        logger.warning(f"Unsupported frame type at index {i}: {type(frame)}")
                        continue

                    # Ensure reasonable size (resize if needed)
                    if img.size[0] > 512 or img.size[1] > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)

                    pil_frames.append(img)
                except Exception as e:
                    logger.warning(f"Failed to process frame {i}: {e}")
                    continue

            if len(pil_frames) < 2:
                logger.error("Could not convert frames to PIL Images")
                return None

            # Generate GIF
            start = time.perf_counter()
            output = io.BytesIO()
            pil_frames[0].save(
                output,
                format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                duration=self.frame_duration_ms,
                loop=self.loop,
                optimize=self.optimize,
            )
            elapsed = time.perf_counter() - start

            gif_bytes = output.getvalue()
            size_mb = len(gif_bytes) / (1024 ** 2)
            logger.info(
                f"GIF generated: {len(pil_frames)} frames, {size_mb:.2f} MB in {elapsed:.2f}s"
            )

            if elapsed > 2.0:
                logger.warning(f"GIF generation took {elapsed:.2f}s (target: <2s)")

            return gif_bytes
        except Exception as e:
            logger.exception(f"GIF generation failed: {e}")
            return None

    def generate_fast(
        self,
        frames: list[bytes | Image.Image],
        max_frames: int = 8,
    ) -> bytes | None:
        """
        Generate GIF with fast sampling strategy.

        Selects every Nth frame to meet target count, ensuring first and last
        frames are always included.

        Args:
            frames: List of frame data
            max_frames: Maximum frames in output GIF

        Returns:
            GIF bytes or None
        """
        if len(frames) <= max_frames:
            return self.generate(frames)

        # Select evenly-spaced frames, always including first and last
        selected_indices = _select_frames_evenly(len(frames), max_frames)
        return self.generate(frames, selected_indices=selected_indices)


def _select_frames_evenly(total_frames: int, target_count: int) -> list[int]:
    """
    Select evenly-spaced frame indices.

    Always includes first (0) and last (total_frames-1) indices.
    """
    if target_count <= 1:
        return [0]
    if target_count >= total_frames:
        return list(range(total_frames))

    # Create even spacing including first and last
    indices = []
    interval = (total_frames - 1) / (target_count - 1)

    for i in range(target_count):
        idx = int(round(i * interval))
        indices.append(min(idx, total_frames - 1))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(indices))


def get_gif_generator() -> GIFGenerator:
    """Get singleton GIF generator instance."""
    if not hasattr(get_gif_generator, "_instance"):
        get_gif_generator._instance = GIFGenerator(
            frame_duration_ms=100,
            optimize=True,
            quality=85,
        )
    return get_gif_generator._instance
