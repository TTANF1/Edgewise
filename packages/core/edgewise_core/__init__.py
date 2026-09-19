"""Edgewise Layer 1 — deterministic graphics.

Pure numpy/scipy algorithms only. No network, no models, no randomness:
the same input frame always produces the same masks, candidates and
blends. This is the layer that later ports to WASM/WebGPU unchanged.
"""

from edgewise_core import edge, mask, pixel, segmentation

__all__ = ["edge", "mask", "pixel", "segmentation"]
