"""Physics helpers for synthesis and reconstruction."""

from .airlight import airlight_tensor_to_broadcast, airlight_to_tensor, sample_airlight
from .degradations import apply_glare, apply_noise
from .depth_proxy import build_depth_estimator, estimate_depth_proxy, load_precomputed_depth_map, normalize_depth
from .transmission import transmission_from_depth

__all__ = [
    "airlight_to_tensor",
    "airlight_tensor_to_broadcast",
    "apply_glare",
    "apply_noise",
    "build_depth_estimator",
    "estimate_depth_proxy",
    "load_precomputed_depth_map",
    "normalize_depth",
    "sample_airlight",
    "transmission_from_depth",
]
