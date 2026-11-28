"""MiniMax H3 inference using vLLM-Omni Diffusion."""

from .cache import CacheConfig
from .continuous import ContinuousConfig
from .continuous_engine import ContinuousEngine
from .device_memory import DeviceMemoryLimits
from .memory import MemoryLimits
from .models.h3.phased import create_phased_engine
from .models.h3.runtime import create_continuous_engine
from .topology import PipelineTopology

__all__ = [
    "CacheConfig",
    "ContinuousConfig",
    "ContinuousEngine",
    "MemoryLimits",
    "DeviceMemoryLimits",
    "PipelineTopology",
    "create_continuous_engine",
    "create_phased_engine",
]
