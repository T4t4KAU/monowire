"""
Version information for vLLM-Omni.

MonoWire modification: the bundled source uses a fixed version from _version.py.
"""

import warnings

from ._version import __version__, __version_tuple__


def warn_if_misaligned_vllm_version():
    """Warn if vLLM and vllm-omni versions don't match (major.minor)."""
    # Import vllm lazily since import order may be sensitive with current monkeypatching,
    # but we want to check this before potentially breaking imports run.
    from vllm import __version__ as vllm_version
    from vllm import __version_tuple__ as vllm_version_tuple

    omni_ver: tuple[str | int, ...] = __version_tuple__[:2]
    vllm_ver: tuple[str | int, ...] = vllm_version_tuple[:2]
    # Skip if either version is dev (0, 0)
    if omni_ver == (0, 0) or vllm_ver == (0, 0):
        return

    # Compare major.minor
    if omni_ver != vllm_ver:
        warnings.warn(
            "vLLM and vLLM-Omni appear to have mismatched major/minor versions:\n"
            f" --> vLLM-Omni version {__version__}\n"
            f" --> vLLM version {vllm_version}\n"
            "This will likely cause compatibility issues.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["__version__", "__version_tuple__"]

# Run version check automatically when this module is imported
try:
    warn_if_misaligned_vllm_version()
except ModuleNotFoundError:
    # vLLM not installed (e.g., documentation builds)
    pass
