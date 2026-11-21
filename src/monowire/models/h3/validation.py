"""Runtime requirements shared by the H3 inference entry points."""

import os
from importlib.metadata import version


def validate_runtime(pp_size: int) -> None:
    if pp_size < 1:
        raise ValueError("pp_size must be positive")
    allowed = os.environ.get("VLLM_PLUGINS")
    if allowed is not None and "monowire_h3" not in {name.strip() for name in allowed.split(",")}:
        raise ValueError(
            "VLLM_PLUGINS must include monowire_h3 so every worker loads the PP adapter"
        )
    installed = version("vllm")
    if installed.split("+")[0] != "0.32.0":
        raise RuntimeError(f"Expected the pinned vLLM 0.32.0 runtime; found {installed}")
