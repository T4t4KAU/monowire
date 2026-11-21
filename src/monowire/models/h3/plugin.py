"""Opt-in registration preserving upstream H3's modality and serving metadata."""

import os


def register():
    if os.environ.get("MONOWIRE_H3_PP") != "1":
        return
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_diffusion_model(
        model_arch="MiniMaxH3Pipeline",
        module_name="monowire.models.h3.pipeline",
        class_name="MonoWireH3Pipeline",
        post_process_func_name="get_minimax_h3_post_process_func",
    )
