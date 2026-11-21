"""Native H3 request state and stage operations for continuous serving."""

from contextlib import contextmanager

import torch

from ...continuous import PreparedStep
from .packing import pack_block_kwargs, validate_block_kwargs


class H3RequestAdapter:
    """Own native request state only at ingress; reuse upstream Euler and VAE."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @contextmanager
    def _local_io(self):
        # The pinned native pipeline normally distributes encoder inputs across
        # its DiT world. Actor workers only own blocks and never enter that
        # collective. Scope request I/O to its single owner, without changing
        # the PP group used by the actor transport. All adapter calls are on
        # one event-loop thread; other ranks are separate OS processes.
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        if self.pipeline.text_encoder_tp_size != 1:
            raise ValueError("Continuous ingress requires a single-rank text encoder")
        original = native._dit_rank_world
        native._dit_rank_world = lambda: (None, 0, 1)
        try:
            yield
        finally:
            native._dit_rank_world = original

    def admit(self, key, value):
        from vllm_omni.diffusion.worker.utils import StepRequestState

        prompt, sampling = value
        if (sampling.extra_args or {}).get("task", "t2va") != "t2va":
            raise ValueError("Continuous H3 currently serves the t2va transformer")
        if (sampling.extra_args or {}).get("sampler", "euler") != "euler":
            raise ValueError("Continuous H3 currently supports Euler only")
        state = StepRequestState(request_id=key, prompt=prompt, sampling=sampling)
        with self._local_io():
            return self.pipeline.prepare_encode(state)

    def prepare(self, state):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        transformer = state.extra[native._STEP_TRANSFORMER]
        if transformer.request_cache.config.policy not in ("none", "first_block"):
            raise ValueError("Continuous H3 supports none or first_block caching")
        # Stage actors execute blocks outside the normal module offload hooks.
        transformer.to(self.pipeline.device)
        branch = state.extra[native._STEP_BRANCH]
        schedule = native._minimax_h3_step_schedule(state)
        video_rows, video_times = native.minimax_h3_prepare_edit_rows(
            state.latents,
            branch.update_mask_dev,
            state.extra.get(native._STEP_VIDEO_EDIT),
            schedule["t_video"],
            schedule["imgvid_cond_timestep"],
            sigma=schedule["sigma_video"],
        )
        audio_rows, audio_times = native.minimax_h3_prepare_edit_rows(
            state.extra[native._STEP_AUDIO_ROWS],
            branch.audio_update_mask_dev,
            state.extra.get(native._STEP_AUDIO_EDIT),
            schedule["t_audio"],
            schedule["audio_ref_cond_timestep"],
            sigma=schedule["sigma_audio"],
        )
        inputs = branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            t_video=schedule["t_video"],
            t_audio=schedule["t_audio"],
            imgvid_cond_timestep=schedule["imgvid_cond_timestep"],
            audio_ref_cond_timestep=schedule["audio_ref_cond_timestep"],
            video_target_timesteps=video_times,
            audio_target_timesteps=audio_times,
        )
        continuation = transformer.iter_transformer_forward(**inputs)
        hidden, arguments = next(continuation)
        arguments = dict(arguments)
        validate_block_kwargs(arguments)
        cache = transformer.request_cache
        enabled = cache.config.policy == "first_block"
        entry = cache.entry(state.request_id, state.step_index) if enabled else None
        hit = False
        if enabled:
            if transformer.adaln_cache is not None:
                transformer.adaln_cache.prepare(arguments["t_emb"])
            first = transformer.blocks[0](hidden, **arguments)
            hit = cache.should_reuse(
                entry,
                first - hidden,
                state.step_index,
                len(state.extra[native._STEP_SIGMAS_VIDEO]) - 1,
            )
            hidden = first
        payload = dict(
            hidden=hidden,
            arguments=arguments,
            entry=entry,
            first_done=enabled,
            transformer=transformer,
            continuation=continuation,
        )
        # Different steps can pack: pack_block_kwargs preserves per-request
        # timestep indices, attention documents and positional embeddings.
        return PreparedStep(
            state.request_id,
            state.step_index,
            hit,
            (id(transformer), hidden.dtype, tuple(hidden.shape[1:]), enabled),
            payload,
        )

    def commit(self, state, prepared, output):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        p = prepared.payload
        transformer = p["transformer"]
        transformer.to(self.pipeline.device)
        residual = output - p["hidden"] if p["entry"] is not None and not prepared.hit else None
        if transformer.adaln_cache is not None:
            transformer.adaln_cache.prepare(p["arguments"]["t_emb"])
        try:
            p["continuation"].send(output)
        except StopIteration as complete:
            velocities = complete.value
        else:
            raise RuntimeError("Unexpected extra transformer yield")
        video_velocity, audio_velocity = velocities
        state.extra[native._STEP_AUDIO_NOISE_PRED] = audio_velocity
        self.pipeline.step_scheduler(state, video_velocity)
        if p["entry"] is not None:
            transformer.request_cache.commit(
                p["entry"],
                prepared.step,
                prepared.hit,
                residual,
            )

    def finished(self, state):
        return state.denoise_completed

    def finish(self, state):
        with self._local_io():
            return self.pipeline.post_decode(state)

    def release(self, state):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        transformer = state.extra.get(native._STEP_TRANSFORMER)
        if transformer is not None:
            transformer.request_cache.entries.pop(state.request_id, None)
        state.extra.clear()
        state.latents = None


def compute_stage(transformer, packet, *, first_done=False):
    """One actor exclusively owns its GPU and shared AdaLN cache."""
    transformer.to(packet["hidden"].device)
    args = {key: value for key, value in packet.items() if key not in ("hidden", "ticket")}
    if transformer.adaln_cache is not None:
        transformer.adaln_cache.prepare(args["t_emb"])
    hidden = packet["hidden"]
    with torch.profiler.record_function(
        f"monowire/continuous/pp{transformer.pp_group.rank_in_group}/blocks"
    ):
        for index in range(transformer.start_layer + int(first_done), transformer.end_layer):
            hidden = transformer.blocks[index](hidden, **args)
    packet["hidden"] = hidden
    return packet


class H3StageAdapter:
    """Model operations injected into the generic tensor pipeline executor."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.group = transformer.pp_group

    def validate(self, prepared):
        if any(p.payload["transformer"] is not self.transformer for p in prepared):
            raise ValueError("A physical batch cannot mix H3 transformers")

    def reuse(self, prepared):
        return [p.payload["hidden"] + p.payload["entry"].residual for p in prepared]

    def pack(self, prepared):
        lengths = [p.payload["hidden"].shape[0] for p in prepared]
        packet = pack_block_kwargs([p.payload["arguments"] for p in prepared])
        packet = dict(packet)
        packet["hidden"] = torch.cat([p.payload["hidden"] for p in prepared])
        compute_stage(self.transformer, packet, first_done=prepared[0].payload["first_done"])
        return packet, lengths

    def compute(self, packet):
        return compute_stage(self.transformer, packet)

    def unpack(self, output, context):
        return list(output.split(context))
