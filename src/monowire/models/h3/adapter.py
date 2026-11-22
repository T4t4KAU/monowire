"""Native H3 request state and stage operations for continuous serving."""

import torch

from ...channels import record_stream
from ...continuous import PreparedStep
from ...tensor_payload import SharedTensor, TensorParts
from .io import CodecWorker
from .packing import pack_requests, validate_block_kwargs


class H3RequestAdapter:
    """Own native request state only at ingress; reuse upstream Euler and VAE."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.io = CodecWorker(pipeline.device)

    async def admit(self, key, value):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native
        from vllm_omni.diffusion.worker.utils import StepRequestState

        prompt, sampling = value
        if (sampling.extra_args or {}).get("task", "t2va") != "t2va":
            raise ValueError("Continuous H3 currently serves the t2va transformer")
        if (sampling.extra_args or {}).get("sampler", "euler") != "euler":
            raise ValueError("Continuous H3 currently supports Euler only")
        if sampling.lora_request is not None:
            raise ValueError("Continuous H3 requires the original weights without request LoRA")
        state = StepRequestState(request_id=key, prompt=prompt, sampling=sampling)
        try:
            state = await self.io.run(self.pipeline.prepare_encode, state)
            # Codec allocations now enter the DiT compute stream. Retain their
            # storage even if a request fails before its queued GPU work ends.
            record_stream(state.latents)
            record_stream(state.timesteps)
            record_stream(state.extra)
            record_stream(vars(state.extra[native._STEP_BRANCH]))
            return state
        except BaseException:
            self.release(state)
            raise

    def prepare(self, state):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        transformer = state.extra[native._STEP_TRANSFORMER]
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
        prepared = prepare_transformer_step(
            transformer,
            continuation,
            state.request_id,
            state.step_index,
            len(state.extra[native._STEP_SIGMAS_VIDEO]) - 1,
        )
        rope = prepared.payload["arguments"]["rope_table"]
        previous = state.extra.get("monowire_rope")
        if previous is None or previous[0] is not rope:
            previous = (rope, SharedTensor.snapshot(rope))
            state.extra["monowire_rope"] = previous
        prepared.payload["shared_rope"] = previous[1]
        return prepared

    def commit(self, state, prepared, output):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        payload = prepared.payload
        transformer = payload["transformer"]
        cache_entry = payload["entry"]
        residual = None
        if cache_entry is not None and not prepared.hit:
            residual = output - payload["hidden"]
        if transformer.adaln_cache is not None:
            transformer.adaln_cache.prepare_from_key(
                payload["arguments"]["t_emb"], payload["adaln_key"]
            )
        try:
            payload["continuation"].send(output)
        except StopIteration as complete:
            velocities = complete.value
        else:
            raise RuntimeError("Unexpected extra transformer yield")
        video_velocity, audio_velocity = velocities
        state.extra[native._STEP_AUDIO_NOISE_PRED] = audio_velocity
        self.pipeline.step_scheduler(state, video_velocity)
        if cache_entry is not None:
            transformer.request_cache.commit(
                cache_entry,
                prepared.step,
                prepared.hit,
                residual,
            )

    def prepare_batch(self, prepared):
        return prepare_transformer_batch(prepared)

    def finished(self, state):
        return state.denoise_completed

    async def finish(self, state):
        output = await self.io.run(self.pipeline.post_decode, state)
        record_stream(output)
        return output

    async def close(self):
        await self.io.close()

    def release(self, state):
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as native

        transformer = state.extra.get(native._STEP_TRANSFORMER)
        if transformer is not None:
            transformer.request_cache.entries.pop(state.request_id, None)
        state.extra.clear()
        state.latents = None


def prepare_transformer_step(transformer, continuation, request_id, step, total_steps):
    """Prepare request-local embeddings and metadata without executing any blocks."""
    if transformer.start_layer != 0:
        raise ValueError("Request preparation requires the first pipeline stage")
    hidden, arguments = next(continuation)
    arguments = dict(arguments)
    validate_block_kwargs(arguments)
    cache = transformer.request_cache
    policy = cache.config.policy
    probe_end = {
        "none": 0,
        "first_block": 1,
        "first_stage": transformer.end_layer,
        "first_stage_periodic": transformer.end_layer,
    }[policy]
    if cache.config.probe_blocks and cache.config.probe_blocks[-1] > probe_end:
        raise ValueError("All probe blocks must belong to the ingress stage")
    # The cached residual starts at this prefix output, including any skipped
    # local blocks for FirstBlockCache, or only downstream blocks for FirstStageCache.
    payload = {
        "hidden": hidden,
        "arguments": arguments,
        "entry": None,
        "next_layer": 0,
        "probe_end": probe_end,
        "total_steps": total_steps,
        "transformer": transformer,
        "continuation": continuation,
        "adaln_key": (
            transformer.adaln_cache.prepared_key(arguments["t_emb"])
            if transformer.adaln_cache is not None
            else None
        ),
    }
    # Different steps can pack: metadata preserves per-request timestep indices,
    # attention documents and positional embeddings. Prefix boundaries must match.
    return PreparedStep(
        request_id,
        step,
        False,
        (id(transformer), hidden.dtype, tuple(hidden.shape[1:]), probe_end),
        payload,
    )


def prepare_transformer_batch(prepared):
    """Execute a packed prefix once, then make independent request cache decisions."""
    if not prepared:
        return []
    first = prepared[0]
    if any(item.compatibility != first.compatibility for item in prepared):
        raise ValueError("Ingress batch contains incompatible requests")
    transformer = first.payload["transformer"]
    probe_end = first.payload["probe_end"]
    if not probe_end:
        return prepared
    cache = transformer.request_cache
    packet, lengths = pack_requests([item.payload for item in prepared])
    block_probes = dict.fromkeys(cache.config.probe_blocks)
    options = {"probe_residuals": block_probes} if block_probes else {}
    output = compute_stage(transformer, packet, end_layer=probe_end, **options)["hidden"]
    probe_chunks = [residual.split(lengths) for residual in block_probes.values()]
    outputs = output.split(lengths)
    probes = []
    for index, (item, hidden) in enumerate(zip(prepared, outputs, strict=True)):
        if probe_chunks:
            # Preserve each selected block's residual for independent comparison.
            probe = torch.stack([chunks[index] for chunks in probe_chunks])
        else:
            probe = hidden - item.payload["hidden"]
        probes.append(probe)
    entries = [cache.entry(item.request_id, item.step) for item in prepared]
    decisions = cache.should_reuse_batch(
        entries,
        probes,
        [item.step for item in prepared],
        [item.payload["total_steps"] for item in prepared],
    )
    for item, hidden, entry, hit in zip(prepared, outputs, entries, decisions, strict=True):
        item.hit = hit
        item.payload.update(hidden=hidden, entry=entry, next_layer=probe_end)
    return prepared


def compute_stage(transformer, packet, *, start_layer=None, end_layer=None, probe_residuals=None):
    """One actor exclusively owns its GPU and shared AdaLN cache."""
    start = transformer.start_layer if start_layer is None else start_layer
    end = transformer.end_layer if end_layer is None else end_layer
    if not transformer.start_layer <= start <= end <= transformer.end_layer:
        raise ValueError("Block range is outside the local pipeline stage")
    if start == end:
        return packet
    arguments = {key: value for key, value in packet.items() if key not in {"hidden", "adaln_key"}}
    rope = arguments.get("rope_table")
    if isinstance(rope, TensorParts):
        arguments["rope_table"] = rope.materialize()
    elif isinstance(rope, SharedTensor):
        arguments["rope_table"] = rope.value
    if transformer.adaln_cache is not None:
        transformer.adaln_cache.prepare_from_key(arguments["t_emb"], packet["adaln_key"])
    hidden = packet["hidden"]
    with torch.profiler.record_function(
        f"monowire/continuous/pp{transformer.pp_group.rank_in_group}/blocks"
    ):
        for index in range(start, end):
            block_input = hidden
            hidden = transformer.blocks[index](hidden, **arguments)
            if probe_residuals is not None and index + 1 in probe_residuals:
                probe_residuals[index + 1] = hidden - block_input
    packet["hidden"] = hidden
    return packet


class H3StageAdapter:
    """Model operations injected into the generic tensor pipeline executor."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.group = transformer.pp_group

    def validate(self, prepared):
        if any(item.payload["transformer"] is not self.transformer for item in prepared):
            raise ValueError("A physical batch cannot mix H3 transformers")
        if prepared and any(
            item.payload["next_layer"] != prepared[0].payload["next_layer"] for item in prepared
        ):
            raise ValueError("A physical batch cannot mix cache prefix boundaries")

    def reuse(self, prepared):
        return [item.payload["hidden"] + item.payload["entry"].residual for item in prepared]

    def pack(self, prepared):
        packet, lengths = pack_requests([item.payload for item in prepared])
        compute_stage(self.transformer, packet, start_layer=prepared[0].payload["next_layer"])
        return packet, lengths

    def compute(self, packet):
        return compute_stage(self.transformer, packet)

    def export_requests(self, prepared):
        """Run the local miss suffix once, then expose independently routable rows."""
        if prepared[0].payload["next_layer"] == self.transformer.end_layer:
            outputs = [item.payload["hidden"] for item in prepared]
        else:
            packet, lengths = self.pack(prepared)
            outputs = packet["hidden"].split(lengths)
        return [
            {
                "hidden": hidden,
                "arguments": {
                    **item.payload["arguments"],
                    "rope_table": item.payload.get(
                        "shared_rope", item.payload["arguments"]["rope_table"]
                    ),
                },
                "adaln_key": item.payload["adaln_key"],
                "compatibility": (hidden.dtype, tuple(hidden.shape[1:])),
            }
            for item, hidden in zip(prepared, outputs, strict=True)
        ]

    def combine_requests(self, items):
        packet, _ = pack_requests(items)
        return packet
