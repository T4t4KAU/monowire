"""H3's native Qwen mathematics with layer PP, without TP collectives."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from vllm_omni.diffusion.models.minimax_h3 import encoder as native

from .phase_utils import encoder_partition


class QwenPPTextModel(native.MiniMaxH3Qwen3VLTextModel):
    def __init__(self, config, start, end):
        nn.Module.__init__(self)
        self.start, self.end = start, end
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        if start == 0:
            self.embed_tokens = native.MiniMaxH3Qwen3VLVocabParallelEmbedding(
                group, config.vocab_size, config.hidden_size, dtype=torch.bfloat16
            )
        self.rotary_emb = native.MiniMaxH3Qwen3VLTextRotaryEmbedding(config)
        self.layers = nn.ModuleDict(
            {
                str(index): native.MiniMaxH3Qwen3VLTextDecoderLayer(
                    group,
                    config,
                    torch.bfloat16,
                    prefix=f"text_encoder.text_model.layers.{index}",
                )
                for index in range(start, end)
            }
        )
        self.last_positions = None

    def forward(
        self,
        inputs_embeds,
        positions,
        *,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        rotary_inv_freq=None,
    ):
        # Native _encode constructs multimodal positions before entering this method.
        self.last_positions = positions
        hidden = inputs_embeds
        rotary = self.rotary_emb(hidden, positions, inv_freq=rotary_inv_freq)
        for key, layer in self.layers.items():
            hidden = layer(hidden, position_embeddings=rotary)
            index = int(key)
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                hidden = self._deepstack_process(
                    hidden, visual_pos_masks, deepstack_visual_embeds[index]
                )
        # H3 consumes the unnormalized output of layer 49. No final norm or LM head.
        return hidden


class QwenPPEncoder(native.MiniMaxH3Qwen3VLEncoder):
    def __init__(self, config, stage_layers, rank, device):
        nn.Module.__init__(self)
        layers = encoder_partition(
            len(stage_layers),
            stage_layers,
            deepstack=len(config.vision_config.deepstack_visual_indexes),
        )
        if not 0 <= rank < len(layers):
            raise ValueError("Invalid encoder pipeline rank")
        self.start = sum(layers[:rank])
        self.end = self.start + layers[rank]
        self.device_target = torch.device(device)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self._tp_size = 1
        self.quant_config = None
        if self.start == 0:
            self.vision = native.MiniMaxH3Qwen3VLVisionModel(config.vision_config)
        self.text_model = QwenPPTextModel(config.text_config, self.start, self.end)
        self.to(dtype=torch.bfloat16)

    def _map_weight_name(self, name):
        mapped = super()._map_weight_name(name)
        if mapped is None:
            return None
        target = mapped[0]
        if target.startswith("text_model.layers."):
            index = int(target.split(".")[2])
            return mapped if self.start <= index < self.end else None
        return mapped if self.start == 0 else None

    def load_checkpoint(self, path):
        from safetensors import safe_open

        files = sorted(Path(path).glob("*.safetensors"))
        if not files:
            raise ValueError(f"No Qwen safetensors checkpoint at {path}")

        def weights():
            for filename in files:
                with safe_open(filename, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        # Filter before materializing: never allocate other ranks' layers.
                        if self._map_weight_name(key) is not None:
                            yield key, handle.get_tensor(key)

        self.load_weights(weights())
        self.to(self.device_target)
        self.eval()

    def first(self, ids, vision_kwargs):
        if self.start != 0:
            raise RuntimeError("Only stage 0 owns vision and token embeddings")
        hidden = self._encode(ids, **vision_kwargs)
        return hidden.contiguous(), self.text_model.last_positions.contiguous()

    def continuation(self, hidden, positions):
        return self.text_model(hidden.unsqueeze(0), positions)[0].contiguous()


def run_encoder_pipeline(
    stage, inputs, lengths, device_group, cpu_group, *, on_output=None, on_ready=None
):
    """Clocked 1F pipeline: one request per stage, bounded activation handoffs.

    With on_output, the last rank publishes each CPU tensor immediately and each
    rank returns after its final handoff. The caller owns rolling output transport.
    Without it, preserve the barrier baseline and return all outputs on rank 0.
    """
    import torch.distributed as dist

    rank, world = dist.get_rank(device_group), dist.get_world_size(device_group)
    count = len(lengths)
    current = None
    outputs = []
    hidden_dim = native.MINIMAX_H3_QWEN3VL_HIDDEN_DIM
    # Initialize the communicator before a subset enters batched P2P.
    dist.barrier(group=device_group)
    if on_ready is not None:
        on_ready()
    old_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        for tick in range(count + (rank if on_output is not None else world - 1)):
            index = tick - rank
            outgoing = None
            if 0 <= index < count:
                if rank == 0:
                    ids, kwargs = inputs[index]
                    outgoing = stage.first(ids, kwargs)
                else:
                    hidden, positions = current
                    outgoing = (stage.continuation(hidden, positions), positions)
                if rank == world - 1:
                    tensor = outgoing[0].cpu()
                    if on_output is None:
                        outputs.append(tensor)
                    else:
                        on_output(index, tensor)
                    del tensor
            incoming = None
            incoming_index = tick - rank + 1
            operations = []
            if outgoing is not None and rank < world - 1:
                operations.extend(
                    dist.P2POp(dist.isend, tensor, rank + 1, device_group) for tensor in outgoing
                )
            if rank > 0 and 0 <= incoming_index < count:
                length = lengths[incoming_index]
                incoming = (
                    torch.empty(
                        (length, hidden_dim), dtype=torch.bfloat16, device=stage.device_target
                    ),
                    torch.empty((3, 1, length), dtype=torch.long, device=stage.device_target),
                )
                operations.extend(
                    dist.P2POp(dist.irecv, tensor, rank - 1, device_group) for tensor in incoming
                )
            if operations:
                for work in dist.batch_isend_irecv(operations):
                    work.wait()
            current = incoming
        if on_output is not None:
            return []
        if world > 1:
            if rank == world - 1:
                for tensor in outputs:
                    dist.send(tensor, dst=0, group=cpu_group)
                outputs = []
            elif rank == 0:
                for length in lengths:
                    tensor = torch.empty((length, hidden_dim), dtype=torch.bfloat16)
                    dist.recv(tensor, src=world - 1, group=cpu_group)
                    outputs.append(tensor)
        return outputs
    finally:
        stage.text_model.last_positions = None
        torch.backends.cuda.enable_cudnn_sdp(old_cudnn)
