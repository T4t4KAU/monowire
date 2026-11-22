"""Native H3 attention and timestep packing."""

import torch


def validate_block_kwargs(arguments):
    if arguments.get("vdn_window") is not None:
        raise ValueError("H3 pipeline batches require dense attention")


def pack_requests(items):
    """Pack request-local hidden rows and metadata; preserve boundaries for splitting."""
    packet = dict(pack_block_kwargs([item["arguments"] for item in items]))
    rows = [item["hidden"] for item in items]
    packet["hidden"] = torch.cat(rows)
    return packet, [hidden.shape[0] for hidden in rows]


def pack_block_kwargs(arguments):
    """Pack dense H3 block inputs without mixing attention or AdaLN documents."""
    if not arguments:
        raise ValueError("Cannot pack an empty H3 batch")
    for args in arguments:
        validate_block_kwargs(args)
        if args["num_requests"] != 1:
            raise ValueError("Pack request-local block metadata, not an already packed batch")
    if len(arguments) == 1:
        return arguments[0]
    token_offset = timestep_offset = 0
    cumulative, combined = [], []
    for args in arguments:
        # Native single-request metadata contains [0, used, padded].
        # Build only nonempty document boundaries using host lengths; a repeated
        # interior boundary is not accepted by every varlen attention backend.
        bounds = [token_offset]
        if args["max_seqlen"] < args["packed_total"]:
            bounds.append(token_offset + args["max_seqlen"])
        cumulative.append(args["cu_seqlens"].new_tensor(bounds))
        combined.append(args["combined_indices"] + timestep_offset * 3)
        token_offset += args["packed_total"]
        timestep_offset += args["t_emb"].shape[0]
    return {
        "t_emb": torch.cat([args["t_emb"] for args in arguments], dim=0),
        "combined_indices": torch.cat(combined, dim=0),
        "rope_table": torch.cat([args["rope_table"] for args in arguments], dim=0),
        "cu_seqlens": torch.cat(
            [
                *cumulative,
                arguments[0]["cu_seqlens"].new_tensor([token_offset]),
            ]
        ),
        "max_seqlen": max(
            max(args["max_seqlen"], args["packed_total"] - args["max_seqlen"]) for args in arguments
        ),
        "packed_total": token_offset,
        "num_requests": sum(args["num_requests"] for args in arguments),
        "video_layout": None,
        "vsa_prefix_segments": (),
        "vdn_window": None,
    }
