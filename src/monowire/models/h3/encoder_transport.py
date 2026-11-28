"""Encoder PP transport with subgroup-local stage indices and global P2P peers."""

import torch


def run_encoder_pipeline(
    stage, inputs, lengths, device_group, cpu_group, *, hidden_dim, on_output=None, on_ready=None
):
    """Clocked 1F pipeline: one request per stage, bounded activation handoffs.

    With on_output, the last rank publishes each CPU tensor immediately and each
    rank returns after its final handoff. The caller owns rolling output transport.
    Without it, preserve the barrier baseline and return all outputs on rank 0.
    """
    import torch.distributed as dist

    rank, world = dist.get_rank(device_group), dist.get_world_size(device_group)

    def global_rank(local):
        return dist.get_global_rank(device_group, local)

    def cpu_rank(local):
        return dist.get_global_rank(cpu_group, local)

    count = len(lengths)
    current = None
    outputs = []
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
                    dist.P2POp(dist.isend, tensor, global_rank(rank + 1), device_group)
                    for tensor in outgoing
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
                    dist.P2POp(dist.irecv, tensor, global_rank(rank - 1), device_group)
                    for tensor in incoming
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
                    dist.send(tensor, dst=cpu_rank(0), group=cpu_group)
                outputs = []
            elif rank == 0:
                for length in lengths:
                    tensor = torch.empty((length, hidden_dim), dtype=torch.bfloat16)
                    dist.recv(tensor, src=cpu_rank(world - 1), group=cpu_group)
                    outputs.append(tensor)
        return outputs
    finally:
        stage.text_model.last_positions = None
        torch.backends.cuda.enable_cudnn_sdp(old_cudnn)
