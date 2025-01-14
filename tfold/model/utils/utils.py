# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 17:44
from typing import Any, List, Callable, Optional, Union

import torch
from torch.utils import checkpoint


def checkpoint_blocks(
        blocks: Union[List[Callable], torch.nn.ModuleList],
        args: Any,
        interval: Optional[int] = 1,
        checkpoint_func=checkpoint.checkpoint,
        return_intervals=False
):
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        interval:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if interval is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif interval < 1 or interval > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    if return_intervals:
        interval_outputs = []
        for s in range(0, len(blocks), interval):
            e = s + interval
            args = checkpoint_func(chunker(s, e), *args)
            interval_outputs.append(args)
            args = wrap(args)
        return interval_outputs

    for s in range(0, len(blocks), interval):
        e = s + interval
        args = checkpoint_func(chunker(s, e), *args)
        args = wrap(args)

    return args
