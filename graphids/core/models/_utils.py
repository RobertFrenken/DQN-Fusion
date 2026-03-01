"""Shared utilities for model forward passes."""
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def checkpoint_conv(conv, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
    """Run a graph conv layer through gradient checkpointing.

    Uses default-arg capture to avoid the stale-closure bug in loops.
    """
    if edge_attr is not None:
        return checkpoint(lambda xi, c=conv, ei=edge_index, ea=edge_attr: c(xi, ei, ea), x, use_reentrant=False)
    return checkpoint(lambda xi, c=conv, ei=edge_index: c(xi, ei), x, use_reentrant=False)
