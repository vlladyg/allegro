"""nequip.data.jit: TorchScript functions for dealing with AtomicData.

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Computing spin distance for nearest neighbors
Authors: Vladimir ladygin
"""

from typing import Dict, Any

import torch
import torch.jit

from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys
from nequip.data import AtomicDataDict


# Define a type alias
Type = Dict[str, torch.Tensor]



@torch.jit.script
def with_edge_spin_length(data: Type, with_distance: bool = True) -> Type:
    """Compute the edge distance vectors between the spins for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 1] edge distance vectors
        or 
        Tensor [n_nodes, 1] edge nodes if with distance = False
    """
    # Build node spin norms
    if not _keys.NODE_SPIN_LENGTH in data:
        data[_keys.NODE_SPIN_LENGTH] = torch.linalg.norm(data[_keys.NODE_SPIN], dim=-1)

    # Build spin distance
    if with_distance and _keys.EDGE_SPIN_DISTANCE not in data:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        spin = data[_keys.NODE_SPIN]

        data[_keys.EDGE_SPIN_DISTANCE] = torch.einsum("ki,kj->k", spin[edge_index[1]], spin[edge_index[0]])
        
    return data