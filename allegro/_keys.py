"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FEATURES: Final[str] = "edge_features"

PER_ATOM_SPIN_KEY: Final[str] = "atomic_spin"

NODE_SPIN: Final[str] = "node_spin"
NODE_SPIN_LENGTH: Final[str] = "node_spin_length"
EDGE_SPIN: Final[str] = "edge_spin"
EDGE_SPIN_DISTANCE: Final[str] = "edge_spin_distance"
    

register_fields(node_fields=[NODE_SPIN, NODE_SPIN_LENGTH])
register_fields(edge_fields=[EDGE_ENERGY, EDGE_FEATURES, EDGE_SPIN, EDGE_SPIN_DISTANCE])
register_fields(graph_fields=[PER_ATOM_SPIN_KEY])
