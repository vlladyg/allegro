"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_SPIN: Final[str] = "edge_spin"
EDGE_FEATURES: Final[str] = "edge_features"
PER_ATOM_SPIN_KEY: Final[str] = "atomic_spin"

register_fields(edge_fields=[EDGE_ENERGY, EDGE_FEATURES, EDGE_SPIN, PER_ATOM_SPIN_KEY])
