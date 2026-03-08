"""
AetherForge – Continuous-Time Geometric Graph Network (CT-GGN) runtime.

Subsystems
----------
topology     : Flower-of-Life 37-node graph (inner-19 liquid core + outer-18 perimeter)
liquid_core  : Liquid Neural Network nodes driven by Neural ODEs
perimeter    : Geometric Algebra (Clifford Cl(3,0)) perimeter nodes with DDL erasure
echo_ripple  : Wave-interference Hebbian online learning protocol
ledger       : Hash-chained immutable conscience ledger
actor        : Aetherling actor model (spawn / fork / swarm)
aetherscript : AetherScript YAML deployment-spec parser & validator
runtime      : Top-level CT-GGN runtime combining all subsystems
"""

from .topology import FlowerOfLife
from .liquid_core import LiquidNode, LiquidCore
from .perimeter import PerimeterNode, PerimeterRing
from .echo_ripple import EchoRippleProtocol
from .ledger import ConscienceLedger
from .actor import Aetherling
from .aetherscript import load_aetherscript, validate_aetherscript
from .runtime import CTGGNRuntime

__all__ = [
    "FlowerOfLife",
    "LiquidNode",
    "LiquidCore",
    "PerimeterNode",
    "PerimeterRing",
    "EchoRippleProtocol",
    "ConscienceLedger",
    "Aetherling",
    "load_aetherscript",
    "validate_aetherscript",
    "CTGGNRuntime",
]
