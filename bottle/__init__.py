"""Impossible Bottle – Bayesian layered ship-in-bottle generator."""
from .controller import BottleController, SceneConfig
from .pipeline import run, generate_from_config
from .layers import LayerGenerator

__all__ = [
    "BottleController",
    "SceneConfig",
    "run",
    "generate_from_config",
    "LayerGenerator",
]
__version__ = "0.2.0"
