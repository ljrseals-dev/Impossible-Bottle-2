"""Canonical SceneConfig schema that plugs directly into main.py /api/generate."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class SceneConfig:
    """Strict scene description consumed by the generator and the API."""

    # Core visual variables
    bottleShape: str = "classic round, long neck"
    shipType: str = "full-rigged ship"
    table: str = "dark oak table, grain left-to-right, worn finish"
    light: str = "camera-left window light"
    aspect: str = "portrait 3:4"
    background: str = "deep charcoal studio"

    # Generation controls
    resolution: int = 256
    seed: int = 42
    epochs: int = 40          # residual VAE refinement steps (toy)
    refraction_eta: float = 1.52
    fresnel_strength: float = 0.65

    # Optional free-form overrides / prompts for the API
    prompt_overrides: Dict[str, str] = field(default_factory=dict)
    negative_prompt: str = "blurry, low-res, distorted glass, floating ship, bad anatomy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_api_payload(self) -> Dict[str, Any]:
        """Payload ready for POST /api/generate."""
        prompts = self.build_layer_prompts()
        return {
            "config": self.to_dict(),
            "prompts": prompts,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "width": self.resolution,
            "height": int(self.resolution * 1.25) if "portrait" in self.aspect else self.resolution,
        }

    def build_layer_prompts(self) -> Dict[str, str]:
        """Layer-specific prompts that the downstream model (or procedural backend) can use."""
        base = (
            f"ship in a bottle, {self.bottleShape}, {self.shipType}, "
            f"on {self.table}, illuminated by {self.light}, "
            f"{self.background}, highly detailed, sharp glass"
        )
        overrides = self.prompt_overrides
        return {
            "table": overrides.get("table", f"{self.table}, wood grain, realistic surface, soft shadows"),
            "glass": overrides.get(
                "glass",
                f"clear glass demijohn bottle, {self.bottleShape}, refractive glass, "
                f"subtle Fresnel highlights, clean edges, η≈1.52",
            ),
            "hull": overrides.get(
                "hull",
                f"{self.shipType} wooden hull, detailed planking, realistic scale inside bottle",
            ),
            "rigging": overrides.get(
                "rigging",
                f"fine ship rigging, masts, yards, sails for {self.shipType}, delicate lines",
            ),
            "resin": overrides.get(
                "resin",
                "clear casting resin seal at bottle neck, subtle meniscus, no bubbles",
            ),
            "full": overrides.get("full", base),
        }


# Canonical domain values used by the Bayesian network
BOTTLE_SHAPES = [
    "classic round, long neck",
    "demijohn with wide belly",
    "square apothecary bottle",
    "slim cylindrical flask",
]
SHIP_TYPES = [
    "full-rigged ship",
    "schooner",
    "brigantine",
    "sloop",
    "clipper",
]
TABLES = [
    "dark oak table, grain left-to-right, worn finish",
    "mahogany desk, polished, warm tone",
    "pine workbench, rough saw marks",
    "black slate surface, matte",
]
LIGHTS = [
    "camera-left window light",
    "soft overhead studio key",
    "warm sunset side light",
    "cool north-facing window",
]
ASPECTS = ["portrait 3:4", "square 1:1", "landscape 4:3"]
