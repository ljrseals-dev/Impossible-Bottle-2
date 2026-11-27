"""Layered compositing with approximate Snell's-law refraction and Fresnel.

Writes:
  - final composite PNG
  - individual layer PNGs
  - per-layer uncertainty maps
  - manifest.json with hash-chained provenance
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from .scene_config import SceneConfig
from .controller import BottleController
from .layers import LayerGenerator, LayerResult


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _save_rgba(arr: np.ndarray, path: Path):
    """arr H×W×4 float32 → PNG."""
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(u8, "RGBA").save(path)


def _save_gray(arr: np.ndarray, path: Path):
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(u8, "L").save(path)


def _refract_displace(
    src: np.ndarray,
    glass_mask: np.ndarray,
    eta: float = 1.52,
    strength: float = 3.5,
) -> np.ndarray:
    """Very approximate radial refraction displacement inside the glass mask.

    For a toy pipeline we displace pixels toward the bottle centre
    proportionally to a crude normal estimate.  Real Snell would need
    a proper surface normal field; this is enough to show the effect.
    """
    H, W = src.shape[:2]
    out = src.copy()
    yy, xx = np.mgrid[:H, :W]
    cx, cy = W / 2.0, H * 0.48

    # radial vector
    dx = (xx - cx) / max(W * 0.28, 1)
    dy = (yy - cy) / max(H * 0.35, 1)
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

    # approximate outward normal
    nx, ny = dx / r, dy / r

    # crude Snell scaling (air → glass)
    # displace inward for the background / ship layers
    factor = (1.0 - 1.0 / eta) * strength
    map_x = np.clip(xx - nx * factor * glass_mask, 0, W - 1).astype(np.float32)
    map_y = np.clip(yy - ny * factor * glass_mask, 0, H - 1).astype(np.float32)

    # bilinear sample
    x0 = np.floor(map_x).astype(int)
    y0 = np.floor(map_y).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)

    wa = (x1 - map_x) * (y1 - map_y)
    wb = (map_x - x0) * (y1 - map_y)
    wc = (x1 - map_x) * (map_y - y0)
    wd = (map_x - x0) * (map_y - y0)

    for c in range(src.shape[2]):
        out[..., c] = (
            wa * src[y0, x0, c]
            + wb * src[y0, x1, c]
            + wc * src[y1, x0, c]
            + wd * src[y1, x1, c]
        )
    return out


def composite_layers(
    layers: Dict[str, LayerResult],
    cfg: SceneConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Bottom-up composite with refraction of ship layers through glass."""
    order = ["table", "hull", "rigging", "resin", "glass"]
    H, W = layers["table"].rgba.shape[:2]
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    # dark studio background
    canvas[..., 0:3] = 0.06
    canvas[..., 3] = 1.0

    glass_mask = layers["glass"].mask.astype(np.float32)
    eta = cfg.refraction_eta

    # first lay table (no refraction)
    t = layers["table"].rgba
    a = t[..., 3:4]
    canvas = canvas * (1 - a) + t * a

    # ship layers – refract them
    for name in ("hull", "rigging"):
        src = layers[name].rgba
        refracted = _refract_displace(src, glass_mask, eta=eta, strength=2.8)
        a = refracted[..., 3:4]
        canvas = canvas * (1 - a) + refracted * a

    # resin (inside neck, no heavy refraction)
    r = layers["resin"].rgba
    a = r[..., 3:4]
    canvas = canvas * (1 - a) + r * a

    # glass on top with Fresnel already baked into its alpha
    g = layers["glass"].rgba
    a = g[..., 3:4] * cfg.fresnel_strength
    # screen-like highlight blend
    canvas[..., :3] = np.clip(
        canvas[..., :3] * (1 - a * 0.6) + g[..., :3] * a, 0, 1
    )
    canvas[..., 3] = np.clip(canvas[..., 3] + a[..., 0] * 0.3, 0, 1)

    # mean uncertainty per layer
    unc = {k: float(v.uncertainty[v.mask].mean()) if v.mask.any() else float(v.uncertainty.mean())
           for k, v in layers.items()}
    return canvas, unc


def run(
    evidence: Optional[Dict[str, str]] = None,
    seed: int = 1,
    outdir: str = "bottle_out",
    epochs: int = 40,
    resolution: int = 256,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, LayerResult]]:
    """Full pipeline used by the original run_bottle.py script."""
    ctl = BottleController(seed=seed)
    cfg = ctl.sample_config(evidence=evidence or {}, seed=seed)
    cfg.resolution = resolution
    cfg.epochs = epochs
    cfg.seed = seed

    return generate_from_config(cfg, outdir=outdir)


def generate_from_config(
    cfg: SceneConfig,
    outdir: str = "bottle_out",
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, LayerResult]]:
    """Generate from an already-resolved SceneConfig (API path)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    gen = LayerGenerator(cfg)
    layers = gen.generate_all()

    # optional residual “epochs” – just tiny noise that decreases (toy)
    for e in range(cfg.epochs):
        noise_scale = 0.008 * (1.0 - e / max(cfg.epochs, 1))
        for name in layers:
            if name == "glass":
                continue
            n = gen.rng.normal(0, noise_scale, layers[name].rgba.shape)
            layers[name].rgba = np.clip(layers[name].rgba + n * layers[name].mask[..., None], 0, 1)

    canvas, layer_unc = composite_layers(layers, cfg)

    # write artefacts
    _save_rgba(canvas, out / "composite.png")
    for name, lr in layers.items():
        _save_rgba(lr.rgba, out / f"layer_{name}.png")
        _save_gray(lr.uncertainty, out / f"unc_{name}.png")

    # provenance
    hashes = {}
    for p in sorted(out.glob("*.png")):
        hashes[p.name] = _hash_bytes(p.read_bytes())

    # scene log-prob (toy): product of CPT probs under the controller
    ctl = BottleController(seed=cfg.seed)
    logprob = 0.0
    assignment = {
        "bottleShape": cfg.bottleShape,
        "shipType": cfg.shipType,
        "table": cfg.table,
        "light": cfg.light,
        "aspect": cfg.aspect,
    }
    for var in ctl._topo_order():
        pars = ctl.parents[var]
        pvals = tuple(assignment[p] for p in pars)
        p = ctl._get_cpt(var, pvals).get(assignment[var], 1e-9)
        logprob += float(np.log(p + 1e-12))

    checks = {
        "has_table": bool(layers["table"].mask.any()),
        "has_glass": bool(layers["glass"].mask.any()),
        "has_hull": bool(layers["hull"].mask.any()),
        "has_rigging": bool(layers["rigging"].mask.any()),
        "composite_finite": bool(np.isfinite(canvas).all()),
        "resolution_ok": canvas.shape[0] >= 64 and canvas.shape[1] >= 64,
    }

    manifest = {
        "version": "0.2.0",
        "config": cfg.to_dict(),
        "api_payload": cfg.to_api_payload(),
        "layer_uncertainty": layer_unc,
        "scene_logprob": logprob,
        "checks": checks,
        "hashes": hashes,
        "psd": "PNG layers + manifest (psd-tools not required)",
    }
    # hash-chain the manifest itself
    man_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_hash"] = _hash_bytes(man_bytes)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    res = {
        "config": cfg.to_dict(),
        "scene_logprob": logprob,
        "layer_uncertainty": layer_unc,
        "checks": checks,
        "psd": manifest["psd"],
        "outdir": str(out),
    }
    return res, canvas, layers
