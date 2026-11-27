"""Procedural layer generator – table / glass / hull / rigging / resin.

Produces higher-resolution (default 256) RGBA layers with per-pixel
uncertainty maps.  Ship geometry is now a proper hull + masts + yards
rather than a block + stick.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from dataclasses import dataclass

from .scene_config import SceneConfig


@dataclass
class LayerResult:
    rgba: np.ndarray          # H×W×4 float32 [0,1]
    uncertainty: np.ndarray   # H×W float32
    mask: np.ndarray          # H×W bool structural mask


class LayerGenerator:
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.H = int(cfg.resolution * 1.25) if "portrait" in cfg.aspect else cfg.resolution
        self.W = cfg.resolution
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_all(self) -> Dict[str, LayerResult]:
        layers = {}
        layers["table"] = self._table()
        layers["glass"] = self._glass()
        layers["hull"] = self._hull()
        layers["rigging"] = self._rigging()
        layers["resin"] = self._resin()
        return layers

    # ------------------------------------------------------------------
    # Individual layers
    # ------------------------------------------------------------------
    def _table(self) -> LayerResult:
        H, W = self.H, self.W
        img = np.zeros((H, W, 4), dtype=np.float32)
        # table occupies bottom ~18 %
        y0 = int(H * 0.82)
        # wood colour depends on description
        if "oak" in self.cfg.table.lower():
            base = np.array([0.45, 0.28, 0.12])
        elif "mahogany" in self.cfg.table.lower():
            base = np.array([0.38, 0.18, 0.10])
        elif "pine" in self.cfg.table.lower():
            base = np.array([0.55, 0.42, 0.25])
        else:
            base = np.array([0.15, 0.15, 0.16])

        # grain
        grain = self.rng.normal(0, 0.04, (H - y0, W))
        for c in range(3):
            img[y0:, :, c] = np.clip(base[c] + grain, 0, 1)
        img[y0:, :, 3] = 1.0

        # subtle left-to-right grain lines
        for x in range(0, W, 7):
            strength = 0.03 * (0.5 + 0.5 * np.sin(x * 0.3))
            img[y0:, x, :3] = np.clip(img[y0:, x, :3] - strength, 0, 1)

        # soft shadow under bottle
        cx, cy = W // 2, int(H * 0.78)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt(((xx - cx) / (W * 0.22)) ** 2 + ((yy - cy) / (H * 0.08)) ** 2)
        shadow = np.clip(1.0 - dist, 0, 1) ** 1.8 * 0.35
        for c in range(3):
            img[..., c] = np.clip(img[..., c] - shadow, 0, 1)

        mask = np.zeros((H, W), dtype=bool)
        mask[y0:] = True
        unc = np.full((H, W), 0.04, dtype=np.float32)
        unc[y0:] = 0.025
        return LayerResult(img, unc, mask)

    def _glass(self) -> LayerResult:
        """Cleaner bottle silhouette with Fresnel-ready edge mask."""
        H, W = self.H, self.W
        img = np.zeros((H, W, 4), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        # bottle geometry – classic demijohn / round
        cx = W // 2
        # body
        body_top = int(H * 0.28)
        body_bot = int(H * 0.78)
        body_rx = int(W * 0.28)
        body_ry = int((body_bot - body_top) * 0.48)

        # neck
        neck_top = int(H * 0.12)
        neck_bot = body_top + 4
        neck_rx = int(W * 0.07)

        yy, xx = np.ogrid[:H, :W]

        # elliptical body
        body = ((xx - cx) / body_rx) ** 2 + ((yy - (body_top + body_bot) // 2) / body_ry) ** 2 <= 1.0
        body &= (yy >= body_top) & (yy <= body_bot)

        # neck cylinder
        neck = (np.abs(xx - cx) <= neck_rx) & (yy >= neck_top) & (yy <= neck_bot)

        # lip
        lip = (np.abs(xx - cx) <= neck_rx + 3) & (yy >= neck_top - 4) & (yy <= neck_top + 2)

        mask = body | neck | lip

        # glass is almost transparent – we only write a faint tint + edge
        img[mask, 0] = 0.85
        img[mask, 1] = 0.92
        img[mask, 2] = 0.95
        img[mask, 3] = 0.18          # low alpha; refraction handled later

        # edge strength (for Fresnel)
        from scipy.ndimage import binary_erosion, distance_transform_edt
        eroded = binary_erosion(mask, iterations=2)
        edge = mask & ~eroded
        # distance from edge → soft Fresnel falloff
        dist = distance_transform_edt(~edge)
        fresnel = np.exp(-dist / 3.5) * mask
        for c in range(3):
            img[..., c] = np.clip(img[..., c] + fresnel * 0.55, 0, 1)
        img[..., 3] = np.clip(img[..., 3] + fresnel * 0.7, 0, 1)

        # highlight streak (camera-left light)
        if "left" in self.cfg.light.lower():
            hx = cx - int(body_rx * 0.55)
            highlight = ((xx - hx) ** 2 / 8 + (yy - (body_top + body_bot) // 2) ** 2 / (body_ry ** 2)) < 1
            highlight &= mask
            img[highlight, :3] = np.clip(img[highlight, :3] + 0.35, 0, 1)
            img[highlight, 3] = np.clip(img[highlight, 3] + 0.25, 0, 1)

        unc = np.full((H, W), 0.06, dtype=np.float32)
        unc[mask] = 0.035
        unc[edge] = 0.055
        return LayerResult(img, unc, mask)

    def _hull(self) -> LayerResult:
        """Proper ship hull + deck, scaled to sit inside the bottle body."""
        H, W = self.H, self.W
        img = np.zeros((H, W, 4), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        cx = W // 2
        # ship sits in lower half of bottle body
        ship_y = int(H * 0.55)
        ship_len = int(W * 0.38)
        ship_h = int(H * 0.09)

        # hull outline (side view, simplified)
        # left bow, right stern
        bow_x = cx - ship_len // 2
        stern_x = cx + ship_len // 2

        # polygon for hull
        hull_pts = [
            (bow_x + 4, ship_y),                    # bow tip
            (bow_x + int(ship_len * 0.15), ship_y - ship_h),
            (stern_x - int(ship_len * 0.12), ship_y - ship_h),
            (stern_x, ship_y - 2),
            (stern_x - 6, ship_y + int(ship_h * 0.55)),
            (bow_x + 10, ship_y + int(ship_h * 0.55)),
        ]

        pil = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil)
        # wood brown
        wood = (110, 70, 35, 255)
        draw.polygon(hull_pts, fill=wood)

        # deck line
        deck_y = ship_y - ship_h + 2
        draw.line([(bow_x + 8, deck_y), (stern_x - 8, deck_y)], fill=(90, 55, 25, 255), width=2)

        # simple cabin
        cabin_x0 = cx - 6
        cabin_x1 = cx + 10
        draw.rectangle([cabin_x0, deck_y - 10, cabin_x1, deck_y], fill=(95, 60, 30, 255))

        arr = np.array(pil).astype(np.float32) / 255.0
        img = arr
        mask = arr[..., 3] > 0.1

        # slight shading
        yy, xx = np.ogrid[:H, :W]
        shade = (xx - cx) / (ship_len * 0.6)
        for c in range(3):
            img[..., c] = np.clip(img[..., c] * (1.0 - 0.15 * shade * mask), 0, 1)

        unc = np.full((H, W), 0.08, dtype=np.float32)
        unc[mask] = 0.022
        return LayerResult(img, unc, mask)

    def _rigging(self) -> LayerResult:
        """Masts, yards, sails and fine lines – the part that makes it a ship."""
        H, W = self.H, self.W
        img = np.zeros((H, W, 4), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        cx = W // 2
        ship_y = int(H * 0.55)
        ship_h = int(H * 0.09)
        deck_y = ship_y - ship_h + 2

        # mast heights depend on ship type
        if "full-rigged" in self.cfg.shipType or "clipper" in self.cfg.shipType:
            mast_heights = [0.22, 0.26, 0.20]   # three masts
            mast_xs = [-0.12, 0.0, 0.13]
        elif "schooner" in self.cfg.shipType or "brigantine" in self.cfg.shipType:
            mast_heights = [0.24, 0.20]
            mast_xs = [-0.08, 0.10]
        else:  # sloop
            mast_heights = [0.25]
            mast_xs = [0.0]

        pil = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil)

        mast_color = (55, 40, 25, 230)
        yard_color = (70, 50, 30, 200)
        sail_color = (245, 240, 225, 180)
        line_color = (40, 30, 20, 160)

        for mx_rel, mh_rel in zip(mast_xs, mast_heights):
            mx = int(cx + mx_rel * W * 0.5)
            mh = int(H * mh_rel)
            top = deck_y - mh
            # mast
            draw.line([(mx, deck_y), (mx, top)], fill=mast_color, width=2)
            # yards (cross spars)
            for frac in (0.25, 0.55, 0.85):
                yy = int(deck_y - mh * frac)
                half = int(W * 0.06 * (1.1 - frac))
                draw.line([(mx - half, yy), (mx + half, yy)], fill=yard_color, width=1)
                # simple triangular sail
                if frac < 0.9:
                    sail = [
                        (mx, top + 4),
                        (mx - half + 2, yy),
                        (mx + half - 2, yy),
                    ]
                    draw.polygon(sail, fill=sail_color)
            # stay / shroud lines
            for side in (-1, 1):
                draw.line(
                    [(mx, top + 6), (mx + side * int(W * 0.09), deck_y)],
                    fill=line_color,
                    width=1,
                )

        arr = np.array(pil).astype(np.float32) / 255.0
        img = arr
        mask = arr[..., 3] > 0.05

        unc = np.full((H, W), 0.10, dtype=np.float32)
        unc[mask] = 0.045
        return LayerResult(img, unc, mask)

    def _resin(self) -> LayerResult:
        """Clear resin seal at the neck – least certain layer."""
        H, W = self.H, self.W
        img = np.zeros((H, W, 4), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        cx = W // 2
        neck_rx = int(W * 0.07)
        y0 = int(H * 0.12)
        y1 = int(H * 0.18)

        yy, xx = np.ogrid[:H, :W]
        resin = (np.abs(xx - cx) <= neck_rx - 1) & (yy >= y0) & (yy <= y1)
        # meniscus curve
        meniscus = (yy - y0) < (3 + 2 * np.cos((xx - cx) / max(neck_rx, 1) * np.pi))
        resin &= meniscus

        img[resin, 0] = 0.9
        img[resin, 1] = 0.95
        img[resin, 2] = 0.98
        img[resin, 3] = 0.35

        mask = resin
        unc = np.full((H, W), 0.12, dtype=np.float32)
        unc[mask] = 0.09
        return LayerResult(img, unc, mask)
