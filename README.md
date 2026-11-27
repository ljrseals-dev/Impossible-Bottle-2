# Impossible Bottle v0.2

Bayesian layered generator for ship-in-a-bottle scenes.

Produces a higher-resolution (default 256×320) composite with:

- **Controller** – Bayesian network over `bottleShape → shipType`, `table → light`, etc.  
  Corrected edge directions and Gibbs sampling option.
- **Layer generator** – procedural table / glass / hull / rigging / resin with per-pixel uncertainty.
- **Pipeline** – approximate Snell refraction (η = 1.52) + Fresnel edge blend.
- **API** – `POST /api/generate` accepts evidence or full `SceneConfig` JSON and returns prompts ready for any downstream model.

## Quick start

```bash
pip install -r requirements.txt
python run_bottle.py
```

Artefacts land in `bottle_out/`:

- `composite.png`
- `layer_{table,glass,hull,rigging,resin}.png`
- `unc_*.png` (uncertainty maps)
- `manifest.json` (hash-chained provenance + SceneConfig + prompts)

## SceneConfig → /api/generate

```python
from bottle import BottleController, SceneConfig
from main import generate_api

ctl = BottleController()
cfg = ctl.sample_config({"bottleShape": "demijohn with wide belly"}, seed=7)
payload = cfg.to_api_payload()   # ready for POST /api/generate
result = generate_api(config=cfg.to_dict(), seed=7, resolution=256)
```

### Example payload fragment

```json
{
  "config": { "bottleShape": "...", "shipType": "...", ... },
  "prompts": {
    "table": "...",
    "glass": "clear glass demijohn ...",
    "hull": "...",
    "rigging": "...",
    "resin": "...",
    "full": "ship in a bottle, ..."
  },
  "negative_prompt": "blurry, low-res, ..."
}
```

## Network structure (fixed)

```
bottleShape ──► shipType ──► light
     │                          ▲
     └──────────► aspect ◄──────┤
                                │
table ──────────────────────────┘
```

## Serve the API

```bash
python main.py --serve --port 8000
# POST http://localhost:8000/api/generate
# GET  http://localhost:8000/api/sample?seed=42
```

## Honest limits

- Refraction is a radial displacement approximation, not a full ray-trace.
- Ship geometry is procedural (hull + masts + yards + sails); still a toy, not a photoreal model.
- Controller CPTs are seeded from a small set of aesthetic examples; replace with your real accepted configs for production.
- PSD export writes ordered PNGs + manifest (no `psd-tools` dependency).

## License

MIT
