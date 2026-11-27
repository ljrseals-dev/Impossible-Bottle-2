"""
Impossible Bottle API server.

Endpoints
---------
POST /api/generate
    Accepts either a partial evidence dict or a full SceneConfig JSON.
    Returns the resolved SceneConfig, layer prompts, and paths to the
    generated artefacts.  Designed so the controller output plugs
    straight in.

GET  /api/query
    Bayesian query: P(variable | evidence)

GET  /health
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from bottle import BottleController, SceneConfig, generate_from_config, run

# Optional FastAPI – fall back to a plain function interface if not installed
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
# Shared logic (works with or without FastAPI)
# ---------------------------------------------------------------------------
def generate_api(
    evidence: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    resolution: int = 256,
    outdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Core entry point used by both the HTTP handler and CLI."""
    ctl = BottleController(seed=seed)

    if config:
        cfg = SceneConfig.from_dict(config)
        cfg.seed = seed
        cfg.resolution = resolution
    else:
        cfg = ctl.sample_config(evidence=evidence or {}, seed=seed)
        cfg.resolution = resolution

    if outdir is None:
        outdir = tempfile.mkdtemp(prefix="bottle_")

    res, canvas, layers = generate_from_config(cfg, outdir=outdir)

    payload = cfg.to_api_payload()
    return {
        "status": "ok",
        "config": res["config"],
        "prompts": payload["prompts"],
        "negative_prompt": payload["negative_prompt"],
        "scene_logprob": res["scene_logprob"],
        "layer_uncertainty": res["layer_uncertainty"],
        "checks": res["checks"],
        "outdir": res["outdir"],
        "composite": str(Path(outdir) / "composite.png"),
        "manifest": str(Path(outdir) / "manifest.json"),
    }


def query_api(variable: str, evidence: Dict[str, str], method: str = "enumerate") -> Dict[str, Any]:
    ctl = BottleController()
    dist = ctl.query(variable, evidence, method=method)
    return {"variable": variable, "evidence": evidence, "distribution": dist}


# ---------------------------------------------------------------------------
# FastAPI app (if available)
# ---------------------------------------------------------------------------
if HAS_FASTAPI:
    app = FastAPI(title="Impossible Bottle", version="0.2.0")

    class GenerateRequest(BaseModel):
        evidence: Optional[Dict[str, str]] = None
        config: Optional[Dict[str, Any]] = None
        seed: int = 42
        resolution: int = 256
        outdir: Optional[str] = None

    class QueryRequest(BaseModel):
        variable: str
        evidence: Dict[str, str] = Field(default_factory=dict)
        method: str = "enumerate"

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/api/generate")
    def api_generate(req: GenerateRequest):
        try:
            return generate_api(
                evidence=req.evidence,
                config=req.config,
                seed=req.seed,
                resolution=req.resolution,
                outdir=req.outdir,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/query")
    def api_query(req: QueryRequest):
        try:
            return query_api(req.variable, req.evidence, req.method)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sample")
    def api_sample(seed: int = 42):
        ctl = BottleController(seed=seed)
        cfg = ctl.sample_config(seed=seed)
        return cfg.to_api_payload()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Impossible Bottle generator")
    parser.add_argument("--evidence", type=str, default="{}", help="JSON evidence dict")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--outdir", type=str, default="bottle_out")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        if not HAS_FASTAPI:
            print("FastAPI not installed.  pip install fastapi uvicorn")
            raise SystemExit(1)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        evidence = json.loads(args.evidence)
        result = generate_api(
            evidence=evidence,
            seed=args.seed,
            resolution=args.resolution,
            outdir=args.outdir,
        )
        print(json.dumps(result, indent=2))
