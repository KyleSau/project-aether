"""Project Aether backend (FastAPI).

Serves the built frontend and a small API. All heavy computation is done offline by
backend/precompute.py; the API only serves those files or runs short reference computations.

    uvicorn backend.app:app --port 8300
"""
from __future__ import annotations

import json
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "physics"))
sys.path.insert(0, HERE)

from aether_physics.sim import run_mission, orbit_check  # noqa: E402
from aether_physics.vehicle import load_config  # noqa: E402
from openfoam import adapter as of_adapter  # noqa: E402
from openfoam import case_generator  # noqa: E402

DATA = os.path.join(ROOT, "data")
DIST = os.path.join(ROOT, "frontend", "dist")

app = FastAPI(title="Project Aether API", version="0.1.0")


def _json(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        raise HTTPException(404, f"{name} not generated; run backend/precompute.py")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"ok": True, "time": time.time()}


@app.get("/api/vehicle")
def vehicle():
    return _json("vehicle.json")


@app.get("/api/thermo")
def thermo():
    return _json("thermo.json")


@app.get("/api/plume")
def plume():
    return _json("plume_table.json")


@app.get("/api/reference")
def reference():
    return _json("reference_trajectory.json")


class SimRequest(BaseModel):
    kick_angle_deg: float | None = Field(None, ge=0.5, le=15)
    target_alt_m: float | None = Field(None, ge=150e3, le=600e3)
    payload_kg: float | None = Field(None, ge=0, le=25000)
    launch_azimuth_deg: float | None = Field(None, ge=30, le=150)
    rotating_earth: bool | None = None
    j2: bool | None = None


@app.post("/api/simulate")
def simulate(req: SimRequest):
    """Run the authoritative solve_ivp mission with optional overrides (takes ~1 s)."""
    cfg = load_config()
    g = cfg["guidance"]
    if req.kick_angle_deg is not None:
        g["kick_angle_deg"] = req.kick_angle_deg
    if req.target_alt_m is not None:
        g["target_alt_m"] = req.target_alt_m
    if req.payload_kg is not None:
        cfg["mass"]["payload_kg"] = req.payload_kg
    if req.launch_azimuth_deg is not None:
        cfg["site"]["launch_azimuth_deg"] = req.launch_azimuth_deg
    if req.rotating_earth is not None:
        cfg["site"]["rotating_earth"] = req.rotating_earth
    if req.j2 is not None:
        cfg["site"]["j2"] = req.j2
    t0 = time.time()
    try:
        r = run_mission(cfg, t_orbit_coast=60.0, sample_dt=2.0)
    except RuntimeError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
    el = r["elements_seco"]
    return {
        "ok": True, "label": "SOLVER DATA (scipy solve_ivp DOP853)", "wall_s": time.time() - t0,
        "events": [[n, float(t)] for n, t in r["events"]],
        "max_q": {k: float(v) for k, v in r["max_q"].items()},
        "elements_seco": {k: float(v) for k, v in el.items()},
        "orbit_achieved": orbit_check(el),
        "track": [[float(s["t"]), float(s["h"]), float(s["v_rel"])] for s in r["samples"]],
    }


@app.get("/api/openfoam/status")
def openfoam_status():
    return of_adapter.detect()


class CaseRequest(BaseModel):
    stage: str = Field("stage1", pattern="^stage[12]$")
    altitude_m: float = Field(10000.0, ge=0, le=150000)


@app.post("/api/openfoam/case")
def openfoam_case(req: CaseRequest):
    """Generate (but do not run) an OpenFOAM case directory for the selected condition."""
    case = case_generator.from_project(req.stage, req.altitude_m)
    out = os.path.join(HERE, "openfoam", "cases", f"{req.stage}_h{int(req.altitude_m / 1000)}km")
    files = case_generator.write_case(case, out)
    return {"status": of_adapter.detect()["status"], "case_dir": out, "files": files,
            "note": "Case generated only. No solver has been run; no CFD results exist."}


if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")
    for sub in ("data", "tex"):
        if os.path.isdir(os.path.join(DIST, sub)):
            app.mount(f"/{sub}", StaticFiles(directory=os.path.join(DIST, sub)), name=sub)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(DIST, "index.html"))
