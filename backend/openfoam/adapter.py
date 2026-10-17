"""OpenFOAM adapter: availability detection, run orchestration and result parsing.

Detection is read-only (it never installs anything, never starts Docker, never pulls images).
Current machine status is reported as NOT CONNECTED unless an OpenFOAM executable is found.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

REQUIREMENTS = [
    "Option A (recommended on Windows): WSL2 + Ubuntu 22.04/24.04 + OpenFOAM from openfoam.org "
    "(`sudo apt install openfoam11`) or ESI (`openfoam2406` packages). Run: `wsl -d Ubuntu bash -lc 'cd <case> && ./Allrun'`.",
    "Option B: Docker Desktop running + an OpenFOAM image (e.g. `opencfd/openfoam-run:2406`); "
    "mount the generated case directory and run ./Allrun inside the container.",
    "CPU: 8+ cores recommended (case defaults to 8 MPI ranks, ~60k cells per 2-D wedge case; minutes per case).",
    "Disk: ~2 GB for OpenFOAM, <200 MB per case.",
    "Permission from the user is required to install WSL distributions / pull Docker images.",
]


def _which(cmd):
    return shutil.which(cmd)


def detect() -> dict:
    checks = []
    native = _which("blockMesh") and _which("rhoCentralFoam")
    checks.append(dict(check="native blockMesh/rhoCentralFoam on PATH", ok=bool(native)))
    wsl_ok = False
    wsl = _which("wsl") or _which("wsl.exe")
    if wsl:
        try:
            out = subprocess.run([wsl, "-l", "-q"], capture_output=True, timeout=5)
            names = out.stdout.decode("utf-16-le", errors="ignore").replace("\x00", "").split()
            checks.append(dict(check="WSL distributions", ok=bool(names), detail=names))
            wsl_ok = bool(names)
        except Exception as e:  # pragma: no cover
            checks.append(dict(check="WSL distributions", ok=False, detail=str(e)))
    else:
        checks.append(dict(check="WSL available", ok=False))
    docker = _which("docker")
    docker_running = False
    if docker:
        try:
            r = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=8)
            docker_running = r.returncode == 0
        except Exception:
            docker_running = False
    checks.append(dict(check="Docker engine running", ok=docker_running, detail="installed" if docker else "not installed"))
    connected = bool(native)
    return dict(status="CONNECTED" if connected else "NOT CONNECTED", connected=connected, checks=checks,
                requirements=REQUIREMENTS,
                note="WSL distro or Docker presence alone is not sufficient; an OpenFOAM installation inside it is required."
                if (wsl_ok or docker_running) else "No OpenFOAM installation detected.")


def parse_axis_results(case_dir: str):
    """Parse sampled axis data (p, T, U, Ma) from a COMPLETED run. Returns None if no completed run exists."""
    if not os.path.exists(os.path.join(case_dir, "AETHER_RUN_COMPLETE")):
        return None
    files = sorted(glob.glob(os.path.join(case_dir, "postProcessing", "axisProbe", "*", "axis*.csv")))
    if not files:
        return None
    import csv
    rows = list(csv.DictReader(open(files[-1], encoding="utf-8")))
    return dict(label="SOLVER DATA (OpenFOAM rhoCentralFoam)", source=files[-1], rows=rows)
