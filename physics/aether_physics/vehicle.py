"""Vehicle model: configuration, mass properties, engine and aerodynamic models."""
from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(DATA_DIR, "vehicle.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + f * (ys[i + 1] - ys[i])


@dataclass
class MassProps:
    m: float
    x_cg: float
    ixx: float
    iyy: float  # = izz (axisymmetric)


def _cyl(m, x0, x1, r, shell):
    """Mass, centroid, Ixx, Iyy-about-own-centroid of a cylinder segment."""
    l = max(x1 - x0, 1e-6)
    if shell:
        ixx = m * r * r
        iyy = m * (r * r / 2.0 + l * l / 12.0)
    else:
        ixx = 0.5 * m * r * r
        iyy = m * (3.0 * r * r + l * l) / 12.0
    return m, 0.5 * (x0 + x1), ixx, iyy


def mass_props(cfg: dict, mp1: float, mp2: float, stage: int, fairing: bool) -> MassProps:
    g = cfg["geometry"]
    ms = cfg["mass"]
    r = g["radius_m"]
    comps = []
    if stage == 1:
        s1 = g["stage1"]
        comps.append(_cyl(ms["stage1_dry_kg"], s1["x0"], s1["x1"], r, True))
        if mp1 > 0:
            frac = mp1 / ms["stage1_prop_kg"]
            comps.append(_cyl(mp1, s1["tank_x0"], s1["tank_x0"] + frac * (s1["tank_x1"] - s1["tank_x0"]), r * 0.98, False))
    s2 = g["stage2"]
    comps.append(_cyl(ms["stage2_dry_kg"], s2["x0"], s2["x1"], r, True))
    if mp2 > 0:
        frac = mp2 / ms["stage2_prop_kg"]
        comps.append(_cyl(mp2, s2["tank_x0"], s2["tank_x0"] + frac * (s2["tank_x1"] - s2["tank_x0"]), r * 0.98, False))
    pl = g["payload"]
    comps.append(_cyl(ms["payload_kg"], pl["x0"], pl["x1"], pl["radius_m"], False))
    if fairing:
        fa = g["fairing"]
        comps.append(_cyl(ms["fairing_kg"], fa["x0"], fa["x1"], fa["radius_m"], True))
    m = sum(c[0] for c in comps)
    xcg = sum(c[0] * c[1] for c in comps) / m
    ixx = sum(c[2] for c in comps)
    iyy = sum(c[3] + c[0] * (c[1] - xcg) ** 2 for c in comps)
    return MassProps(m, xcg, ixx, iyy)


def engine_thrust(eng: dict, thr: float, pa: float) -> tuple[float, float]:
    """Total thrust [N] and total mass flow [kg/s] for a stage's engine set.

    F = n * ( thr*mdot0*ve + (thr*pe0 - pa)*Ae ), clamped >= 0.
    Exit pressure scales with chamber pressure, i.e. with throttle.
    """
    if thr <= 0.0:
        return 0.0, 0.0
    n = eng["count"]
    mdot = thr * eng["mdot0_kg_s"]
    f = n * (mdot * eng["ve_m_s"] + (thr * eng["pe_pa"] - pa) * eng["ae_m2"])
    return max(f, 0.0), n * mdot


def ca_of_mach(cfg: dict, mach: float) -> float:
    a = cfg["aero"]
    return interp(a["ca_mach"], a["ca_value"], mach)


def wind_speed(cfg: dict, h: float) -> float:
    w = cfg["wind"]
    return interp(w["alt_m"], w["speed_m_s"], h)
