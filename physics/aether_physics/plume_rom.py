"""Reduced-order nozzle/plume model (REDUCED-ORDER MODEL - not CFD).

* Quasi-1D isentropic nozzle relations (area-Mach, pressure-Mach) for a calorically perfect
  gas with the Cantera-derived exit gamma.
* Initial jet-boundary turning angle from the Prandtl-Meyer function:
      theta_b = theta_lip + nu(M_j) - nu(M_e),   M_j: Mach after isentropic expansion to p_a
* Fully-expanded jet diameter D_j = D_e * sqrt(A(M_j)/A(M_e))
* Shock-cell spacing (Pack 1950):  L_s = 1.306 * D_j * sqrt(M_j^2 - 1)
* First Mach-disk distance for under-expanded jets (Crist, Sherman & Glass 1966 correlation,
  using the throat diameter):  x_m = 0.67 * D_t * sqrt(p_c / p_a)
The same formulas are evaluated per frame in frontend/src/engine/plumeModel.ts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def area_ratio(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area_ratio(eps, g, supersonic=True):
    f = lambda M: area_ratio(M, g) - eps
    return brentq(f, 1.0 + 1e-12, 100.0) if supersonic else brentq(f, 1e-6, 1.0)


def p0_over_p(M, g):
    return (1.0 + 0.5 * (g - 1.0) * M * M) ** (g / (g - 1.0))


def mach_from_pressure_ratio(p0p, g):
    return math.sqrt(2.0 / (g - 1.0) * (p0p ** ((g - 1.0) / g) - 1.0))


def prandtl_meyer(M, g):
    if M <= 1.0:
        return 0.0
    a = math.sqrt((g + 1.0) / (g - 1.0))
    return a * math.atan(math.sqrt((g - 1.0) / (g + 1.0) * (M * M - 1.0))) - math.atan(math.sqrt(M * M - 1.0))


def plume_state(pc, pe, ae, g, pa, throttle=1.0, lip_deg=10.0):
    pc = pc * max(throttle, 0.05)
    pe = pe * max(throttle, 0.05)
    Me = mach_from_pressure_ratio(pc / pe, g)
    pa_eff = max(pa, 1e-3)
    ratio = pe / pa_eff
    Mj = mach_from_pressure_ratio(min(pc / pa_eff, 1e12), g)
    turn = prandtl_meyer(Mj, g) - prandtl_meyer(Me, g)
    theta_b = max(-0.35, min(1.45, math.radians(lip_deg) + turn))
    de = 2.0 * math.sqrt(ae / math.pi)
    dj = de * math.sqrt(min(area_ratio(Mj, g) / area_ratio(Me, g), 1e4))
    cell = 1.306 * dj * math.sqrt(max(Mj * Mj - 1.0, 0.0))
    strength = min(abs(math.log(ratio)) / math.log(2.5), 1.0)
    density = min(pa / 25000.0, 1.0)
    diamonds = 0.0 if pa < 500 else max(0.15, strength) * density
    mach_disk = pa > 500 and (ratio < 0.45 or ratio > 2.2)
    dt = de / math.sqrt(area_ratio(Me, g))
    xm = 0.67 * dt * math.sqrt(pc / pa_eff) if ratio > 1 else 0.6 * cell
    regime = "vacuum" if pa < 50 else "over-expanded" if ratio < 0.9 else "under-expanded" if ratio > 1.1 else "near-ideal"
    return dict(pe=pe, pa=pa, ratio=ratio, Me=Me, Mj=Mj, thetaB=theta_b, dj=dj, cellLength=cell, diamonds=diamonds,
                machDisk=bool(mach_disk), machDiskX=xm, regime=regime)


def nozzle_profile(eps, g, n=60):
    """Quasi-1D supersonic-branch profile: A/A*, M, p/p0, T/T0 from throat to exit."""
    ar = np.linspace(1.0, eps, n)
    M = np.array([mach_from_area_ratio(a, g) if a > 1.0 + 1e-9 else 1.0 for a in ar])
    return dict(area_ratio=ar.tolist(), mach=M.tolist(),
                p_over_p0=[1.0 / p0_over_p(m, g) for m in M],
                T_over_T0=[1.0 / (1.0 + 0.5 * (g - 1.0) * m * m) for m in M])
