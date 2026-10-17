"""U.S. Standard Atmosphere 1976.

0-86 km: exact layer equations of USSA-1976 (geopotential altitude, 7 layers).
86-1000 km: log-linear interpolation of the published USSA-1976 tabulated
values (T, P, rho) -- an approximation of the full 1976 upper-atmosphere
model, accurate to the table resolution.
"""
from __future__ import annotations

import math

G0 = 9.80665
R_STAR = 8314.32          # J/(kmol K) (USSA-1976 value)
M0 = 28.9644              # kg/kmol
R_AIR = R_STAR / M0       # 287.0531 J/(kg K)
R0_GEOPOT = 6356766.0     # m, effective Earth radius for geopotential altitude
GAMMA_AIR = 1.4

# base geopotential altitude [m], base temperature [K], lapse rate [K/m]
_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
    (84852.0, 186.946, 0.0),
]


def _base_pressures():
    p = [101325.0]
    for i in range(len(_LAYERS) - 1):
        hb, tb, lb = _LAYERS[i]
        h1 = _LAYERS[i + 1][0]
        if lb == 0.0:
            p.append(p[-1] * math.exp(-G0 * (h1 - hb) / (R_AIR * tb)))
        else:
            t1 = tb + lb * (h1 - hb)
            p.append(p[-1] * (t1 / tb) ** (-G0 / (R_AIR * lb)))
    return p


_PB = _base_pressures()

# USSA-1976 tabulated upper atmosphere (geometric altitude m, T K, P Pa, rho kg/m3)
UPPER_TABLE = [
    (86000.0, 186.87, 3.7338e-1, 6.958e-6),
    (90000.0, 186.87, 1.8359e-1, 3.416e-6),
    (100000.0, 195.08, 3.2011e-2, 5.604e-7),
    (110000.0, 240.00, 7.1042e-3, 9.708e-8),
    (120000.0, 360.00, 2.5382e-3, 2.222e-8),
    (150000.0, 634.39, 4.5422e-4, 2.076e-9),
    (200000.0, 854.56, 8.4736e-5, 2.541e-10),
    (300000.0, 976.01, 8.7704e-6, 1.916e-11),
    (400000.0, 995.83, 1.4518e-6, 2.803e-12),
    (500000.0, 999.24, 3.0236e-7, 5.215e-13),
    (600000.0, 999.85, 8.2130e-8, 1.137e-13),
    (800000.0, 999.99, 1.7036e-8, 1.136e-14),
    (1000000.0, 1000.0, 7.5138e-9, 3.561e-15),
]


def geopotential(z: float) -> float:
    return R0_GEOPOT * z / (R0_GEOPOT + z)


def atmosphere(z: float):
    """Return (T [K], P [Pa], rho [kg/m^3], a [m/s]) at geometric altitude z [m]."""
    if z < 0.0:
        z = 0.0
    if z <= 86000.0:
        h = geopotential(z)
        i = len(_LAYERS) - 1
        while i > 0 and h < _LAYERS[i][0]:
            i -= 1
        hb, tb, lb = _LAYERS[i]
        pb = _PB[i]
        if lb == 0.0:
            t = tb
            p = pb * math.exp(-G0 * (h - hb) / (R_AIR * tb))
        else:
            t = tb + lb * (h - hb)
            p = pb * (t / tb) ** (-G0 / (R_AIR * lb))
        # Molecular-temperature to kinetic: M/M0 = 1 below 80 km; ignore small 80-86 km correction
        rho = p / (R_AIR * t)
        return t, p, rho, math.sqrt(GAMMA_AIR * R_AIR * t)
    tab = UPPER_TABLE
    if z >= tab[-1][0]:
        z0, t0, p0, r0 = tab[-2]
        z1, t1, p1, r1 = tab[-1]
    else:
        k = 0
        while tab[k + 1][0] < z:
            k += 1
        z0, t0, p0, r0 = tab[k]
        z1, t1, p1, r1 = tab[k + 1]
    f = (z - z0) / (z1 - z0)
    t = t0 + f * (t1 - t0)
    p = math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))
    rho = math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0)))
    return t, p, rho, math.sqrt(GAMMA_AIR * R_AIR * t)
