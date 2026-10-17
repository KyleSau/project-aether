"""Classical orbital elements from an inertial state vector."""
from __future__ import annotations

import math

from .dynamics import MU, RE


def elements(r, v, mu=MU):
    rx, ry, rz = r
    vx, vy, vz = v
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    v2 = vx * vx + vy * vy + vz * vz
    hx, hy, hz = ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    energy = 0.5 * v2 - mu / rn
    a = -mu / (2 * energy) if abs(energy) > 1e-12 else math.inf
    # eccentricity vector
    rv = rx * vx + ry * vy + rz * vz
    ex = ((v2 - mu / rn) * rx - rv * vx) / mu
    ey = ((v2 - mu / rn) * ry - rv * vy) / mu
    ez = ((v2 - mu / rn) * rz - rv * vz) / mu
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, hz / hn))))
    if energy < 0:
        rp, ra = a * (1 - e), a * (1 + e)
        period = 2 * math.pi * math.sqrt(a ** 3 / mu)
    else:
        rp, ra, period = hn * hn / mu / (1 + e), math.inf, math.inf
    return dict(a=a, e=e, i_deg=inc, rp=rp, ra=ra, periapsis_alt=rp - RE, apoapsis_alt=ra - RE,
                period=period, energy=energy, h=hn)
