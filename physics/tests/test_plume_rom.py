"""Reduced-order plume model: textbook gas-dynamics checks (NACA 1135 table values)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aether_physics import plume_rom as pr  # noqa: E402
from aether_physics.atmosphere import atmosphere  # noqa: E402


def test_area_mach_naca1135():
    assert pr.area_ratio(2.0, 1.4) == pytest.approx(1.6875, rel=1e-4)
    assert pr.area_ratio(3.0, 1.4) == pytest.approx(4.2346, rel=1e-4)
    assert pr.mach_from_area_ratio(1.6875, 1.4) == pytest.approx(2.0, rel=1e-6)


def test_prandtl_meyer_naca1135():
    assert math.degrees(pr.prandtl_meyer(2.0, 1.4)) == pytest.approx(26.380, abs=0.01)
    assert math.degrees(pr.prandtl_meyer(3.0, 1.4)) == pytest.approx(49.757, abs=0.01)


def test_pressure_mach_roundtrip():
    for M in (1.5, 3.0, 5.0):
        assert pr.mach_from_pressure_ratio(pr.p0_over_p(M, 1.22), 1.22) == pytest.approx(M, rel=1e-10)


def test_regimes_follow_ambient_pressure():
    pc, pe, ae, g = 9.7e6, 69060.0, 0.8364, 1.226
    sl = pr.plume_state(pc, pe, ae, g, atmosphere(0)[1])
    hi = pr.plume_state(pc, pe, ae, g, atmosphere(30000)[1])
    vac = pr.plume_state(pc, pe, ae, g, 1e-6)
    assert sl["regime"] == "over-expanded" and sl["dj"] < 2 * math.sqrt(ae / math.pi)
    assert hi["regime"] == "under-expanded" and hi["thetaB"] > sl["thetaB"]
    assert vac["regime"] == "vacuum" and vac["diamonds"] == 0.0
    # perfectly expanded jet: no turning beyond the lip angle and D_j = D_e
    ideal = pr.plume_state(pc, pe, ae, g, pe)
    assert ideal["dj"] == pytest.approx(2 * math.sqrt(ae / math.pi), rel=1e-9)
    assert ideal["thetaB"] == pytest.approx(math.radians(10.0), abs=1e-9)


def test_quasi1d_profile_monotonic():
    prof = pr.nozzle_profile(16.0, 1.226)
    M = prof["mach"]
    assert M[0] == 1.0 and all(M[i + 1] > M[i] for i in range(len(M) - 1))
    assert prof["p_over_p0"][-1] == pytest.approx(1 / pr.p0_over_p(M[-1], 1.226))
