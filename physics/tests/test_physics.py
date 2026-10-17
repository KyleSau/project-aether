import copy
import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether_physics import dynamics as dyn  # noqa: E402
from aether_physics.atmosphere import atmosphere  # noqa: E402
from aether_physics.orbit import elements  # noqa: E402
from aether_physics.sim import run_mission, orbit_check  # noqa: E402
from aether_physics.vehicle import load_config, engine_thrust, mass_props  # noqa: E402

G0 = 9.80665


@pytest.fixture(scope="module")
def mission():
    return run_mission(t_orbit_coast=200.0)


# ---------------------------------------------------------------- atmosphere
# USSA-1976 published table values (geometric altitude): T [K], P [Pa], rho [kg/m^3]
USSA = [
    (0, 288.150, 101325.0, 1.2250),
    (1000, 281.651, 89874.6, 1.11164),
    (5000, 255.676, 54048.3, 0.736429),
    (10000, 223.252, 26499.9, 0.413510),
    (11000, 216.774, 22699.9, 0.364801),
    (20000, 216.650, 5529.31, 0.0889097),
    (30000, 226.509, 1197.03, 0.0184101),
    (50000, 270.650, 79.7787, 1.02688e-3),
    (70000, 219.585, 5.22088, 8.28283e-5),
    (80000, 198.639, 1.05247, 1.84580e-5),
]


@pytest.mark.parametrize("z,T,P,rho", USSA)
def test_atmosphere_ussa1976(z, T, P, rho):
    t, p, r, a = atmosphere(z)
    assert t == pytest.approx(T, abs=0.01)
    assert p == pytest.approx(P, rel=2e-4)
    assert r == pytest.approx(rho, rel=2e-4)


def test_speed_of_sound_sea_level():
    assert atmosphere(0)[3] == pytest.approx(340.294, abs=0.01)


def test_upper_atmosphere_monotonic():
    zs = np.linspace(0, 1000e3, 2001)
    rho = [atmosphere(z)[2] for z in zs]
    assert all(rho[i + 1] < rho[i] for i in range(len(rho) - 1))
    assert atmosphere(200e3)[2] == pytest.approx(2.541e-10, rel=1e-6)


# ---------------------------------------------------------------- helpers
def vacuum_cfg(j2=False):
    cfg = load_config()
    cfg["site"]["j2"] = j2
    cfg["site"]["aero"] = False
    cfg["site"]["rotating_earth"] = False
    return cfg


def coast(cfg, y0, t1, rtol=1e-11):
    ctx = dyn.Ctx(phase=dyn.ORBIT, fairing=False)
    atol = np.array([1e-6] * 3 + [1e-9] * 3 + [1e-12] * 7 + [1e-6] * 2)
    return solve_ivp(lambda t, y: dyn.evaluate(t, y, ctx, cfg), (0, t1), y0, method="DOP853",
                     rtol=rtol, atol=atol, dense_output=True)


def state(r, v):
    return np.array(list(r) + list(v) + [1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)


def kepler_fg(r0, v0, dt, mu=dyn.MU):
    """Universal-variable Kepler propagation (Curtis, Algorithm 3.4)."""
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    rn = np.linalg.norm(r0)
    vr = r0 @ v0 / rn
    alpha = 2 / rn - v0 @ v0 / mu

    def C(z):
        return (1 - math.cos(math.sqrt(z))) / z if z > 0 else ((math.cosh(math.sqrt(-z)) - 1) / -z if z < 0 else 0.5)

    def S(z):
        if z > 0:
            s = math.sqrt(z)
            return (s - math.sin(s)) / s ** 3
        if z < 0:
            s = math.sqrt(-z)
            return (math.sinh(s) - s) / s ** 3
        return 1 / 6

    x = math.sqrt(mu) * abs(alpha) * dt
    for _ in range(100):
        z = alpha * x * x
        f = rn * vr / math.sqrt(mu) * x * x * C(z) + (1 - alpha * rn) * x ** 3 * S(z) + rn * x - math.sqrt(mu) * dt
        df = rn * vr / math.sqrt(mu) * x * (1 - alpha * x * x * S(z)) + (1 - alpha * rn) * x * x * C(z) + rn
        dx = f / df
        x -= dx
        if abs(dx) < 1e-12:
            break
    z = alpha * x * x
    fq = 1 - x * x / rn * C(z)
    gq = dt - x ** 3 / math.sqrt(mu) * S(z)
    r = fq * r0 + gq * v0
    return r


# ---------------------------------------------------------------- free fall / ballistic
def test_radial_free_fall_closed_form():
    cfg = vacuum_cfg()
    r0 = dyn.RE + 1500e3
    r1 = dyn.RE + 600e3
    x = r1 / r0
    t_exact = math.sqrt(r0 ** 3 / (2 * dyn.MU)) * (math.sqrt(x * (1 - x)) + math.acos(math.sqrt(x)))
    y0 = state((r0, 0, 0), (0, 0, 0))
    sol = coast(cfg, y0, t_exact)
    assert np.linalg.norm(sol.y[0:3, -1]) == pytest.approx(r1, abs=0.05)


def test_ballistic_arc_matches_kepler():
    cfg = vacuum_cfg()
    r0 = np.array([dyn.RE + 800e3, 0.0, 0.0])
    v0 = np.array([1500.0, 5200.0, 900.0])
    sol = coast(cfg, state(r0, v0), 900.0)
    rk = kepler_fg(r0, v0, 900.0)
    assert np.linalg.norm(sol.y[0:3, -1] - rk) < 0.05


# ---------------------------------------------------------------- conservation, circular orbit
def test_kepler_orbit_closes_and_conserves():
    cfg = vacuum_cfg()
    r = dyn.RE + 200e3
    vc = math.sqrt(dyn.MU / r)
    inc = math.radians(28.5)
    r0 = np.array([r, 0, 0])
    v0 = np.array([0, vc * math.cos(inc), vc * math.sin(inc)])
    el0 = elements(r0, v0)
    assert el0["e"] < 1e-12
    assert el0["a"] == pytest.approx(r, rel=1e-12)
    assert el0["i_deg"] == pytest.approx(28.5, abs=1e-9)
    period = el0["period"]
    sol = coast(cfg, state(r0, v0), period)
    rf, vf = sol.y[0:3, -1], sol.y[3:6, -1]
    el1 = elements(rf, vf)
    assert abs(el1["energy"] - el0["energy"]) / abs(el0["energy"]) < 1e-10
    assert abs(el1["h"] - el0["h"]) / el0["h"] < 1e-10
    assert np.linalg.norm(rf - r0) < 1.0      # orbit closes within 1 m after one period
    ts = np.linspace(0, period, 50)
    for t in ts:
        yy = sol.sol(t)
        assert np.linalg.norm(yy[0:3]) == pytest.approx(r, abs=0.5)


def test_j2_precesses_node():
    cfg = vacuum_cfg(j2=True)
    r = dyn.RE + 200e3
    vc = math.sqrt(dyn.MU / r)
    inc = math.radians(28.5)
    sol = coast(cfg, state((r, 0, 0), (0, vc * math.cos(inc), vc * math.sin(inc))), 86400.0, rtol=1e-10)
    rf, vf = sol.y[0:3, -1], sol.y[3:6, -1]
    h = np.cross(rf, vf)
    raan = math.degrees(math.atan2(h[0], -h[1]))
    # J2 nodal regression for 200 km, 28.5 deg: ~ -7.9 deg/day (first-order theory)
    n = math.sqrt(dyn.MU / r ** 3)
    expect = math.degrees(-1.5 * n * dyn.J2 * (dyn.RE / r) ** 2 * math.cos(inc)) * 86400
    assert raan == pytest.approx(expect, abs=0.3)


# ---------------------------------------------------------------- engine model
def test_thrust_vs_ambient_pressure():
    cfg = load_config()
    e = cfg["engines"]["stage1"]
    f_vac, md = engine_thrust(e, 1.0, 0.0)
    f_sl, md2 = engine_thrust(e, 1.0, 101325.0)
    assert md == md2
    assert f_vac - f_sl == pytest.approx(e["count"] * e["ae_m2"] * 101325.0, rel=1e-12)
    isp_vac = f_vac / md / G0
    isp_sl = f_sl / md / G0
    assert 305 < isp_vac < 320
    assert 275 < isp_sl < 290
    # throttled: exit pressure scales with throttle -> vacuum Isp unchanged
    f_half, md_half = engine_thrust(e, 0.5, 0.0)
    assert f_half / md_half == pytest.approx(f_vac / md, rel=1e-12)


def test_tsiolkovsky_matches_integrated(mission):
    """Integrate |F|/m over the whole S2 burn and compare with c*ln(m0/m1)."""
    cfg = mission["cfg"]
    e = cfg["engines"]["stage2"]
    c = e["ve_m_s"] + e["pe_pa"] * e["ae_m2"] / e["mdot0_kg_s"]
    ev = dict(mission["events"])
    seg = [s for s in mission["segments"] if s[3].phase == dyn.S2_BURN]
    dv = 0.0
    dv_rocket = 0.0
    for sol, a, b, ctx in seg:
        ts = np.linspace(a, b, 20001)
        acc = []
        for t in ts:
            _, tel = dyn.evaluate(t, sol.sol(t), ctx, cfg, aux=True)
            acc.append(tel["thrust"] / tel["mass"])
        dv += np.trapezoid(acc, ts)
        m0 = mass_props(cfg, 0, sol.sol(a)[14], 2, ctx.fairing).m
        m1 = mass_props(cfg, 0, sol.sol(b)[14], 2, ctx.fairing).m
        dv_rocket += c * math.log(m0 / m1)
    assert dv > 5000
    assert dv == pytest.approx(dv_rocket, rel=1e-6)


# ---------------------------------------------------------------- staging & mission
def test_staging_mass_bookkeeping(mission):
    cfg = mission["cfg"]
    ms = cfg["mass"]
    s = mission["samples"]
    liftoff = ms["stage1_dry_kg"] + ms["stage1_prop_kg"] + ms["stage2_dry_kg"] + ms["stage2_prop_kg"] + ms["payload_kg"] + ms["fairing_kg"]
    assert s[0]["mass"] == pytest.approx(liftoff)
    ev = dict(mission["events"])
    pre = [x for x in s if x["phase"] == dyn.COAST_SEP][-1]
    post = [x for x in s if x["phase"] == dyn.COAST_S2][0]
    assert pre["mass"] - post["mass"] == pytest.approx(ms["stage1_dry_kg"], abs=1e-6)
    assert post["mass"] == pytest.approx(ms["stage2_dry_kg"] + ms["stage2_prop_kg"] + ms["payload_kg"] + ms["fairing_kg"], abs=1e-6)
    end = s[-1]
    assert end["mass"] == pytest.approx(ms["stage2_dry_kg"] + ms["payload_kg"] + mission["final_state"][14], abs=1e-6)
    assert "FAIRING_SEP" in ev
    # propellant consumed equals integral of mdot
    assert ev["MECO"] == pytest.approx(ms["stage1_prop_kg"] / (9 * 298.0) + 3.0, abs=8.0)


def test_cg_moves_and_inertia_drops():
    cfg = load_config()
    full = mass_props(cfg, cfg["mass"]["stage1_prop_kg"], cfg["mass"]["stage2_prop_kg"], 1, True)
    low = mass_props(cfg, cfg["mass"]["stage1_prop_kg"] * 0.05, cfg["mass"]["stage2_prop_kg"], 1, True)
    # propellant settles aft and drains from the top: CG first moves aft, then forward near burnout
    assert low.x_cg > full.x_cg
    assert low.iyy < full.iyy
    s2 = mass_props(cfg, 0, cfg["mass"]["stage2_prop_kg"], 2, False)
    assert s2.x_cg > 45.0 and s2.m == pytest.approx(4000 + 92670 + 15000)


def test_mission_reaches_orbit(mission):
    names = [e[0] for e in mission["events"]]
    assert names[:6] == ["IGNITION", "LIFTOFF", "MECO", "STAGE_SEP", "FAIRING_SEP", "SES"] or \
        names[:6] == ["IGNITION", "LIFTOFF", "MECO", "STAGE_SEP", "SES", "FAIRING_SEP"]
    assert "SECO" in names
    el = mission["elements_seco"]
    assert orbit_check(el)
    assert 180e3 < el["periapsis_alt"] < 210e3
    assert 190e3 < el["apoapsis_alt"] < 220e3
    assert el["i_deg"] == pytest.approx(28.5, abs=0.6)
    # orbit stays (elements after coast within drag/J2 tolerance)
    assert mission["elements_end"]["periapsis_alt"] > 170e3


def test_max_q_reasonable(mission):
    mq = mission["max_q"]
    assert 20e3 < mq["qbar"] < 40e3
    assert 0.8 < mq["mach"] < 2.0


def test_g_limit_respected(mission):
    assert max(s["g_load"] for s in mission["samples"] if s["phase"] in (dyn.S1_BURN, dyn.S2_BURN)) < 4.5 + 0.05
