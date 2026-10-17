"""Authoritative mission simulation using scipy.integrate.solve_ivp with exact event location."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from . import dynamics as dyn
from .dynamics import Ctx, evaluate, initial_state, MU, RE
from .orbit import elements
from .vehicle import load_config


def _run_phase(cfg, ctx, t0, t1, y0, events=(), rtol=1e-9, atol=None, max_step=0.5, method="DOP853"):
    if atol is None:
        atol = np.array([1e-4] * 3 + [1e-7] * 3 + [1e-11] * 4 + [1e-11] * 3 + [1e-5] * 2)

    def f(t, y):
        return evaluate(t, y, ctx, cfg)

    evs = []
    for e in events:
        def ev(t, y, e=e):
            return e(t, y)
        ev.terminal = True
        ev.direction = getattr(e, "direction", 0)
        evs.append(ev)
    sol = solve_ivp(f, (t0, t1), y0, method=method, rtol=rtol, atol=atol, events=evs or None,
                    dense_output=True, max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    hit = None
    if evs:
        for k, te in enumerate(sol.t_events):
            if len(te):
                hit = k
    return sol, hit


def _ev(fn, direction=0):
    fn.direction = direction
    return fn


def run_mission(cfg=None, t_orbit_coast=600.0, sample_dt=0.5, rtol=1e-9):
    """Simulate pad -> orbit. Returns dict with samples, events, final elements."""
    cfg = cfg or load_config()
    gd = cfg["guidance"]
    ctx = Ctx()
    y = np.array(initial_state(cfg), dtype=float)
    segments = []   # (sol, t0, t1, ctx snapshot)
    events = []
    t = -cfg["engines"]["stage1"]["startup_s"]

    def seg(t0, t1, evs=()):
        nonlocal y
        sol, hit = _run_phase(cfg, ctx, t0, t1, y, evs, rtol=rtol)
        tend = sol.t[-1]
        y = sol.y[:, -1].copy()
        segments.append((sol, t0, tend, Ctx(**vars(ctx))))
        return tend, hit

    # Pad hold-down: engine startup, clamped to the rotating ground
    events.append(("IGNITION", t))
    t, _ = seg(t, 0.0)
    ctx.phase = dyn.S1_BURN
    events.append(("LIFTOFF", t))
    # Stage 1 burn until propellant depletion
    ev_meco = _ev(lambda tt, yy: yy[13], -1)
    ev_crash = _ev(lambda tt, yy: (yy[0] ** 2 + yy[1] ** 2 + yy[2] ** 2) ** 0.5 - RE + 5.0, -1)
    t, hit = seg(t, 600.0, [ev_meco, ev_crash])
    if hit != 0:
        raise RuntimeError("Stage 1 did not reach MECO (hit=%s)" % hit)
    y[13] = 0.0
    ctx.t_meco = t
    events.append(("MECO", t))
    ctx.phase = dyn.COAST_SEP
    t, _ = seg(t, t + gd["sep_delay_s"])
    ctx.phase = dyn.COAST_S2
    ctx.t_sep = t
    events.append(("STAGE_SEP", t))
    ev_fair = _ev(lambda tt, yy: (yy[0] ** 2 + yy[1] ** 2 + yy[2] ** 2) ** 0.5 - RE - gd["fairing_alt_m"], 1)
    tses = t + gd["ses_delay_s"]
    while True:
        evs = [ev_fair] if ctx.fairing else []
        t, hit = seg(t, tses, evs)
        if hit == 0 and ctx.fairing:
            ctx.fairing = False
            events.append(("FAIRING_SEP", t))
            continue
        break
    ctx.phase = dyn.S2_BURN
    ctx.t_ses = t
    events.append(("SES", t))
    rt = RE + gd["target_alt_m"]
    eps_t = -MU / (2 * rt)
    ev_mp2 = _ev(lambda tt, yy: yy[14], -1)
    ev_seco = _ev(lambda tt, yy: 0.5 * (yy[3] ** 2 + yy[4] ** 2 + yy[5] ** 2) - MU / (yy[0] ** 2 + yy[1] ** 2 + yy[2] ** 2) ** 0.5 - eps_t, 1)
    while True:
        evs = [ev_mp2, ev_seco, ev_crash] + ([ev_fair] if ctx.fairing else [])
        t, hit = seg(t, t + 900.0, evs)
        if hit == 3:
            ctx.fairing = False
            events.append(("FAIRING_SEP", t))
            continue
        break
    if hit == 0:
        ctx.phase = dyn.S2_DEPLETED
        events.append(("S2_DEPLETION", t))
    elif hit == 1:
        ctx.phase = dyn.ORBIT
        events.append(("SECO", t))
    else:
        raise RuntimeError("S2 burn ended unexpectedly (hit=%s)" % hit)
    ctx.t_seco = t
    el_seco = elements(y[0:3], y[3:6])
    t, _ = seg(t, t + t_orbit_coast)
    el_end = elements(y[0:3], y[3:6])

    # --------- sample output -----------
    samples = []
    t_first = segments[0][1]
    for sol, a, b, c in segments:
        ts = np.arange(math.ceil(a / sample_dt) * sample_dt, b, sample_dt)
        ts = np.concatenate([[a], ts[ts > a + 1e-9], [b]])
        for tt in ts:
            yy = sol.sol(tt)
            _, tel = evaluate(tt, yy, c, cfg, aux=True)
            tel["y"] = [float(x) for x in yy]
            tel["fairing"] = c.fairing
            samples.append(tel)
    max_q = max(samples, key=lambda s: s["qbar"] if s["phase"] <= dyn.S2_BURN else 0)
    return dict(cfg=cfg, samples=samples, events=events, elements_seco=el_seco, elements_end=el_end,
                max_q=dict(t=max_q["t"], qbar=max_q["qbar"], h=max_q["h"], mach=max_q["mach"]),
                final_state=[float(x) for x in y], segments=segments, t_first=t_first)


def orbit_check(el, min_periapsis_alt=150e3, max_e=0.02):
    return bool(el["periapsis_alt"] > min_periapsis_alt and el["e"] < max_e)
