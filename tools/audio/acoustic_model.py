"""Reference evaluation of data/audio/acoustic_profile.json.

This is the executable form of the output contract: render_launch.py, validate.py and the tests use
only these functions (plus the asset files), so if a runtime reproduces these numbers it reproduces
the model. Pure numpy; atmosphere and plume ROM come from Aether's authoritative Python physics.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "physics"))

from aether_physics.atmosphere import atmosphere  # noqa: E402
from aether_physics.plume_rom import plume_state  # noqa: E402

PROFILE_PATH = os.path.join(ROOT, "data", "audio", "acoustic_profile.json")
P_REF = 101325.0
T0 = 293.15
T01 = 273.16
RHO0C0 = 413.3


def load_profile(path=PROFILE_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ ISO 9613-1
def molar_h2o_from_rh(rh_percent, T, P):
    psat_pr = 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    return rh_percent * psat_pr / (P / P_REF)


def iso9613_alpha(f, T, P, h):
    """Pure-tone atmospheric absorption coefficient [dB/m] (ISO 9613-1:1993 eqs. 3-5)."""
    f = np.asarray(f, dtype=float)
    pr = P / P_REF
    tr = T / T0
    frO = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pr * tr ** -0.5 * (9 + 280 * h * math.exp(-4.170 * (tr ** (-1 / 3) - 1)))
    return 8.686 * f * f * (1.84e-11 / pr * tr ** 0.5 + tr ** -2.5 * (
        0.01275 * math.exp(-2239.1 / T) / (frO + f * f / frO) + 0.1068 * math.exp(-3352.0 / T) / (frN + f * f / frN)))


def h2o_at(z, prof=None):
    rh0, Hs = 70.0, 2500.0
    if prof:
        hp = prof["propagation"]["atmospheric_absorption"]["humidity_profile"]
        rh0, Hs = hp["surface_RH_percent"], hp["h2o_scale_height_m"]
    T0g, P0g, _, _ = atmosphere(0.0)
    h0 = molar_h2o_from_rh(rh0, T0g, P0g)
    # molar fraction decays with scale height; never exceed saturation at altitude
    T, P, _, _ = atmosphere(max(z, 0.0))
    hsat = molar_h2o_from_rh(100.0, T, P)
    return float(min(h0 * math.exp(-max(z, 0.0) / Hs), hsat))


def path_samples(src, lis, n=24):
    s = np.linspace(0.0, 1.0, n)
    pts = np.asarray(lis)[None, :] + (np.asarray(src) - np.asarray(lis))[None, :] * s[:, None]
    return s, pts


def path_absorption_db(freqs, src, lis, prof=None, n=24):
    """Total absorption [dB] along the straight ray, ISO 9613-1 with local T, p, humidity."""
    freqs = np.asarray(freqs, dtype=float)
    r = float(np.linalg.norm(np.asarray(src) - np.asarray(lis)))
    if r < 1.0:
        return np.zeros_like(freqs)
    s, pts = path_samples(src, lis, n)
    acc = np.zeros((n, len(freqs)))
    for i, p in enumerate(pts):
        z = max(float(p[2]), 0.0)
        T, P, _, _ = atmosphere(z)
        acc[i] = iso9613_alpha(freqs, T, max(P, 1e-6), h2o_at(z, prof))
    return np.trapezoid(acc, s, axis=0) * r


def c_path(src, lis, n=16):
    """Harmonic mean of the sound speed along the ray (travel time = r / c_path)."""
    s, pts = path_samples(src, lis, n)
    inv = [1.0 / atmosphere(max(float(p[2]), 0.0))[3] for p in pts]
    return 1.0 / float(np.trapezoid(inv, s))


# ------------------------------------------------------------------ source
def stage_cfg(veh, phase):
    return veh["engines"]["stage1" if phase <= 2 else "stage2"]


def source_state(prof, veh, tel):
    """Source quantities at the emission time from runtime telemetry (keys as in the Python sim)."""
    eng = stage_cfg(veh, int(tel["phase"]))
    n = eng["count"]
    F, mdot, pa = float(tel["thrust"]), float(tel["mdot"]), float(tel["P"])
    if F <= 1.0 or mdot <= 1e-6:
        return None
    U = F / mdot
    src = prof["source"]
    eta = src["acoustic_efficiency"]["value"]
    co = src["source_continuum_cutoff"]
    w_src = float(np.clip((math.log10(max(pa, 1e-9)) - math.log10(co["pa_lo"])) /
                          (math.log10(co["pa_hi"]) - math.log10(co["pa_lo"])), 0, 1))
    W = eta * 0.5 * mdot * U * U * w_src
    de1 = 2 * math.sqrt(eng["ae_m2"] / math.pi)
    De = math.sqrt(n) * de1
    thr = max(float(tel.get("throttle", 1.0)), 0.05)
    ps = plume_state(eng["chamber_pressure_pa"], eng["pe_pa"], eng["ae_m2"], eng["gamma_exit"], pa, thr)
    D_spec = max(De, math.sqrt(n) * ps["dj"]) if pa < ps["pe"] else De
    Me = ps["Me"]
    xt = 3.45 * De * (1 + 0.38 * Me) ** 2  # cluster-equivalent diameter (merged flow, SP-8072 s2.1.2)
    T_amb, a_amb = float(tel["T"]), float(tel["a"])
    Mc = 0.65 * U / a_amb
    w_c = float(smoothstep((Mc - 1.5) / 2.0) * min(max(pa / P_REF, 0.0), 1.0) ** 0.5)
    return dict(U=U, W=W, Lw=10 * math.log10(max(W, 1e-12)) + 120, De=De, D_spec=D_spec, de1=de1, n=n, xt=xt,
                x_src=1.5 * xt, Me=Me, Mc=Mc, w_crackle=w_c, f_peak=0.02 * U / D_spec, pa=pa, w_src=w_src)


def spectrum_level_db(prof, st):
    """SP-8072 normalised spectrum level at Strouhal st; beyond the digitised range the end segments' slopes
    are continued (log-linear), never a step."""
    sp = prof["source"]["spectrum"]
    x, y = np.log(sp["strouhal"]), np.array(sp["level_db"])
    ls = np.log(np.asarray(st, dtype=float))
    L = np.interp(ls, x, y)
    lo_slope = (y[1] - y[0]) / (x[1] - x[0])
    hi_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    L = np.where(ls < x[0], y[0] + lo_slope * (ls - x[0]), L)
    L = np.where(ls > x[-1], y[-1] + hi_slope * (ls - x[-1]), L)
    return L


def crackle_weight(prof, w_source, theta_deg, r=0.0, xt=60.0):
    """Crackle layer LINEAR amplitude weight (multiplies the jet-group EQ gain).
    Crackle is Mach-wave radiation: strongest near the peak-directivity angle, weak near the plume axis; its HF
    content grows with propagation distance (nonlinear steepening, ROM)."""
    c = prof["crackle"]
    d = c["directivity"]
    w_dir = math.exp(-((theta_deg - d["peak_deg"]) / d["width_deg"]) ** 2)
    nl = c["nonlinear_growth"]
    g = 1.0 + nl["max_gain"] * min(1.0, math.log2(1.0 + math.log(max(r, xt) / xt)) / nl["log2_saturation"])
    return float(np.clip(w_source * c["reference_weight"] * w_dir * g, 0.0, c["max_weight"]))


def band_power_db(prof, Lw, U, D, f_lo, f_hi, n=8):
    """Sound-power level per band [dB re 1 pW] from the normalised SP-8072 spectrum."""
    out = []
    for a, b in zip(np.atleast_1d(f_lo), np.atleast_1d(f_hi)):
        st = np.geomspace(a * D / U, b * D / U, n)
        dens = 10 ** (spectrum_level_db(prof, st) / 10)
        frac = np.trapezoid(dens * st, np.log(st))
        out.append(Lw + 10 * math.log10(max(frac, 1e-30)))
    return np.array(out)


def directivity_db(prof, theta_deg, st):
    d = prof["directivity"]
    ks = np.array(d["strouhal"])
    tab = np.array([d["di_db"][str(k)] for k in d["strouhal"]])  # [nSt, nAngle]
    col = np.array([np.interp(theta_deg, d["angles_deg"], row) for row in tab])
    return np.interp(np.log(np.clip(st, ks[0], ks[-1])), np.log(ks), col)


def ground_gain_db(freqs, c, h_l, psi_rad, elevated=False):
    freqs = np.asarray(freqs, dtype=float)
    if elevated:
        return np.where(freqs < 40.0, 3.0, 0.0)
    dpath = max(2.0 * h_l * math.sin(max(psi_rad, 1e-3)), 1e-3)
    fc = c / (4.0 * dpath)
    x = np.clip(np.log(np.maximum(freqs, 1e-3) / fc) / math.log(4.0), 0.0, 1.0)
    return 6.0 - 3.0 * x


def pad_weight(h_nozzle, De):
    return float(math.exp(-max(h_nozzle, 0.0) / (10.0 * De)))


def onboard_air_weight(mach, rho, a):
    return float((1 - smoothstep((mach - 0.8) / 0.4)) * math.sqrt(max(rho * a, 0.0) / (1.225 * 340.3)))


def aero_weight(qbar, mach):
    return float((qbar / 30000.0) * (0.35 + 0.65 * math.exp(-((mach - 1.0) / 0.25) ** 2)))


def cabin_tl_db(freqs, m_s=20.0):
    return np.maximum(0.0, 20 * np.log10(m_s * np.maximum(np.asarray(freqs, float), 1.0)) - 47.0)


def listener_impedance_db(lis_alt):
    T, P, rho, a = atmosphere(max(lis_alt, 0.0))
    if P < 1.0:
        return -np.inf
    return 10 * math.log10(rho * a / RHO0C0)


def band_spl(prof, st_src, nozzle, axis, lis, f_lo, f_hi, f_c, ground=True, absorb=True, h_l=1.7):
    """SPL per band at the listener for the free-jet term. nozzle/lis: ENU metres; axis: exhaust-flow unit vector."""
    nozzle = np.asarray(nozzle, float)
    axis = np.asarray(axis, float) / (np.linalg.norm(axis) + 1e-12)
    # apparent source: x_src down the plume, but not below ground (the rest becomes the pad term)
    x = st_src["x_src"]
    if axis[2] < -1e-3 and nozzle[2] > 0:
        x = min(x, max(nozzle[2], 0.0) / -axis[2])
    elif nozzle[2] <= 0.5:
        x = 0.0
    src = nozzle + axis * x
    src[2] = max(src[2], 1.0)
    v = np.asarray(lis, float) - src
    r = float(np.linalg.norm(v))
    theta = math.degrees(math.acos(np.clip(np.dot(v / (r + 1e-9), axis), -1, 1)))
    r_eff = math.sqrt(r * r + st_src["xt"] ** 2)
    Lw_b = band_power_db(prof, st_src["Lw"], st_src["U"], st_src["D_spec"], f_lo, f_hi)
    DI = directivity_db(prof, theta, np.asarray(f_c) * st_src["D_spec"] / st_src["U"])
    L = Lw_b + DI - 10 * math.log10(4 * math.pi * r_eff ** 2) + listener_impedance_db(lis[2])
    c = c_path(src, lis)
    if absorb:
        L = L - path_absorption_db(f_c, src, lis, prof)
    if ground:
        psi = math.asin(np.clip((src[2] - lis[2]) / (r + 1e-9), 0, 1))
        L = L + ground_gain_db(f_c, c, h_l, psi, elevated=lis[2] > 30)
    return L, dict(r=r, theta=theta, src=src, c=c, delay=r / c)


def third_octave_bands(fmin=6.3, fmax=16000):
    k = np.arange(-22, 13)  # 6.3 Hz .. 16 kHz
    fc = 1000.0 * 10 ** (k / 10.0)
    fc = fc[(fc >= fmin * 0.99) & (fc <= fmax * 1.01)]
    return fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6), fc
