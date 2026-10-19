"""Reference renderer: a full launch, as heard by one listener, from the acoustic profile + assets + a trajectory.

This is the ground truth the browser and Unreal front-ends should match. It uses only
data/audio/acoustic_profile.json, data/audio/assets/*.wav and telemetry (the Python reference sim, or a CSV
exported from it / from a runtime with the same columns).

  python tools/audio/render_launch.py --listener pad   --out pad.wav
  python tools/audio/render_launch.py --pos 3536,-3536,1.7 --t1 220 --out obs5km.wav
  python tools/audio/render_launch.py --mode cabin --t1 200 --out cabin.wav
  python tools/audio/render_launch.py --export-telemetry telemetry.csv

Pipeline (ground listener):
  1. emission-time grid tau (sim samples, 0.05 s): source state (W, U, D, crackle weight), nozzle position and
     plume axis in pad-local ENU, per-1/3-octave SPL at the listener (spreading, directivity, rho*c, ISO 9613-1
     absorption along the ray, ground reflection), delay d/c_path.
  2. reception time t_r(tau) = tau + d(tau)/c(tau) (monotonic) is inverted per audio sample, and every loop is
     READ at its emission time tau(t_r): this variable delay line is the only Doppler (no extra pitch shift).
  3. layer EQ per STFT frame: EQ_dB[b] = D[b] - A_group[b] (digital target minus asset band level).
  4. one-shots are placed at event time + delay with a static EQ; a look-ahead peak limiter (profile) finishes the mix.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "physics"))
import acoustic_model as am  # noqa: E402
from aether_physics.dynamics import quat_to_mat, OMEGA_E  # noqa: E402

np.seterr(divide="ignore")  # 10*log10(0) -> -inf is intended for silent bands

FS = 48000
ASSETS = os.path.join(ROOT, "data", "audio")
F_LO, F_HI, F_C = am.third_octave_bands(6.3, 16000)
PRESETS = {
    "pad": dict(mode="ground", pos=(310.0, -260.0, 1.7)),           # Aether 'Pad' camera, 405 m
    "near": dict(mode="ground", pos=(-1061.0, 1061.0, 1.7)),        # 1.5 km NW (typical close press site)
    "tracker": dict(mode="ground", pos=(-1800.0, -2600.0, 2.0)),    # Aether 'Tracker'
    "obs5km": dict(mode="ground", pos=(-3536.0, 3536.0, 1.7)),      # 5 km NW (uprange-side public viewing)
    "obs20km": dict(mode="ground", pos=(-14142.0, 14142.0, 1.7)),   # 20 km NW
    "onboard": dict(mode="onboard"),
    "cabin": dict(mode="cabin"),
}
TEL_COLS = ["t", "phase", "thrust", "mdot", "throttle", "P", "T", "rho", "a", "h", "mach", "qbar",
            "noz_e", "noz_n", "noz_u", "ax_e", "ax_n", "ax_u"]


# ------------------------------------------------------------------ trajectory
def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def sim_telemetry(dt=0.05, t_end=480.0):
    """Run the authoritative Python sim and return telemetry arrays with nozzle/axis in pad-local ENU."""
    from aether_physics.sim import run_mission
    r = run_mission(t_orbit_coast=max(5.0, t_end - 470.0), sample_dt=dt)
    S = r["samples"]
    t_first = r["t_first"]
    p0 = np.array(S[0]["y"][0:3])
    out = {k: [] for k in TEL_COLS}
    for s in S:
        t = s["t"]
        Rz = _rotz(OMEGA_E * (t - t_first))
        pad = Rz @ p0
        up = pad / np.linalg.norm(pad)
        east = np.cross([0, 0, 1.0], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        B = np.vstack([east, north, up])  # rows: ENU basis in ECI
        R = np.array(quat_to_mat(s["y"][6:10]))
        bx = R[:, 0]
        stage2 = s["phase"] > 2
        noz = np.array(s["y"][0:3]) + bx * (45.0 if stage2 else 0.0)
        d = B @ (noz - pad)
        ax = B @ (-bx)
        for k, v in zip(TEL_COLS, [t, s["phase"], s["thrust"], s["mdot"], s["throttle"], s["P"], s["T"], s["rho"], s["a"],
                                   s["h"], s["mach"], s["qbar"], d[0], d[1], d[2], ax[0], ax[1], ax[2]]):
            out[k].append(float(v))
    tel = {k: np.array(v) for k, v in out.items()}
    # de-duplicate segment-boundary samples (keep the later one, which carries the new phase)
    _, idx = np.unique(tel["t"][::-1], return_index=True)
    idx = len(tel["t"]) - 1 - idx
    tel = {k: v[np.sort(idx)] for k, v in tel.items()}
    events = [(n, float(t)) for n, t in r["events"]]
    return tel, events


def export_csv(tel, events, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# aether telemetry for tools/audio/render_launch.py; ENU metres relative to the pad; events: "
                 + ";".join(f"{n}={t:.3f}" for n, t in events) + "\n")
        w = csv.writer(fh)
        w.writerow(TEL_COLS)
        for i in range(len(tel["t"])):
            w.writerow([f"{tel[k][i]:.6g}" for k in TEL_COLS])


def load_csv(path):
    events = []
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
        if first.startswith("#") and "events:" in first:
            for kv in first.split("events:")[1].strip().split(";"):
                n, t = kv.split("=")
                events.append((n, float(t)))
        rows = list(csv.DictReader(fh))
    tel = {k: np.array([float(r[k]) for r in rows]) for k in TEL_COLS}
    return tel, events


# ------------------------------------------------------------------ assets
def load_assets(prof):
    lay = {}
    for e in prof["layers"] + prof["oneshots"]:
        fs, x = wavfile.read(os.path.join(ASSETS, e["file"]))
        assert fs == FS
        lay[e["name"]] = dict(x=x.astype(np.float64) / 32768.0, meta=e)
    return lay


def asset_third_db(meta):
    """Asset band levels on the renderer's band grid (6.3 Hz..16 kHz)."""
    hz = np.array(meta.get("third_oct_hz") or meta["octave_hz"])
    db = np.array(meta.get("third_oct_db") or meta["octave_db"])
    if "third_oct_hz" not in meta:  # octave data: convert to per-1/3-octave power
        db = db - 10 * math.log10(3)
    return np.interp(np.log(F_C), np.log(hz), db)


def group_db(*metas):
    return 10 * np.log10(sum(10 ** (asset_third_db(m) / 10) for m in metas))


# ------------------------------------------------------------------ DSP helpers
def read_loop(x, pos):
    """Cubic (Catmull-Rom) read of a circular buffer at fractional sample positions."""
    n = len(x)
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    xm1, x0, x1, x2 = x[(i - 1) % n], x[i % n], x[(i + 1) % n], x[(i + 2) % n]
    return x0 + 0.5 * f * (x1 - xm1 + f * (2 * xm1 - 5 * x0 + 4 * x1 - x2 + f * (3 * (x0 - x1) + x2 - xm1)))


def read_loop_ar(x, pos, prof):
    """Anti-repetition read: two voices of the same loop, offset by a golden-ratio fraction of its length, mixed with
    an equal-power crossfade whose phase moves with an incommensurate period (profile.anti_repetition). The two voices
    are uncorrelated noise, so cos^2 + sin^2 = 1 keeps the level constant; the combined pattern does not repeat for
    loop_len x cycle seconds. All layers of a sync group must use the same pos -> they stay sample-aligned."""
    ar = prof["anti_repetition"]
    n = len(x)
    t = pos / FS
    phi = math.pi / 4 + (math.pi / 4) * np.sin(2 * math.pi * t / ar["crossfade_cycle_s"])
    return np.cos(phi) * read_loop(x, pos) + np.sin(phi) * read_loop(x, pos + ar["voice_offset_fraction"] * n)


def tv_eq(x, frame_t, gains_db, nper=8192, hop=2048):
    """Time-varying zero-phase EQ by STFT (sqrt-Hann analysis/synthesis, 75 % overlap).
    gains_db: [n_frames_in, n_bands] on F_C at times frame_t (seconds from x[0])."""
    win = np.sqrt(np.hanning(nper + 1)[:-1])
    n = len(x)
    pad = nper
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad + hop)])
    nfr = 1 + (len(xp) - nper) // hop
    f = np.fft.rfftfreq(nper, 1.0 / FS)
    lf = np.log(np.clip(f, F_C[0], F_C[-1]))
    lfc = np.log(F_C)
    y = np.zeros(len(xp))
    norm = np.zeros(len(xp))
    centres = (np.arange(nfr) * hop + nper / 2 - pad) / FS
    gi = np.empty((nfr, len(F_C)))
    for b in range(len(F_C)):
        gi[:, b] = np.interp(centres, frame_t, gains_db[:, b], left=-200.0, right=-200.0)
    chunk = 256
    for c0 in range(0, nfr, chunk):
        c1 = min(nfr, c0 + chunk)
        idx = (np.arange(c0, c1) * hop)[:, None] + np.arange(nper)[None, :]
        F = np.fft.rfft(xp[idx] * win, axis=1)
        G = np.empty((c1 - c0, len(f)))
        for k in range(c1 - c0):
            G[k] = np.interp(lf, lfc, gi[c0 + k])
        F *= 10 ** (G / 20)
        seg = np.fft.irfft(F, nper, axis=1) * win
        for k in range(c1 - c0):
            s = (c0 + k) * hop
            y[s:s + nper] += seg[k]
            norm[s:s + nper] += win * win
    y = y / np.maximum(norm, 1e-9)
    return y[pad:pad + n]


def lookahead_limit(x, ceiling_dbfs=-1.0, lookahead_ms=5.0, release_ms=150.0, fs=FS, return_gr=False):
    """Feed-forward look-ahead peak limiter (profile level_mapping.limiter.recipe). Offline form: the gain computed
    from x is applied to x itself, which equals the real-time form (gain on the input, applied to the input delayed
    by L samples) shifted by L samples."""
    from scipy.ndimage import maximum_filter1d
    x = np.asarray(x, dtype=np.float64)
    L = max(1, int(round(lookahead_ms * fs / 1000.0)))
    a = np.abs(x)
    gr_req = np.maximum(0.0, 20 * np.log10(np.maximum(a, 1e-12)) - ceiling_dbfs)
    # look-ahead hold: max over [n, n+L]
    gr_h = maximum_filter1d(gr_req, size=L + 1, origin=-(L // 2), mode="constant", cval=0.0)
    # exponential release of the dB reduction (block-wise max-plus trick, exact)
    dec = math.exp(-1.0 / (release_ms * fs / 1000.0))
    gr_r = np.empty_like(gr_h)
    prev = 0.0
    B = 32768
    for i0 in range(0, len(gr_h), B):
        seg = gr_h[i0:i0 + B]
        e = dec ** -(np.arange(len(seg)) + 1.0)
        v = np.maximum.accumulate(np.maximum(seg * e, prev))
        out = v / e
        gr_r[i0:i0 + B] = out
        prev = out[-1]
    # attack: trailing moving average over L samples (linear ramp; >= gr_req at every peak)
    c = np.concatenate([np.zeros(L), np.cumsum(np.concatenate([np.zeros(1), gr_r]))])
    gr = (c[L + 1:] - c[1:-L]) / L
    gr = np.maximum(gr, gr_req)  # numerical guard (the hold+average already guarantees this)
    y = x * 10 ** (-gr / 20)
    lim = 10 ** (ceiling_dbfs / 20)
    y = np.clip(y, -lim, lim)
    return (y, gr) if return_gr else y


# ------------------------------------------------------------------ model on the emission grid
def model_grid(prof, veh, tel, lis, mode, trench_dir=(0.0, 1.0)):
    """Per emission sample: jet/pad/crackle band SPL at the listener and delays."""
    n = len(tel["t"])
    nb = len(F_C)
    out = dict(L_jet=np.full((n, nb), -200.0), L_pad=np.full((n, nb), -200.0), w_c=np.zeros(n),
               delay=np.zeros(n), delay_pad=np.zeros(n), abs1k_minus=np.zeros((n, nb)), theta=np.zeros(n), r=np.zeros(n),
               Lw=np.full(n, np.nan), f_peak=np.full(n, np.nan), w_pad=np.zeros(n), w_air=np.zeros(n), w_aero=np.zeros(n),
               struct_db=np.full(n, -200.0))
    pad_src = np.array([trench_dir[0] * 60.0, trench_dir[1] * 60.0, 1.0])
    lis = np.asarray(lis, float)
    pad_meta = prof["_pad_shape_db"]
    pa_level = prof["pad_interaction"]["level_db_re_jet"]
    for i in range(n):
        t = {k: tel[k][i] for k in TEL_COLS}
        noz = np.array([t["noz_e"], t["noz_n"], t["noz_u"]])
        ax = np.array([t["ax_e"], t["ax_n"], t["ax_u"]])
        st = am.source_state(prof, veh, t)
        if mode in ("onboard", "cabin"):
            # listener rides at the payload, 55 m forward of the active nozzle
            lis_i = noz - ax * 55.0
        else:
            lis_i = lis
        if st is None:
            out["delay"][i] = np.linalg.norm(noz - lis_i) / 340.0
            out["delay_pad"][i] = np.linalg.norm(pad_src - lis_i) / 340.0
            continue
        hz = max(t["noz_u"], 0.0)
        w_pad = am.pad_weight(hz, st["De"]) if t["phase"] <= 2 else 0.0
        out["w_pad"][i] = w_pad
        L, g = am.band_spl(prof, st, noz, ax, lis_i, F_LO, F_HI, F_C, ground=mode not in ("onboard", "cabin"),
                           absorb=True)
        jet_scale = 20 * math.log10(max(1 - 0.5 * w_pad, 1e-3))
        out["L_jet"][i] = L + jet_scale
        out["delay"][i] = g["delay"]
        out["theta"][i], out["r"][i] = g["theta"], g["r"]
        out["Lw"][i], out["f_peak"][i] = st["Lw"], st["f_peak"]
        out["w_c"][i] = am.crackle_weight(prof, st["w_crackle"], g["theta"], g["r"], st["xt"])
        f_ref = prof["propagation"]["atmospheric_absorption"]["crackle_band_policy"]["evaluate_at_hz"]
        beta = prof["propagation"]["atmospheric_absorption"]["crackle_band_policy"]["exponent"]
        f_eff = np.where(F_C > f_ref, f_ref * (F_C / f_ref) ** beta, F_C)
        A = am.path_absorption_db(F_C, g["src"], lis_i, prof)
        A_eff = am.path_absorption_db(f_eff, g["src"], lis_i, prof)
        r_nl = prof["propagation"]["atmospheric_absorption"]["crackle_band_policy"]["nonlinear_range_m"]
        fr = min(1.0, r_nl / max(g["r"], 1.0))  # reduced absorption only over the first r_nl metres (old-age region beyond)
        out["abs1k_minus"][i] = (A - A_eff) * fr  # crackle: absorption at f_ref*(f/f_ref)^beta instead of f, near part
        if w_pad > 1e-4:
            r = float(np.linalg.norm(lis_i - pad_src))
            r_eff = math.sqrt(r * r + st["xt"] ** 2)
            Lp = st["Lw"] + pa_level + pad_meta - 10 * math.log10(2 * math.pi * r_eff ** 2) + am.listener_impedance_db(lis_i[2])
            Lp = Lp - am.path_absorption_db(F_C, pad_src, lis_i, prof) + 10 * math.log10(w_pad)
            out["L_pad"][i] = Lp
            out["delay_pad"][i] = r / am.c_path(pad_src, lis_i)
        else:
            out["delay_pad"][i] = np.linalg.norm(pad_src - lis_i) / 340.0
        if mode in ("onboard", "cabin"):
            out["w_air"][i] = am.onboard_air_weight(t["mach"], t["rho"], t["a"])
            out["w_aero"][i] = am.aero_weight(t["qbar"], t["mach"])
            out["struct_db"][i] = 20 * math.log10(max(t["thrust"], 1.0) / prof["onboard"]["structure_borne"]["thrust_ref_stage1_N"])
    return out


# ------------------------------------------------------------------ render
def render(listener="pad", pos=None, mode=None, t0=-6.0, t1=None, k=None, tel=None, events=None, prof=None,
           out_path=None, verbose=True, return_debug=False):
    prof = prof or am.load_profile()
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    if tel is None:
        tel, events = sim_telemetry()
    ev = dict(events)
    pre = PRESETS.get(listener, {})
    mode = mode or pre.get("mode", "ground")
    pos = np.array(pos if pos is not None else pre.get("pos", (0, 0, 1.7)), float)
    lay = load_assets(prof)
    A_jet = group_db(lay["rumble_lf"]["meta"], lay["jet_mix"]["meta"])
    A_pad = asset_third_db(lay["pad_interaction"]["meta"])
    prof["_pad_shape_db"] = A_pad - 10 * math.log10(np.sum(10 ** (A_pad / 10)))
    grid = model_grid(prof, veh, tel, pos, mode)
    tau = tel["t"]
    if t1 is None:
        t1 = float(ev.get("MECO", 150.0) + (grid["delay"][np.searchsorted(tau, ev.get("MECO", 150.0))] if mode == "ground" else 0) + 25.0)
        t1 = min(t1, 420.0) if mode == "ground" else 200.0
    n_out = int((t1 - t0) * FS)
    t_r = t0 + np.arange(n_out) / FS
    # ---- reception -> emission mapping (variable delay = Doppler)
    if mode == "ground":
        tr_grid = tau + grid["delay"]
        tr_grid = np.maximum.accumulate(tr_grid)  # guard (never needed for ascent; keeps interp valid)
        tau_r = np.interp(t_r, tr_grid, tau, left=tau[0] - 1.0)
        tau_pad = t_r - np.interp(t_r, tau + grid["delay_pad"], grid["delay_pad"])
    else:
        tau_r = t_r.copy()
        tau_pad = t_r.copy()
    # ---- digital level mapping (mic gain from the predicted maximum OASPL)
    lm = prof["level_mapping"]
    k = lm["k_default"] if k is None else k

    def oa(Lb):
        return 10 * np.log10(np.sum(10 ** (Lb / 10), axis=-1) + 1e-30)

    if mode == "ground":
        L_tot = 10 * np.log10(10 ** (grid["L_jet"] / 10) + 10 ** (grid["L_pad"] / 10))
        Lp_max = float(np.max(oa(L_tot)))
    else:
        w_air_db = 10 * np.log10(np.maximum(grid["w_air"], 1e-12))
        L_ext = 10 * np.log10(10 ** (grid["L_jet"] / 10) + 10 ** (grid["L_pad"] / 10)) + w_air_db[:, None]
        if mode == "cabin":
            L_ext = L_ext - am.cabin_tl_db(F_C, prof["onboard"]["cabin_transmission_loss"]["m_s_kg_m2"])[None, :]
        Lp_max = float(np.max(oa(L_ext)))
    target = lm["target_peak_dbfs"]

    def to_digital(Lb):
        L_oa = oa(Lb)[..., None]
        return Lb - L_oa + target + k * (L_oa - Lp_max)

    y = np.zeros(n_out)
    frame_t = tau  # gains are defined on the emission grid; mapped to reception below
    # reception times of each emission sample (for the EQ trajectories)
    rt_jet = tau + (grid["delay"] if mode == "ground" else 0.0)
    rt_pad = tau + (grid["delay_pad"] if mode == "ground" else 0.0)
    if mode == "ground":
        D_jet = to_digital(grid["L_jet"])
        D_pad = to_digital(grid["L_pad"])
        D_jet[grid["L_jet"][:, 0] < -150] = -200.0
        D_pad[grid["L_pad"][:, 0] < -150] = -200.0
        G_jet = D_jet - A_jet[None, :]
        G_crk = G_jet + 20 * np.log10(np.maximum(grid["w_c"], 1e-6))[:, None] + grid["abs1k_minus"]  # (dB) undo HF absorption
        G_pad = D_pad - A_pad[None, :]
        pos_src = (tau_r - tau[0]) * FS  # loop read positions in source time
        for name, G, rt in (("rumble_lf", G_jet, rt_jet), ("jet_mix", G_jet, rt_jet), ("crackle", G_crk, rt_jet)):
            sig = read_loop_ar(lay[name]["x"], pos_src, prof)
            sig[tau_r < tau[0]] = 0.0
            y += tv_eq(sig, rt - t0, G)
        sigp = read_loop_ar(lay["pad_interaction"]["x"], (tau_pad - tau[0]) * FS, prof)
        y += tv_eq(sigp, rt_pad - t0, G_pad)
    else:
        L_jet_ext = grid["L_jet"] + w_air_db[:, None]
        L_pad_ext = grid["L_pad"] + w_air_db[:, None]
        if mode == "cabin":
            tl = am.cabin_tl_db(F_C, prof["onboard"]["cabin_transmission_loss"]["m_s_kg_m2"])[None, :]
            L_jet_ext, L_pad_ext = L_jet_ext - tl, L_pad_ext - tl
        G_jet = to_digital(L_jet_ext) - A_jet[None, :]
        G_jet[L_jet_ext[:, 0] < -150] = -200
        G_crk = G_jet + 20 * np.log10(np.maximum(grid["w_c"], 1e-6))[:, None]
        G_pad = to_digital(L_pad_ext) - A_pad[None, :]
        G_pad[L_pad_ext[:, 0] < -150] = -200
        pos_src = (t_r - tau[0]) * FS
        for name, G in (("rumble_lf", G_jet), ("jet_mix", G_jet), ("crackle", G_crk), ("pad_interaction", G_pad)):
            sig = read_loop_ar(lay[name]["x"], pos_src, prof)
            y += tv_eq(sig, tau - t0, G)
        # structure-borne and aero (ARTISTIC reference levels; flat EQ, cabin slightly darker)
        dl = prof["onboard"]["digital_reference_levels_dbfs"][mode]
        ref_struct, ref_aero = dl["structure_at_stage1_full_thrust"], dl["aero_at_w_aero_1"]
        A_s = asset_third_db(lay["onboard_structure"]["meta"])
        A_a = asset_third_db(lay["aero_buffet"]["meta"])
        shape_s = A_s - oa(A_s)
        shape_a = A_a - oa(A_a)
        if mode == "cabin":
            shape_a = shape_a - am.cabin_tl_db(F_C, 20.0) * 0.5
        G_s = (ref_struct + grid["struct_db"])[:, None] + shape_s[None, :] - A_s[None, :]
        G_a = (ref_aero + 20 * np.log10(np.maximum(grid["w_aero"], 1e-6)))[:, None] + shape_a[None, :] - A_a[None, :]
        y += tv_eq(read_loop_ar(lay["onboard_structure"]["x"], pos_src, prof), tau - t0, G_s)
        y += tv_eq(read_loop_ar(lay["aero_buffet"]["x"], pos_src + 7.3 * FS, prof), tau - t0, G_a)
    # ---- one-shots
    shots = []

    def place(name, t_emit, delay, G_static):
        x = lay[name]["x"]
        if G_static is not None:
            x = tv_eq(x, np.array([0.0, len(x) / FS]), np.vstack([G_static, G_static]))
        s0 = int(round((t_emit + delay - t0) * FS))
        if s0 >= n_out or s0 + len(x) <= 0:
            return
        a, b = max(0, s0), min(n_out, s0 + len(x))
        y[a:b] += x[a - s0:b - s0]
        shots.append((name, t_emit + delay))

    i_lo = int(np.searchsorted(tau, 0.0))  # liftoff: full-thrust state for the ignition EQ
    if "IGNITION" in ev:
        ti = ev["IGNITION"]
        if mode == "ground":
            Dref = to_digital(grid["L_jet"][i_lo]) - A_jet  # ignition one-shot at full-thrust jet level, pad geometry
            G1 = Dref + (A_jet.max() - asset_third_db(lay["ignition"]["meta"]).max()) * 0  # keep asset-relative balance
            place("ignition", ti, grid["delay_pad"][i_lo], G1 - 2.0)
        else:
            Dref = to_digital(grid["L_jet"][i_lo] + w_air_db[i_lo]) - A_jet
            place("ignition", ti, 0.0, Dref - (am.cabin_tl_db(F_C, 20.0) if mode == "cabin" else 0) - 2.0)
        # ignition overpressure: separate one-shot, scalar gain = oa(D_jet at liftoff) + level - asset main RMS
        if "ignition_overpressure" in lay:
            iop = prof["pad_interaction"]["ignition_overpressure"]
            meta = lay["ignition_overpressure"]["meta"]
            if mode == "ground":
                D_lo = to_digital(grid["L_jet"][i_lo])
                d_iop = grid["delay_pad"][i_lo]
            else:
                D_lo = to_digital(grid["L_jet"][i_lo] + w_air_db[i_lo])
                d_iop = 0.0
            g_iop = float(oa(D_lo)) + iop["level_db_re_full_thrust_jet"] - meta["rms_main_dbfs"]
            G_iop = np.full(len(F_C), g_iop) - (am.cabin_tl_db(F_C, 20.0) if mode == "cabin" else 0.0)
            place("ignition_overpressure", ti + meta["offset_s"], d_iop, G_iop)
    if "MECO" in ev:
        tm = ev["MECO"]
        im = max(0, int(np.searchsorted(tau, tm)) - 2)
        if mode == "ground":
            G = to_digital(grid["L_jet"][im]) - A_jet
            place("meco_tail", tm, grid["delay"][im], G)
        else:
            G = to_digital(grid["L_jet"][im] + w_air_db[im]) - A_jet
            place("meco_tail", tm, 0.0, np.maximum(G, -200))
            ref_struct = prof["onboard"]["digital_reference_levels_dbfs"][mode]["structure_at_stage1_full_thrust"]
            A_m = asset_third_db(lay["meco_tail"]["meta"])
            place("meco_tail", tm, 0.0, (ref_struct + grid["struct_db"][im] - 6.0) + (A_m - oa(A_m)) - A_m
                  - am.cabin_tl_db(F_C, 20.0) * 0.3)
    if "STAGE_SEP" in ev and mode in ("onboard", "cabin"):
        A_sp = asset_third_db(lay["stage_sep_structural"]["meta"])
        place("stage_sep_structural", ev["STAGE_SEP"], 0.0,
              prof["onboard"]["digital_reference_levels_dbfs"][mode]["stage_sep"] + (A_sp - oa(A_sp)) - A_sp)
    y = lookahead_limit(y, **{k: prof['level_mapping']['limiter'][k] for k in ('ceiling_dbfs', 'lookahead_ms', 'release_ms')})
    info = dict(listener=listener, mode=mode, pos=pos.tolist(), t0=t0, t1=t1, Lp_max_db=Lp_max, k=k,
                events=events, oneshots=shots, peak_dbfs=float(20 * np.log10(np.abs(y).max() + 1e-12)))
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        wavfile.write(out_path, FS, np.round(np.clip(y, -1, 1) * 32767).astype(np.int16))
        with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
    if verbose:
        print(f"rendered {listener} ({mode}) {t0}..{t1:.1f} s, predicted max OASPL {Lp_max:.1f} dB, peak {info['peak_dbfs']:.1f} dBFS")
    if return_debug:
        return y, info, grid, tel
    return y, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listener", default="pad", choices=sorted(PRESETS))
    ap.add_argument("--pos", help="e,n,u metres from the pad (overrides preset position)")
    ap.add_argument("--mode", choices=["ground", "onboard", "cabin"])
    ap.add_argument("--t0", type=float, default=-6.0)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--k", type=float, help="loudness compression (1 = physical)")
    ap.add_argument("--telemetry", help="CSV with columns " + ",".join(TEL_COLS))
    ap.add_argument("--export-telemetry", help="write the reference sim telemetry to this CSV and exit")
    ap.add_argument("--out", default="launch.wav")
    a = ap.parse_args()
    if a.export_telemetry:
        tel, ev = sim_telemetry()
        export_csv(tel, ev, a.export_telemetry)
        print("wrote", a.export_telemetry)
        return
    tel = ev = None
    if a.telemetry:
        tel, ev = load_csv(a.telemetry)
    pos = [float(v) for v in a.pos.split(",")] if a.pos else None
    render(a.listener, pos, a.mode, a.t0, a.t1, a.k, tel, ev, out_path=a.out)


if __name__ == "__main__":
    main()
