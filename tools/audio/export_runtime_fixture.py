"""Reference numbers for the browser runtime's unit tests (frontend/tests/audio.test.ts).

Evaluates the Python reference model (acoustic_model.py, render_launch.py) on the reference telemetry in
OCTAVE bands - the runtime grid - at a few emission samples and listeners, plus the reception -> emission
mapping and one-shot arrival times. Writes frontend/tests/fixtures/audio_model_ref.json.

  python tools/audio/export_runtime_fixture.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import acoustic_model as am  # noqa: E402
import render_launch as rl  # noqa: E402

OUT = os.path.join(ROOT, "frontend", "tests", "fixtures", "audio_model_ref.json")
LISTENERS = {"pad": (310.0, -260.0, 1.7), "tracker": (-1800.0, -2600.0, 2.0), "aerial": (1400.0, 1700.0, 650.0),
             "obs5km": (-3536.0, 3536.0, 1.7), "obs20km": (-14142.0, 14142.0, 1.7)}
TIMES = [0.5, 5.0, 20.0, 45.0, 80.0, 120.0, 150.0]


def main():
    prof = am.load_profile()
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    tel, events = rl.load_csv(os.path.join(ROOT, "data", "audio", "validation", "telemetry_reference.csv"))
    fc = np.array(prof["layers"][0]["octave_hz"], float)
    flo, fhi = fc / math.sqrt(2), fc * math.sqrt(2)
    A_pad_oct = np.array(next(x for x in prof["layers"] if x["name"] == "pad_interaction")["octave_db"])
    pad_shape = A_pad_oct - 10 * math.log10(np.sum(10 ** (A_pad_oct / 10)))
    cb = prof["propagation"]["atmospheric_absorption"]["crackle_band_policy"]
    f_eff = np.where(fc > cb["evaluate_at_hz"], cb["evaluate_at_hz"] * (fc / cb["evaluate_at_hz"]) ** cb["exponent"], fc)
    pad_src = np.array([0.0, 60.0, 1.0])
    samples = []
    for lname, lis in LISTENERS.items():
        lis = np.array(lis)
        for tt in TIMES:
            i = int(np.argmin(np.abs(tel["t"] - tt)))
            t = {k: float(tel[k][i]) for k in rl.TEL_COLS}
            noz = np.array([t["noz_e"], t["noz_n"], t["noz_u"]])
            ax = np.array([t["ax_e"], t["ax_n"], t["ax_u"]])
            st = am.source_state(prof, veh, t)
            L, g = am.band_spl(prof, st, noz, ax, lis, flo, fhi, fc, ground=True, absorb=True)
            w_pad = am.pad_weight(max(t["noz_u"], 0.0), st["De"]) if t["phase"] <= 2 else 0.0
            L = L + 20 * math.log10(max(1 - 0.5 * w_pad, 1e-3))
            wc = am.crackle_weight(prof, st["w_crackle"], g["theta"], g["r"], st["xt"])
            A = am.path_absorption_db(fc, g["src"], lis, prof)
            Ae = am.path_absorption_db(f_eff, g["src"], lis, prof)
            fr = min(1.0, cb["nonlinear_range_m"] / max(g["r"], 1.0))
            crk = np.where(fc > cb["evaluate_at_hz"], (A - Ae) * fr, 0.0)
            Lpad = None
            if w_pad > 1e-4:
                r = float(np.linalg.norm(lis - pad_src))
                r_eff = math.sqrt(r * r + st["xt"] ** 2)
                Lp = st["Lw"] + prof["pad_interaction"]["level_db_re_jet"] + pad_shape - 10 * math.log10(2 * math.pi * r_eff ** 2) \
                    + am.listener_impedance_db(lis[2]) - am.path_absorption_db(fc, pad_src, lis, prof) + 10 * math.log10(w_pad)
                Lpad = Lp.tolist()
            samples.append(dict(listener=lname, t=float(t["t"]), L_jet=L.tolist(), L_pad=Lpad, delay=g["delay"], r=g["r"],
                                theta=g["theta"], c=g["c"], w_c=wc, crackle_abs_undo=crk.tolist(), Lw=st["Lw"], f_peak=st["f_peak"],
                                xt=st["xt"], w_pad=w_pad, A_iso=A.tolist()))
    # reception -> emission mapping and one-shot arrivals for two ground listeners (as render())
    mapping = {}
    for lname in ("pad", "obs5km", "obs20km"):
        lis = np.array(LISTENERS[lname])
        grid = rl.model_grid(dict(prof, _pad_shape_db=np.zeros(len(rl.F_C))), veh, tel, lis, "ground")
        tau = tel["t"]
        tr = np.maximum.accumulate(tau + grid["delay"])
        t_r = np.array([-2.0, 0.0, 2.0, 10.0, 30.0, 60.0, 100.0, 150.0, 200.0])
        tau_r = np.interp(t_r, tr, tau, left=np.nan)
        ev = dict(events)
        i_lo = int(np.searchsorted(tau, 0.0))
        im = max(0, int(np.searchsorted(tau, ev["MECO"])) - 2)
        mapping[lname] = dict(t_r=t_r.tolist(), tau=[None if not np.isfinite(v) else float(v) for v in tau_r],
                              ignition_arrival=ev["IGNITION"] + float(grid["delay_pad"][i_lo]),
                              meco_arrival=ev["MECO"] + float(grid["delay"][im]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(generator="tools/audio/export_runtime_fixture.py", octave_hz=fc.tolist(), samples=samples,
                       mapping=mapping, events=events), fh, indent=1)
    print("wrote", OUT, len(samples), "samples")


if __name__ == "__main__":
    main()
