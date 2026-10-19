"""Compare the browser runtime's offline renders with the Python reference renders.

Inputs (both from the same reference telemetry, same listener, same mic gain Lp_max):
  tools/audio/_cache/renders/launch_<name>.wav    render_launch.py (validate.py)
  tools/audio/_cache/renders/browser_<name>.wav   frontend: npm run audio:offline (real Web Audio graph,
                                                  node-web-audio-api OfflineAudioContext, 30 Hz control ticks)
Output: data/audio/validation/browser_runtime_comparison.json and a printed summary.

Metrics use dsp.py (the same code as every other validation number): RMS envelope (125 ms frames, then 1 s
averages so the two noise realisations do not dominate), octave band levels (Welch) per window, HF slope and
derivative skewness (crackle) per window.

  python tools/audio/compare_browser.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import dsp  # noqa: E402

REN = os.path.join(HERE, "_cache", "renders")
OUT = os.path.join(ROOT, "data", "audio", "validation", "browser_runtime_comparison.json")
FS = 48000
T0 = -6.0
WINDOWS = {
    "pad": [(-2.0, 2.0), (2.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, 200.0)],
    "obs5km": [(11.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 150.0), (150.0, 260.0)],
    "onboard": [(-3.0, 0.0), (0.0, 10.0), (10.0, 60.0), (60.0, 150.0), (152.0, 158.0), (160.0, 200.0)],
}
WINDOWS["obs5km_noclamp"] = WINDOWS["obs5km"]  # diagnostic: read-rate clamp relaxed like the Python delay line
REF = {"obs5km_noclamp": "obs5km"}


def load(path):
    fs, x = wavfile.read(path)
    assert fs == FS
    return x.astype(np.float64) / 32768.0


def env_db(x, avg_s=1.0):
    t, e = dsp.rms_envelope(x, FS)
    k = int(round(avg_s / 0.0625))
    n = len(e) // k
    p = (e[: n * k] ** 2).reshape(n, k).mean(1)
    return T0 + (np.arange(n) + 0.5) * avg_s, 10 * np.log10(p + 1e-20)


HP20 = signal.butter(4, 20.0, "hp", fs=FS, output="sos")


def env_stats(ea, eb, ta):
    m = ea > ea.max() - 40
    d = eb[m] - ea[m]
    onset = lambda e: float(ta[np.argmax(e > e.max() - 30)])  # noqa: E731
    return dict(mean_diff_db=float(d.mean()), median_abs_db=float(np.median(np.abs(d))),
                p95_abs_db=float(np.percentile(np.abs(d), 95)), max_abs_db=float(np.abs(d).max()),
                correlation=float(np.corrcoef(ea[m], eb[m])[0, 1]), frames=int(m.sum()),
                peak_python_dbfs=float(ea.max()), peak_browser_dbfs=float(eb.max()),
                onset_python_s=onset(ea), onset_browser_s=onset(eb))


def compare(name):
    a = load(os.path.join(REN, f"launch_{REF.get(name, name)}.wav"))
    b = load(os.path.join(REN, f"browser_{name}.wav"))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ta, ea = env_db(a)
    _, eb = env_db(b)
    ah, bh = signal.sosfilt(HP20, a), signal.sosfilt(HP20, b)
    _, eah = env_db(ah)
    _, ebh = env_db(bh)
    res = dict(
        name=name, seconds=n / FS,
        rms_total_db=dict(python=float(10 * np.log10(np.mean(a * a))), browser=float(10 * np.log10(np.mean(b * b)))),
        rms_total_hp20_db=dict(python=float(10 * np.log10(np.mean(ah * ah))), browser=float(10 * np.log10(np.mean(bh * bh)))),
        envelope_1s=env_stats(ea, eb, ta),
        envelope_1s_hp20=env_stats(eah, ebh, ta),
        windows=[],
    )
    for w0, w1 in WINDOWS[name]:
        i0, i1 = int((w0 - T0) * FS), int((w1 - T0) * FS)
        xa, xb = a[i0:i1], b[i0:i1]
        ha, hb = ah[i0:i1], bh[i0:i1]
        if len(xa) < FS or np.sqrt(np.mean(xa ** 2)) < 1e-6:
            continue
        fa, pa = dsp.welch_psd(xa, FS)
        fb, pb = dsp.welch_psd(xb, FS)
        La = dsp.band_levels(fa, pa, dsp.OCT, frac=1)
        Lb = dsp.band_levels(fb, pb, dsp.OCT, frac=1)
        sel = La > La.max() - 40
        res["windows"].append(dict(
            t=[w0, w1],
            rms_db=dict(python=float(10 * np.log10(np.mean(xa ** 2))), browser=float(10 * np.log10(np.mean(xb ** 2)))),
            rms_hp20_db=dict(python=float(10 * np.log10(np.mean(ha ** 2))), browser=float(10 * np.log10(np.mean(hb ** 2)))),
            octave_hz=dsp.OCT.tolist(), octave_python_db=np.round(La, 2).tolist(), octave_browser_db=np.round(Lb, 2).tolist(),
            octave_max_abs_diff_db=float(np.abs(Lb - La)[sel].max()), octave_mean_abs_diff_db=float(np.abs(Lb - La)[sel].mean()),
            hf_slope_db_oct=dict(python=dsp.hf_slope_db_per_oct(fa, pa), browser=dsp.hf_slope_db_per_oct(fb, pb)),
            dsk=dict(python=dsp.deriv_skewness(xa, FS), browser=dsp.deriv_skewness(xb, FS)),
        ))
    return res


def main():
    out = {"generator": "tools/audio/compare_browser.py", "renders": []}
    for name in WINDOWS:
        if not os.path.exists(os.path.join(REN, f"browser_{name}.wav")):
            print("missing browser render", name, "(run: cd frontend && npm run audio:offline)")
            continue
        r = compare(name)
        info = os.path.join(REN, f"browser_{name}.json")
        if os.path.exists(info):
            with open(info, encoding="utf-8") as fh:
                bi = json.load(fh)
            r["browser_render"] = {k: bi[k] for k in ("Lp_max_used", "Lp_max_runtime", "oneshots", "mean_tick_us", "max_tick_us",
                                                      "realtime_factor", "decoded_mb") if k in bi}
        out["renders"].append(r)
        print(f"\n== {name}: RMS total py {r['rms_total_db']['python']:.2f} / browser {r['rms_total_db']['browser']:.2f} dBFS "
              f"(>20 Hz: {r['rms_total_hp20_db']['python']:.2f} / {r['rms_total_hp20_db']['browser']:.2f})")
        for lab in ("envelope_1s", "envelope_1s_hp20"):
            e = r[lab]
            print(f"   {lab:17s} diff mean {e['mean_diff_db']:+.2f}, median|d| {e['median_abs_db']:.2f}, p95 {e['p95_abs_db']:.2f}, "
                  f"max {e['max_abs_db']:.2f} dB, r={e['correlation']:.3f}; onset {e['onset_python_s']:.2f} vs {e['onset_browser_s']:.2f} s")
        for w in r["windows"]:
            print(f"   {w['t'][0]:7.1f}..{w['t'][1]:6.1f} s  rms {w['rms_db']['python']:7.2f} / {w['rms_db']['browser']:7.2f}  "
                  f">20Hz {w['rms_hp20_db']['python']:7.2f} / {w['rms_hp20_db']['browser']:7.2f}  "
                  f"oct |d| mean {w['octave_mean_abs_diff_db']:.2f} max {w['octave_max_abs_diff_db']:.2f} dB  "
                  f"slope {w['hf_slope_db_oct']['python']:.1f}/{w['hf_slope_db_oct']['browser']:.1f}  "
                  f"dSk {w['dsk']['python']:.2f}/{w['dsk']['browser']:.2f}")
            print("        py ", " ".join(f"{v:6.1f}" for v in w["octave_python_db"]))
            print("        br ", " ".join(f"{v:6.1f}" for v in w["octave_browser_db"]))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
