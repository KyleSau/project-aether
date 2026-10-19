"""Render the validation listeners and compare them with the real references using the same analysis code.

Outputs
  tools/audio/_cache/renders/*.wav          full-quality renders (git-ignored)
  docs/audio/*.ogg                          Opus copies for listening (committed)
  docs/figures/audio/val_*.png              comparison figures
  data/audio/validation/validation_metrics.json
  data/audio/validation/telemetry_reference.csv   reference-sim telemetry for runtime ground-truth checks
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import acoustic_model as am  # noqa: E402
import dsp  # noqa: E402
import render_launch as rl  # noqa: E402

FIG = os.path.join(ROOT, "docs", "figures", "audio")
WAVDIR = os.path.join(HERE, "_cache", "renders")
OGGDIR = os.path.join(ROOT, "docs", "audio")
VALDIR = os.path.join(ROOT, "data", "audio", "validation")
FS = 48000

RENDERS = [
    ("pad", dict(listener="pad", t1=200.0)),
    ("obs5km", dict(listener="obs5km", t1=260.0)),
    ("obs20km", dict(listener="obs20km", t1=300.0)),
    ("onboard", dict(listener="onboard", t1=200.0)),
    ("cabin", dict(listener="cabin", t1=200.0)),
]
# launch phases in EMISSION time (s from liftoff); mapped to reception time per listener
PHASES = [("ignition_ramp", -3.0, -0.6), ("pad_full_thrust", -0.6, 3.0), ("liftoff_pad_interaction", 3.0, 9.0),
          ("early_ascent", 9.0, 30.0), ("mid_ascent", 30.0, 70.0), ("late_ascent", 70.0, 140.0), ("meco_tail", 151.0, 165.0)]


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def old_audio_ts_proxy(seconds=20.0, seed=7):
    """Python re-implementation of the previous frontend/src/engine/Audio.ts patch (for the before/after note):
    4-s looped leaky-integrated white noise through a 220-420 Hz low-pass + 4-s looped sparse clicks band-passed
    at 900 Hz (Q 0.6). Level/LPF follow 'acoustic'; here held at full level."""
    from scipy import signal
    r = np.random.default_rng(seed)
    n4 = 4 * FS
    w = r.uniform(-1, 1, n4)
    b = np.zeros(n4)
    last = 0.0
    for i in range(n4):
        last = (last + 0.02 * w[i]) / 1.02
        b[i] = last * 3.5
    c = np.where(r.random(n4) < 0.004, r.uniform(-1, 1, n4), r.uniform(-1, 1, n4) * 0.08)
    reps = int(math.ceil(seconds / 4))
    b, c = np.tile(b, reps)[:int(seconds * FS)], np.tile(c, reps)[:int(seconds * FS)]
    lp = signal.butter(2, 420, fs=FS, output="sos")
    bp = signal.butter(1, [900 / 1.8, 900 * 1.8], btype="bandpass", fs=FS, output="sos")
    y = 0.6 * signal.sosfilt(lp, b) + 0.6 * 0.5 * signal.sosfilt(bp, c)
    return y / np.abs(y).max() * 0.5


def phase_windows(info, grid, tel):
    """Reception-time windows for each launch phase for this render."""
    tau = tel["t"]
    out = []
    for name, a, b in PHASES:
        if info["mode"] == "ground":
            ra = a + np.interp(a, tau, grid["delay"])
            rb = b + np.interp(b, tau, grid["delay"])
        else:
            ra, rb = a, b
        if ra >= info["t0"] and rb <= info["t1"]:
            out.append((name, ra, rb))
    return out


def analyse_render(y, info, grid, tel):
    segs = []
    for name, a, b in phase_windows(info, grid, tel):
        i0, i1 = int((a - info["t0"]) * FS), int((b - info["t0"]) * FS)
        x = y[i0:i1]
        if len(x) < FS or np.sqrt(np.mean(x * x)) < 1e-6:
            continue
        m = dsp.segment_metrics(x, FS, name)
        m.update(t0=round(a, 2), t1=round(b, 2))
        segs.append(m)
    return segs


def encode(wav, ogg, kbps=64):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "libopus", "-b:a", f"{kbps}k", ogg], check=True)


def oaspl_vs_time(prof, veh, tel, pos):
    sel = np.arange(0, len(tel["t"]), 10)
    sub = {k: v[sel] for k, v in tel.items()}
    lay = rl.load_assets(prof)
    A_pad = rl.asset_third_db(lay["pad_interaction"]["meta"])
    prof["_pad_shape_db"] = A_pad - 10 * math.log10(np.sum(10 ** (A_pad / 10)))
    g = rl.model_grid(prof, veh, sub, np.array(pos), "ground")
    L = 10 * np.log10(10 ** (g["L_jet"] / 10) + 10 ** (g["L_pad"] / 10))
    oa = 10 * np.log10(np.sum(10 ** (L / 10), axis=1))
    return sub["t"] + g["delay"], oa, sub["t"]


def main(render=True):
    os.makedirs(WAVDIR, exist_ok=True)
    os.makedirs(OGGDIR, exist_ok=True)
    os.makedirs(VALDIR, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    prof = am.load_profile()
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    tel, events = rl.sim_telemetry()
    rl.export_csv(tel, events, os.path.join(VALDIR, "telemetry_reference.csv"))
    results = {}
    plt = mpl()
    for name, kw in RENDERS:
        wav = os.path.join(WAVDIR, f"launch_{name}.wav")
        y, info, grid, _ = rl.render(tel=tel, events=events, prof=am.load_profile(), out_path=wav, return_debug=True, **kw)
        encode(wav, os.path.join(OGGDIR, f"launch_{name}.ogg"))
        segs = analyse_render(y, info, grid, tel)
        evs = [(n, t + (np.interp(t, tel["t"], grid["delay"]) if info["mode"] == "ground" else 0.0)) for n, t in events]
        evs = [(n, t) for n, t in evs if info["t0"] < t < info["t1"]]
        dsp.plot_panel(y, FS, f"render: {name} ({info['mode']}, pos {info['pos']})", os.path.join(FIG, f"val_render_{name}.png"),
                       t0=info["t0"], events=evs)
        results[name] = dict(info={k: v for k, v in info.items() if k != "events"}, segments=segs,
                             wav=os.path.relpath(wav, ROOT).replace("\\", "/"), ogg=f"docs/audio/launch_{name}.ogg",
                             bytes_ogg=os.path.getsize(os.path.join(OGGDIR, f"launch_{name}.ogg")))
        print(name, "segments:", [(s["label"], round(s["deriv_skewness"], 2), round(s["hf_slope_db_per_oct"], 1)) for s in segs])
    # before: old Audio.ts proxy
    old = old_audio_ts_proxy()
    results["old_audio_ts_proxy"] = dict(segments=[dsp.segment_metrics(old, FS, "full_level")])
    with open(os.path.join(ROOT, "data", "audio", "reference_analysis.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    with open(os.path.join(VALDIR, "validation_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    figures(results, ref, prof, veh, tel, plt)
    print("wrote", os.path.join(VALDIR, "validation_metrics.json"))
    return results


def _ref_seg(ref, rid, ch, label):
    for r in ref["results"]:
        if r["ref"] == rid and r["channel"] == ch:
            for s in r["segments"]:
                if s["label"] == label:
                    return s
    return None


def figures(results, ref, prof, veh, tel, plt):
    TO = dsp.THIRD_OCT
    m = (TO >= 10) & (TO <= 16000)
    pairs = [  # (render, phase) vs (ref, ch, segment)
        ("Near pad", ("pad", "pad_full_thrust"), ("imap", 0, "receding"), "IMAP Blast-1 (receding, unclipped)"),
        ("Near pad (early ascent)", ("pad", "early_ascent"), ("imap", 0, "decay"), "IMAP Blast-1 (decay)"),
        ("~5 km", ("obs5km", "early_ascent"), ("imap", 1, "peak_level"), "IMAP Blast-2 peak (~5.5-7.5 km)"),
        ("~5 km later", ("obs5km", "mid_ascent"), ("imap", 1, "ascent_decay"), "IMAP Blast-2 ascent decay"),
        ("~20 km", ("obs20km", "mid_ascent"), ("atlas_sdo", 0, "ascent"), "Atlas V SDO (distant, unknown)"),
        ("~20 km vs beach", ("obs20km", "early_ascent"), ("crew12", 0, "ascent_decay_A"), "Crew-12 Beach decay"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    rows = []
    for ax, (title, (rn, ph), (rid, ch, lab), rlabel) in zip(axs.flat, pairs):
        rs = next((s for s in results[rn]["segments"] if s["label"] == ph), None)
        fs_ = _ref_seg(ref, rid, ch, lab)
        if rs is None or fs_ is None:
            continue
        a = np.array(rs["third_oct_db"])
        b = np.array(fs_["third_oct_db"])
        k250 = int(np.where(TO == 250)[0][0])
        ax.semilogx(TO[m], (a - a[k250])[m], "-o", ms=3, color="#b5442c", label=f"render {rn}/{ph}")
        ax.semilogx(TO[m], (b - b[k250])[m], "-s", ms=3, color="#2a5d8f", label=f"real: {rlabel}")
        ax.axvspan(10, 40, color="0.9", zorder=0)
        ax.text(11, -38, "broadcast-mic LF\nresponse unknown", fontsize=7)
        ax.set_ylim(-45, 25)
        ax.set_title(title)
        ax.set_xlabel("Hz"); ax.set_ylabel("dB re 250 Hz band")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=7)
        rows.append(dict(pair=title, render=f"{rn}/{ph}", reference=f"{rid}/{ch}/{lab}",
                         render_hf_slope=rs["hf_slope_db_per_oct"], ref_hf_slope=fs_["hf_slope_db_per_oct"],
                         render_dSk=rs["deriv_skewness"], ref_dSk=fs_["deriv_skewness"],
                         render_mod=[rs["mod_index_0p5_4hz"], rs["mod_index_4_20hz"]],
                         ref_mod=[fs_["mod_index_0p5_4hz"], fs_["mod_index_4_20hz"]],
                         render_crackle_cv=rs["crackle_interval_cv"], ref_crackle_cv=fs_["crackle_interval_cv"]))
    fig.suptitle("1/3-octave spectra, render vs real (normalised at 250 Hz; uncalibrated)")
    fig.savefig(os.path.join(FIG, "val_third_octave.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(VALDIR, "comparison_pairs.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    # crackle metric + HF slope + modulation bars
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    labels, dsk, slope, mod = [], [], [], []
    for rn in ("pad", "obs5km", "obs20km"):
        for s in results[rn]["segments"]:
            if s["label"] in ("pad_full_thrust", "early_ascent", "mid_ascent", "late_ascent"):
                labels.append(f"RENDER {rn}/{s['label']}"); dsk.append(s["deriv_skewness"]); slope.append(s["hf_slope_db_per_oct"])
                mod.append(s["mod_index_4_20hz"])
    for r in ref["results"]:
        for s in r["segments"]:
            if s["clipped"] or any(k in s["label"] for k in ("ambient", "floor", "crosstalk", "tail")):
                continue
            labels.append(f"real {r['ref']}/{r['name'][:7]}/{s['label'][:14]}"); dsk.append(s["deriv_skewness"])
            slope.append(s["hf_slope_db_per_oct"]); mod.append(s["mod_index_4_20hz"])
    o = results["old_audio_ts_proxy"]["segments"][0]
    labels.append("BEFORE: old Audio.ts proxy"); dsk.append(o["deriv_skewness"]); slope.append(o["hf_slope_db_per_oct"]); mod.append(o["mod_index_4_20hz"])
    y = np.arange(len(labels))
    col = ["#b5442c" if l.startswith("RENDER") else ("#999999" if l.startswith("BEFORE") else "#2a5d8f") for l in labels]
    ax[0].barh(y, dsk, color=col); ax[0].axvline(3, ls=":", c="k"); ax[0].axvline(5, ls="--", c="k")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=7); ax[0].set_title("derivative skewness (crackle)")
    ax[1].barh(y, slope, color=col); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_title("PSD slope 250-4000 Hz (dB/oct)")
    ax[2].barh(y, mod, color=col); ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_title("envelope modulation index 4-20 Hz")
    fig.savefig(os.path.join(FIG, "val_metrics.png"), dpi=110)
    plt.close(fig)

    # OASPL vs time (model) with Falcon 9 published points at 8.3 km
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name, pos in (("pad camera 405 m", (310, -260, 1.7)), ("5 km", (-3536, 3536, 1.7)), ("8.3 km", (-5869, 5869, 1.7)),
                      ("20 km", (-14142, 14142, 1.7))):
        tr, oa, te = oaspl_vs_time(am.load_profile(), veh, tel, pos)
        ax.plot(te, oa, label=f"model, {name} (vs EMISSION time)")
    ax.plot([10.5, 40.0], [106, 114], "ks", ms=7, label="Falcon 9 measured @ 8.3 km (Mathews et al. 2020): ~106 dB @10-11 s, max 112-114 dB @~40 s")
    ax.set_xlim(-5, 170); ax.set_ylim(50, 160); ax.grid(alpha=.3)
    ax.set_xlabel("emission time from liftoff (s) - Mathews' times are arrival-corrected launch times")
    ax.set_ylabel("OASPL dB re 20 uPa (model)")
    ax.legend(fontsize=8)
    ax.set_title("Predicted OASPL (eta = 0.5 %) vs the only calibrated Falcon 9 numbers we have")
    fig.savefig(os.path.join(FIG, "val_oaspl.png"), dpi=110)
    plt.close(fig)

    # envelopes: 5 km render vs IMAP Blast-2, aligned at their maxima
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    fs_, xr = wavfile.read(os.path.join(WAVDIR, "launch_obs5km.wav"))
    xr = xr / 32768.0
    t, e = dsp.rms_envelope(xr, FS, 0.5, 0.25)
    L = 20 * np.log10(e + 1e-9)
    ax.plot(t - t[np.argmax(L)], L - L.max(), color="#b5442c", label="render obs5km (RMS 0.5 s)")
    cache = os.path.join(HERE, "_cache", "imap.wav")
    if os.path.exists(cache):
        fs2, xi = wavfile.read(cache)
        xi = xi[:, 1] / 32768.0
        t2, e2 = dsp.rms_envelope(xi[int(100 * fs2):int(260 * fs2)], fs2, 0.5, 0.25)
        L2 = 20 * np.log10(e2 + 1e-9)
        ax.plot(t2 - t2[np.argmax(L2)], L2 - L2.max(), color="#2a5d8f", label="real IMAP Blast-2 (~5.5-7.5 km)")
    ax.set_xlim(-40, 120); ax.set_ylim(-45, 3); ax.grid(alpha=.3); ax.legend()
    ax.set_xlabel("s from envelope maximum"); ax.set_ylabel("dB re max")
    ax.set_title("Envelope shape: rise, peak (directivity) and decay")
    fig.savefig(os.path.join(FIG, "val_envelopes.png"), dpi=110)
    plt.close(fig)

    # modulation spectra
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for lab, path, ch, a, b, col in (("render obs5km mid", os.path.join(WAVDIR, "launch_obs5km.wav"), None, None, None, "#b5442c"),
                                     ("render pad early", os.path.join(WAVDIR, "launch_pad.wav"), None, None, None, "#e08a6c")):
        fs_, x = wavfile.read(path)
        x = x / 32768.0
        seg = next(s for s in results[os.path.basename(path)[7:-4]]["segments"] if s["label"] in ("mid_ascent", "early_ascent"))
        x = x[int((seg["t0"] + 6) * FS):int((seg["t1"] + 6) * FS)]
        fm, pm = dsp.modulation_spectrum(x, FS)
        ax.loglog(fm, pm, color=col, label=lab)
    for lab, rid, ch, a, b, col in (("real IMAP Blast-2 peak", "imap", 1, 122, 145, "#2a5d8f"), ("real Crew-12 beach decay", "crew12", 0, 161, 190, "#5c8fc0")):
        p = os.path.join(HERE, "_cache", rid + ".wav")
        if os.path.exists(p):
            fs2, x = wavfile.read(p)
            x = x[int(a * fs2):int(b * fs2), ch] / 32768.0
            fm, pm = dsp.modulation_spectrum(x, fs2)
            ax.loglog(fm, pm, color=col, label=lab)
    ax.set_xlabel("modulation frequency (Hz)"); ax.set_ylabel("envelope PSD (depth^2/Hz)"); ax.grid(alpha=.3, which="both")
    ax.legend(); ax.set_title("Modulation (fluctuation) spectra")
    fig.savefig(os.path.join(FIG, "val_modulation.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
