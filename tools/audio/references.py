"""Fetch and analyse the real launch recordings used as references.

Nothing fetched here is shipped: the recordings are analysed for statistics only. All items are NASA-produced
media (images.nasa.gov / nasa.gov). NASA media are generally not subject to copyright in the US (NASA Images and
Media Usage Guidelines, https://www.nasa.gov/nasa-brand-center/images-and-media/); NASA must not be implied to
endorse anything, and third-party content inside NASA products can carry rights - the Falcon 9 feeds were recorded
by NASA/KSC staff (credit 'NASA/Lorne Mathre' in the asset metadata).

Download windows are cut with ffmpeg seeking inside the remote CBR MP3 (192 kbit/s) so only ~100 MB per window is
transferred; the cache lives in tools/audio/_cache (git-ignored). Times below are seconds INSIDE each window.

  python tools/audio/references.py            # fetch (if needed) + analyse + figures + data/audio/reference_analysis.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import dsp  # noqa: E402

CACHE = os.environ.get("AETHER_AUDIO_CACHE", os.path.join(HERE, "_cache"))
FIG = os.path.join(ROOT, "docs", "figures", "audio")
OUT = os.path.join(ROOT, "data", "audio", "reference_analysis.json")
NASA_LIC = ("NASA-produced media: generally not copyrighted in the US (NASA Images and Media Usage Guidelines). "
            "Used for ANALYSIS ONLY; nothing from it is shipped.")
IMG = "https://images-assets.nasa.gov/audio/"

REFS = [
    dict(id="imap", nasa_id="KSC-20250924-AU-GEB01-0001-SpaceX_IMAP_Live_Launch_Coverage-BLAST_1_BLAST_2-M16776",
         url=IMG + "KSC-20250924-AU-GEB01-0001-SpaceX_IMAP_Live_Launch_Coverage-BLAST_1_BLAST_2-M16776/"
                   "KSC-20250924-AU-GEB01-0001-SpaceX_IMAP_Live_Launch_Coverage-BLAST_1_BLAST_2-M16776~orig.mp3",
         window=(2990, 420), vehicle="Falcon 9 (9 x Merlin 1D, LOX/RP-1)", launch="IMAP, LC-39A KSC, 2025-09-24",
         type="broadcast technical feed (unmixed blast microphones), MP3 192 kbit/s 48 kHz",
         channels={
             0: dict(name="Blast 1", position="unknown; near-pad 'blast' microphone (onset 45 dB in 1.1 s, heavy clipping)",
                     segments=[("ambient", 0, 80), ("startup_ramp", 84.3, 85.7), ("pad_full_thrust_CLIPPED", 86.0, 96.0),
                               ("receding", 96.0, 110.0), ("decay", 110.0, 118.0)],
                     artefacts=["clipping 86-96 s (up to 21 % of samples)", "level step at 118 s and 202 s: fader/routing change "
                                "(not acoustic)", "LF response unknown (broadcast mic), roll-off below ~12 Hz measured"]),
             1: dict(name="Blast 2", position="remote mic; first acoustic arrival ~20 s after Blast-1 onset -> ~5.5-7.5 km "
                                              "if straight-ray at 343 m/s (inferred, not documented)",
                     segments=[("noise_floor", 0, 80), ("crosstalk_from_blast1", 86, 97), ("arrival_rise", 105, 121),
                               ("peak_level", 122, 145), ("ascent_decay", 145, 190), ("distant_tail", 190, 230), ("ambient", 260, 330)],
                     artefacts=["86-97 s plateau is electrical crosstalk of Blast 1 (flat HF, starts/stops with Blast 1 clipping)",
                                "LF below ~30 Hz depressed (mic/preamp), MP3 low-pass ~16 kHz"]),
         }),
    dict(id="crew12", nasa_id="KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Beach_Causeway-M19013",
         url=IMG + "KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Beach_Causeway-M19013/"
                   "KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Beach_Causeway-M19013~orig.mp3",
         window=(7650, 600), vehicle="Falcon 9", launch="Crew-12, SLC-40 CCSFS, 2026-02-13",
         type="broadcast feed 'Beach (L) / Causeway (R)', MP3 192 kbit/s 48 kHz",
         channels={
             0: dict(name="Beach", position="unknown ('Beach' camera position)",
                     segments=[("ambient", 60, 130), ("gated_onset_CLIPPED", 136, 157), ("ascent_decay_A", 161, 190),
                               ("ascent_decay_B", 198, 220), ("ambient_after", 230, 250)],
                     artefacts=["136 s onset is a 40 dB step in 0.1 s -> channel gate/fader, not acoustic",
                                "136-157 s fully clipped (~12 % of samples at full scale)",
                                "periodic 500-1500 Hz bursts every ~30 s (191, 222, 253, 292 s): surf or PA, excluded"]),
         }),
    dict(id="crew12blast", nasa_id="KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Tech_Feed_Blast-M19013",
         url=IMG + "KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Tech_Feed_Blast-M19013/"
                   "KSC-20260213-AU-LMM01-0001-SpaceX_Crew_12_Live_Launch_Coverage_Tech_Feed_Blast-M19013~orig.mp3",
         window=(7650, 600), vehicle="Falcon 9", launch="Crew-12, SLC-40 CCSFS, 2026-02-13",
         type="broadcast feed 'Tech Feed (L) / Blast (R)', MP3 192 kbit/s",
         channels={
             1: dict(name="Blast", position="unknown near-pad blast mic",
                     segments=[("ambient", 60, 130), ("liftoff_CLIPPED", 137.5, 154.5)],
                     artefacts=["gated/switched feed: after 155 s the channel carries unrelated program bursts",
                                "onset at 136.8 s is a gate step", "clipped throughout the liftoff segment"]),
         }),
    dict(id="atlas_sdo", url="https://www.nasa.gov/wp-content/uploads/2015/01/590329main_ringtone_SDO_launchNats.mp3",
         window=None, vehicle="Atlas V 401 (1 x RD-180, LOX/RP-1, no SRBs)", launch="SDO, SLC-41, 2010-02-11",
         type="NASA 'launch nats' clip (edited ringtone: fades, MP3 128 kbit/s 44.1 kHz, distance unknown)",
         channels={0: dict(name="L", position="unknown, distant (no near-field HF, crackle clicks audible)",
                           segments=[("ascent", 0.6, 19.4)], artefacts=["fade in/out (edited)", "MP3 low-pass 16 kHz",
                                                                        "LF < 20 Hz absent"])}),
    dict(id="sts131", url="https://www.nasa.gov/wp-content/uploads/2015/01/590189main_ringtone_131_launchNats.mp3",
         window=None, vehicle="Space Shuttle (3 x RS-25 LOX/LH2 + 2 SRBs)", launch="STS-131, LC-39A, 2010-04-05",
         type="NASA 'launch nats' clip (edited ringtone, MP3 128 kbit/s)",
         channels={0: dict(name="L", position="unknown", segments=[("ascent", 0.5, 18.5)],
                           artefacts=["edited", "heavy limiting suspected (crest factor 6 dB)", "SRB-dominated: secondary reference"])}),
    dict(id="sls_test", url="https://www.nasa.gov/wp-content/uploads/2015/01/663784main_SLS_Audio_D.mp3",
         window=None, vehicle="SLS-programme static test (NASA 'Sounds of the Future')", launch="static test fire (not a launch)",
         type="NASA clip, MP3 192 kbit/s", channels={0: dict(name="L", position="unknown", segments=[("static_fire", 0.5, 23.0)],
                                                            artefacts=["static test, no ascent", "possibly limited"])}),
]


def fetch(ref):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, ref["id"] + ".wav")
    if os.path.exists(out):
        return out
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if ref["window"]:
        cmd += ["-ss", str(ref["window"][0])]
    cmd += ["-i", ref["url"]]
    if ref["window"]:
        cmd += ["-t", str(ref["window"][1])]
    cmd += ["-ar", "48000", "-c:a", "pcm_s16le", out]
    print("fetching", ref["id"], "...")
    subprocess.run(cmd, check=True)
    return out


def load(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    if x.ndim == 1:
        x = x[:, None]
    return fs, x


def analyse(make_figs=True):
    os.makedirs(FIG, exist_ok=True)
    results = []
    for ref in REFS:
        fs, x = load(fetch(ref))
        for ch, info in ref["channels"].items():
            y = x[:, ch]
            segs = []
            for name, a, b in info["segments"]:
                seg = y[int(a * fs):int(b * fs)]
                m = dsp.segment_metrics(seg, fs, name)
                m.update(t0=a, t1=b, clipped=("CLIPPED" in name) or m["clip_fraction"] > 1e-3)
                segs.append(m)
            tag = f"ref_{ref['id']}_{info['name'].replace(' ', '').lower()}"
            if make_figs:
                spans = [(n, a) for n, a, b in info["segments"]]
                a0 = min(a for _, a, _ in info["segments"]) - 2 if ref["window"] else 0.0
                b0 = max(b for _, _, b in info["segments"]) + 2 if ref["window"] else len(y) / fs
                a0 = max(a0, 0)
                dsp.plot_panel(y[int(a0 * fs):int(b0 * fs)], fs, f"{ref['launch']} - {info['name']}",
                               os.path.join(FIG, tag + ".png"), t0=a0, events=spans)
            results.append(dict(ref=ref["id"], channel=ch, name=info["name"], launch=ref["launch"], vehicle=ref["vehicle"],
                                url=ref["url"], nasa_id=ref.get("nasa_id"), window=ref["window"], type=ref["type"],
                                position=info["position"], artefacts=info["artefacts"], licence=NASA_LIC, figure=tag + ".png",
                                segments=segs))
    summary = summarise(results)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(schema="aether-reference-analysis/1", note="dBFS levels are uncalibrated; see docs/AUDIO_ACOUSTICS.md",
                       results=results, summary=summary), fh, indent=1)
    if make_figs:
        compare_fig(results)
    print("wrote", OUT)
    return results, summary


def _clean(results):
    for r in results:
        for s in r["segments"]:
            if not s["clipped"] and "ambient" not in s["label"] and "floor" not in s["label"] and "crosstalk" not in s["label"]:
                yield r, s


def summarise(results):
    clean = list(_clean(results))
    dsk = {f"{r['ref']}/{r['name']}/{s['label']}": round(s["deriv_skewness"], 2) for r, s in clean}
    slope = {f"{r['ref']}/{r['name']}/{s['label']}": round(s["hf_slope_db_per_oct"], 2) for r, s in clean}
    mod = {f"{r['ref']}/{r['name']}/{s['label']}": [round(s["mod_index_0p5_4hz"], 3), round(s["mod_index_4_20hz"], 3)] for r, s in clean}
    cv = {f"{r['ref']}/{r['name']}/{s['label']}": round(s["crackle_interval_cv"], 2) for r, s in clean}
    f9 = [s for r, s in clean if r["vehicle"].startswith("Falcon")]
    return dict(
        deriv_skewness=dsk, hf_slope_db_per_oct_250_4000=slope, modulation_index_0p5_4_and_4_20hz=mod, crackle_interval_cv=cv,
        falcon9_clean_segments=len(f9),
        falcon9_hf_slope_range=[round(min(s["hf_slope_db_per_oct"] for s in f9), 2), round(max(s["hf_slope_db_per_oct"] for s in f9), 2)] if f9 else None,
        falcon9_abs_dsk_range=[round(min(abs(s["deriv_skewness"]) for s in f9), 2), round(max(abs(s["deriv_skewness"]) for s in f9), 2)] if f9 else None,
        imap_blast1_startup_rise_db_per_s=startup_rise(results),
    )


def startup_rise(results):
    for r in results:
        if r["ref"] == "imap" and r["channel"] == 0:
            fs, x = load(os.path.join(CACHE, "imap.wav"))
            y = x[:, 0]
            t, e = dsp.rms_envelope(y[int(83.5 * fs):int(86.0 * fs)], fs, 0.05, 0.025)
            L = 20 * np.log10(e + 1e-9)
            i0 = np.argmax(L > L[:10].mean() + 6)
            i1 = np.argmax(L > -20)
            return dict(rise_db=round(float(L[i1] - L[i0]), 1), seconds=round(float(t[i1] - t[i0]), 2),
                        note="Blast-1: -65 -> -20 dBFS before clipping; rise time of the engine-start ramp at a near-pad mic")
    return None


def compare_fig(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for r, s in _clean(results):
        L = np.array(s["third_oct_db"])
        m = (dsp.THIRD_OCT >= 20) & (dsp.THIRD_OCT <= 12500)
        Ln = L - np.nanmax(L[m])
        ls = "-" if r["vehicle"].startswith("Falcon") else "--"
        ax[0].semilogx(dsp.THIRD_OCT[m], Ln[m], ls, lw=1, label=f"{r['ref']}/{r['name']}/{s['label']}")
    ax[0].set_title("Real launches: 1/3-octave spectra (unclipped segments, normalised to max)")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB re band max"); ax[0].grid(True, which="both", alpha=.3)
    ax[0].legend(fontsize=6)
    names, dsks, slopes = [], [], []
    for r, s in _clean(results):
        names.append(f"{r['ref']}/{r['name'][:6]}/{s['label'][:14]}")
        dsks.append(abs(s["deriv_skewness"]))
        slopes.append(s["hf_slope_db_per_oct"])
    y = np.arange(len(names))
    ax[1].barh(y, dsks, color="#2a5d8f")
    ax[1].axvline(3, color="k", ls=":", lw=1); ax[1].axvline(5, color="k", ls="--", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xlabel("|derivative skewness|  (3: crackle present, 5: significant shocks)")
    ax[1].set_title("Crackle metric on real recordings (MP3, uncalibrated)")
    fig.savefig(os.path.join(FIG, "ref_comparison.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    res, summ = analyse("--no-figs" not in sys.argv)
    print(json.dumps(summ, indent=1))
