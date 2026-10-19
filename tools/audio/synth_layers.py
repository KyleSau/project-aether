"""Offline synthesis of the launch source textures (data/audio/assets/).

All loops are synthesised CIRCULARLY (FFT-domain shaping, circular modulation, circular nonlinear
steepening), so the last sample flows into the first exactly as any interior sample flows into the next:
no crossfade is needed and there is no seam. tests/test_audio.py checks this.

Layers (mono, 48 kHz, 16-bit WAV + Ogg/Opus):
  rumble_lf          jet noise below ~80 Hz (power-complementary split of the same base process as jet_mix)
  jet_mix            jet-mixing noise above ~80 Hz
  crackle            HF (>400 Hz) of the steepened energetic (<2 kHz) part of the same process; weight w
  pad_interaction    deflector/trench impingement roar + sound-suppression water/steam texture
  onboard_structure  structure-borne random vibration as heard inside the vehicle
  aero_buffet        boundary-layer / transonic buffet noise on the hull
One-shots: ignition, meco_tail, stage_sep_structural.

Every random draw comes from numpy Generator(PCG64(seed)); seeds are in SEEDS and written to the manifest.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import dsp  # noqa: E402
import acoustic_model as am  # noqa: E402

FS = 48000
LOOP_S = 24.0
OUT = os.path.join(ROOT, "data", "audio", "assets")
SEEDS = dict(base=1001, mod_lf=1002, mod_mid=1003, mod_common=1004, pad=1005, pad_mod=1006, water=1007,
             struct=1008, struct_mod=1009, aero=1010, aero_mod=1011, ignition=1012, meco=1013, sep=1014, crackle_mod=1015)
TARGET_RMS_DBFS = -20.0

# Reference condition (stage 1, sea level, full thrust) -> peak frequency 18 Hz (see profile)
REF_U = 2786.0
REF_D = 3.096
CROSSOVER_HZ = 80.0
# steepening strength (dimensionless; total advection / rms slope). Tuned so the steepened waveform's
# derivative skewness lands in the 4-8 range measured near rockets (see docs).
STEEPEN_K = 2.0
CRACKLE_SOFTCLIP = 1.4
CRACKLE_RISE_HZ = 5000.0
ENERGETIC_HZ = 2000.0
CRACKLE_HP_HZ = 400.0
FAST_MOD_SIGMA = 0.35


def rng(name):
    return np.random.Generator(np.random.PCG64(SEEDS[name]))


def freqs(n, fs=FS):
    return np.fft.rfftfreq(n, 1.0 / fs)


def shaped_noise(n, psd_fn, r, fs=FS):
    """Circular Gaussian noise with one-sided PSD shape psd_fn(f) (arbitrary units)."""
    f = freqs(n, fs)
    mag = np.sqrt(np.maximum(psd_fn(np.maximum(f, 1e-3)), 0.0))
    mag[0] = 0.0
    X = (r.standard_normal(len(f)) + 1j * r.standard_normal(len(f))) * mag
    x = np.fft.irfft(X, n)
    return x / (x.std() + 1e-30)


def circ_filter(x, gain_fn, fs=FS):
    """Zero-phase circular filter with amplitude response gain_fn(f)."""
    f = freqs(len(x), fs)
    return np.fft.irfft(np.fft.rfft(x) * gain_fn(np.maximum(f, 1e-3)), len(x))


def mod_process(n, r, f_lo, f_hi, sigma, fs=FS, slope=-1.0):
    """Log-normal (multiplicative) modulation envelope, circular, unit mean.
    Underlying Gaussian has a band-limited 1/f^slope-ish spectrum between f_lo and f_hi."""
    m = 64  # synthesise at low rate then interpolate circularly via FFT zero-padding
    nl = n // m
    fl = freqs(nl, fs / m)

    def psd(f):
        return np.where((f >= f_lo) & (f <= f_hi), (f / f_lo) ** slope, 0.0) + 1e-12

    z = shaped_noise(nl, psd, r, fs / m)
    Z = np.fft.rfft(z)
    full = np.zeros(n // 2 + 1, complex)
    full[:len(Z)] = Z * (n / nl)
    z = np.fft.irfft(full, n)
    z = z / (z.std() + 1e-30)
    e = np.exp(sigma * z - 0.5 * sigma * sigma)
    return e / e.mean()


def sp8072_psd(prof, U=REF_U, D=REF_D):
    """Per-Hz PSD shape of the SP-8072 normalised spectrum (same evaluator as the runtime model)."""
    def fn(f):
        return 10 ** (am.spectrum_level_db(prof, np.asarray(f) * D / U) / 10)
    return fn


def steepen(x, K, fs=FS, n_steps=48, f_abs=12000.0):
    """Circular nonlinear steepening (Earnshaw/Burgers, Pestorius-style): each step advances samples by
    k*x (compressions travel faster), resamples to the uniform grid, then applies a weak thermoviscous
    low-pass. K is the total advection in units of 1/rms(dx/dt) (Gol'dberg-like strength)."""
    n = len(x)
    y = x.copy()
    slope_rms = np.std(np.diff(np.concatenate([y, y[:1]])) * fs)
    k_total = K / slope_rms  # seconds of advection per unit amplitude
    done = 0.0
    f = freqs(n, fs)
    for _ in range(n_steps * 12):
        if done >= k_total * 0.999:
            break
        dy = np.diff(np.concatenate([y, y[:1]])) * fs
        k = min(0.45 / max(dy.max(), 1e-9), k_total - done, k_total / n_steps)
        t = np.arange(n) / fs - k * y  # new arrival times (compressions earlier)
        order = np.argsort(t, kind="stable")
        ts, ys = t[order], y[order]
        T = n / fs
        ts = np.concatenate([ts[-1:] - T, ts, ts[:1] + T])
        ys = np.concatenate([ys[-1:], ys, ys[:1]])
        y = np.interp(np.arange(n) / fs, ts, ys)
        # thermoviscous absorption per step (keeps shocks finite, ~ f^2)
        Y = np.fft.rfft(y) * np.exp(-(f / f_abs) ** 2 * (k / k_total) * 0.6)
        y = np.fft.irfft(Y, n)
        done += k
    return y


def xover(f, fc=CROSSOVER_HZ, width_oct=2.0):
    """Power-complementary crossover masks (cos/sin), transition +-width/2 octaves around fc."""
    u = np.clip((np.log2(f / fc) + width_oct / 2) / width_oct, 0, 1)
    return np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)


def lowcut(f, fc, order=2):
    return 1.0 / np.sqrt(1 + (fc / f) ** (2 * order))


def highcut(f, fc, order=2):
    return 1.0 / np.sqrt(1 + (f / fc) ** (2 * order))


# ------------------------------------------------------------------ loops
def make_jet_layers(prof, n):
    r = rng("base")
    psd = sp8072_psd(prof)
    # base Gaussian process: the source spectrum, LF limited at 3 Hz (nothing below is reproducible)
    g0 = shaped_noise(n, lambda f: psd(f) * lowcut(f, 3.0, 2) ** 2, r)
    # large-scale-structure modulation applied BEFORE steepening: louder passages steepen more, so the shock
    # content arrives in clusters (real recordings: crackle inter-event CV ~ 2, i.e. burstier than Poisson)
    common = mod_process(n, rng("mod_common"), 0.08, 2.5, 0.25)
    g = g0 * common
    f = np.maximum(freqs(n), 1e-3)
    lo_m, hi_m = xover(f)
    G = np.fft.rfft(g)
    g_lo = np.fft.irfft(G * lo_m, n)
    g_hi = np.fft.irfft(G * hi_m, n)
    hp_c = lowcut(f, CRACKLE_HP_HZ, 2)
    # crackle: only the ENERGETIC part of the waveform (< ENERGETIC_HZ) steepens into shocks (the weak HF turbulence is
    # merely carried); the shock train's HF (> 700 Hz) is the crackle layer. Shocks sit on the jet's own large
    # compressions (same realisation), so the layer fuses with jet_mix instead of sounding like a click track.
    g_energetic = np.fft.irfft(G * highcut(f, ENERGETIC_HZ, 4), n)
    s = steepen(g_energetic, STEEPEN_K)
    # finite shock rise time after propagation (~0.05-0.2 ms): gentle 2nd-order roll-off above CRACKLE_RISE_HZ
    shock_hf = np.fft.irfft(np.fft.rfft(s) * hp_c * highcut(f, CRACKLE_RISE_HZ, 1), n)
    lin_hf = np.fft.irfft(G * hp_c, n)
    # soft-limit the very largest shock spikes (a = CRACKLE_SOFTCLIP x rms): far-field shock trains are dense,
    # many moderate shocks rather than a few isolated giants (keeps dSk in the measured 3-8 range at distance)
    a = CRACKLE_SOFTCLIP * shock_hf.std()
    shock_hf = a * np.tanh(shock_hf / a)
    excess = shock_hf * (lin_hf.std() / shock_hf.std())  # at w=1 crackle carries the same HF energy as the linear jet
    # band-specific slow irregularity on top (rumble breathes more slowly and deeply than the mid band)
    m_lf = mod_process(n, rng("mod_lf"), 0.1, 2.0, 0.18)
    m_mid = mod_process(n, rng("mod_mid"), 0.3, 8.0, 0.08)
    # fast (3-20 Hz) roughness of the mid/HF noise driven by the jet's OWN low-frequency waveform: the large-scale
    # structures that carry the 10-30 Hz energy also gate the small-scale (HF) noise -> coherent 'crackling rumble'
    zf = np.fft.irfft(np.fft.rfft(g0) * lowcut(f, 3.0, 2) * highcut(f, 20.0, 2), n)
    zf = zf / (zf.std() + 1e-30)
    m_fast = np.exp(FAST_MOD_SIGMA * zf - 0.5 * FAST_MOD_SIGMA ** 2)
    m_mid = m_mid * m_fast / m_fast.mean()
    rumble = g_lo * m_lf
    jet = g_hi * m_mid
    # crackle arrives in bursts (real recordings: inter-event CV ~2): extra slow intermittency on the shock layer only
    crackle = excess * m_mid * mod_process(n, rng("crackle_mod"), 0.2, 3.0, 0.3)
    info = dict(steepen_K=STEEPEN_K, dSk_gaussian=dsp.deriv_skewness(g_hi, FS),
                crackle_events=dsp.crackle_events(jet + crackle + rumble, FS),
                dSk_crackle_alone=dsp.deriv_skewness(crackle, FS),
                dSk_jet_plus_0p6_crackle=dsp.deriv_skewness(jet + 0.6 * crackle, FS))
    return dict(rumble_lf=rumble, jet_mix=jet, crackle=crackle), info


def make_pad(n):
    r = rng("pad")
    f = freqs(n)
    # impingement/trench roar: broadband, peak ~70 Hz, -4 dB/oct above, 6 dB/oct below 30 Hz
    roar = shaped_noise(n, lambda f: (f / 70.0) ** 1.0 / (1 + (f / 70.0) ** 2.4) * lowcut(f, 12, 2) ** 2, r)
    roar = steepen(roar, 1.2) / roar.std()
    water = shaped_noise(n, lambda f: lowcut(f, 900, 2) ** 2 * highcut(f, 9000, 2) ** 2 / (1 + f / 4000.0), rng("water"))
    surge = mod_process(n, rng("pad_mod"), 0.1, 2.0, 0.45)
    flutter = mod_process(n, rng("water"), 2.0, 25.0, 0.35)
    x = roar * surge + 0.35 * water * flutter * surge ** 0.5
    return x


GEVS = [(20, 0.026), (50, 0.16), (800, 0.16), (2000, 0.026)]  # (Hz, g^2/Hz) GSFC-STD-7000 qualification shape
MODES = [(13.0, 6, 1.0), (21.0, 10, 0.8), (34.0, 12, 0.7), (48.0, 14, 0.6), (67.0, 15, 0.5), (95.0, 18, 0.45),
         (133.0, 20, 0.35), (190.0, 22, 0.3), (260.0, 25, 0.25), (410.0, 30, 0.2)]


def gevs_psd(f):
    fk, pk = np.log([a for a, _ in GEVS]), np.log([b for _, b in GEVS])
    lf = np.log(np.clip(f, 1, 1e5))
    p = np.exp(np.interp(lf, fk, pk))
    p = np.where(f < 20, 0.026 * (f / 20.0) ** 2, p)
    p = np.where(f > 2000, 0.026 * (f / 2000.0) ** -4, p)
    return p


def mode_gain(f):
    g = np.ones_like(f)
    for fm, q, a in MODES:
        g = g + a * 6.0 / (1 + (q * (f / fm - fm / f)) ** 2)
    return g


def make_struct(n):
    r = rng("struct")
    # GEVS shape (ACCELERATION PSD) with modal emphasis -> panel VELOCITY (radiated sound ~ velocity): divide by
    # (f/50)^2 above 50 Hz, so the cabin hears a low, heavy structure-borne roar; plus 8-20 Hz thrust shake
    x = shaped_noise(n, lambda f: gevs_psd(f) * mode_gain(f) * highcut(f, 1200, 2) ** 2 * (50.0 / np.maximum(f, 50.0)) ** 2
                     + 0.05 * lowcut(f, 6, 2) ** 2 * highcut(f, 20, 4) ** 2, r)
    m = mod_process(n, rng("struct_mod"), 0.2, 6.0, 0.18)
    return x * m


def make_aero(n):
    r = rng("aero")
    tbl = shaped_noise(n, lambda f: (f / 250.0) ** 1.5 / (1 + (f / 250.0) ** 3.2) * lowcut(f, 15, 2) ** 2, r)
    buffet = shaped_noise(n, lambda f: lowcut(f, 4, 2) ** 2 * highcut(f, 35, 3) ** 2, rng("aero_mod"))
    m = mod_process(n, rng("aero_mod"), 1.5, 12.0, 0.45)
    return tbl * m + 0.6 * buffet


# ------------------------------------------------------------------ one-shots
def env_ar(t, t0, a, d):
    """Attack-decay envelope (s): linear-in-dB attack a, exponential decay time-constant d."""
    x = np.zeros_like(t)
    m = t >= t0
    tt = t[m] - t0
    x[m] = np.where(tt < a, (tt / a) ** 2, np.exp(-(tt - a) / d))
    return x


def make_ignition(prof):
    """Engine-start transient one-shot, t=0 at the IGNITION event. Contains ONLY the transient parts: the
    steady jet roar comes from the loops following thrust. Not an explosion: no broadband impulsive 'bang'."""
    dur = 6.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    r = rng("ignition")
    x = np.zeros(n)
    # (1) igniter/valve tick ~2 s before main-stage ramp in the IMAP blast-mic record; we place it at
    #     t=0.05 s (the sim ramps immediately), 40 dB below full roar, soft band-limited crackle.
    tick = shaped_noise(n, lambda f: lowcut(f, 300, 2) ** 2 * highcut(f, 4000, 2) ** 2, r)
    x += 0.01 * tick * env_ar(t, 0.05, 0.004, 0.06)
    # (2) staggered engine light-ups (9 engines over ~0.5 s, pairs): rough-combustion swell, crackly
    starts = np.sort(0.35 + np.array([0, 0, 0.12, 0.12, 0.24, 0.24, 0.36, 0.36, 0.48]) + r.uniform(0, 0.03, 9))
    psd = sp8072_psd(prof)
    for k, ts in enumerate(starts):
        g = shaped_noise(n, lambda f: psd(f) * lowcut(f, 25, 2) ** 2, r)
        g = steepen(g, 2.8, n_steps=24) / g.std()
        x += 0.22 * g * env_ar(t, ts, 0.06, 0.45) * (1 + 0.3 * r.standard_normal())
    # (3) ignition overpressure: single slow compression-rarefaction (~0.8 s period, Mathews 2020) +
    #     a 16 Hz damped component; emitted at the flame-trench exit when thrust rises fastest.
    t_iop = 0.9
    tau = (t - t_iop)
    iop = -tau / 0.13 * np.exp(-0.5 * (tau / 0.13) ** 2)  # Gaussian-derivative (N-like) pulse
    iop += 0.45 * np.sin(2 * np.pi * 16 * tau) * np.exp(-np.clip(tau, 0, None) / 0.25) * (tau > 0)
    x += 0.5 * iop * (np.abs(tau) < 2.0)
    # gentle fade-out and DC removal
    x *= np.clip((dur - t) / 0.5, 0, 1)
    x -= x.mean()
    return x, dict(t_iop=t_iop, engine_starts=[round(float(s), 3) for s in starts])


def make_meco(prof):
    """Thrust tail-off: broadband jet noise whose spectrum slides down as chamber pressure decays,
    then a short low plume-collapse rumble. t=0 at MECO."""
    dur = 3.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    r = rng("meco")
    hi = shaped_noise(n, sp8072_psd(prof), r)
    lo = shaped_noise(n, lambda f: lowcut(f, 8, 2) ** 2 * highcut(f, 60, 2) ** 2, r)
    # time-varying low-pass via STFT: cutoff 3 kHz -> 150 Hz over 0.6 s
    fs_, ts_, Z = signal.stft(hi, FS, nperseg=2048)
    fc = 150 + 2850 * np.exp(-ts_ / 0.25)
    Z *= 1.0 / np.sqrt(1 + (fs_[:, None] / fc[None, :]) ** 4)
    _, hi2 = signal.istft(Z, FS, nperseg=2048)
    hi2 = hi2[:n]
    x = hi2 * np.exp(-t / 0.35) + 0.4 * lo * env_ar(t, 0.15, 0.25, 0.7)
    x *= np.clip((dur - t) / 0.4, 0, 1)
    return x, {}


def make_sep():
    """Stage separation as heard INSIDE the vehicle: pneumatic pushers / pyro release excite structural
    modes (a muffled clunk and ring-down), low-passed by the structure. Not airborne, not a gunshot."""
    dur = 2.5
    n = int(dur * FS)
    t = np.arange(n) / FS
    r = rng("sep")
    exc = np.zeros(n)
    for t0, a in ((0.02, 1.0), (0.09, 0.6), (0.31, 0.35)):
        exc += a * r.standard_normal(n) * env_ar(t, t0, 0.002, 0.012)
    y = np.zeros(n)
    for fm, q, a in MODES[1:] + [(560.0, 35, 0.12), (780.0, 40, 0.08)]:
        b, aa = signal.iirpeak(fm, q, FS)
        y += a * signal.lfilter(b, aa, exc)
    y = circ_filter(y, lambda f: highcut(f, 900, 2))
    y *= np.clip((dur - t) / 0.3, 0, 1)
    return y, {}


# ------------------------------------------------------------------ output
def norm_rms(x, dbfs=TARGET_RMS_DBFS, max_peak_dbfs=-1.0):
    """Normalise to an RMS target, lowering it if needed so the peak stays <= max_peak_dbfs (never clip)."""
    g = 10 ** (dbfs / 20) / (np.sqrt(np.mean(x * x)) + 1e-30)
    pk = np.abs(x).max() * g
    lim = 10 ** (max_peak_dbfs / 20)
    if pk > lim:
        g *= lim / pk
    return x * g, 20 * math.log10(g)


def write_wav(path, x):
    y = np.clip(x, -1, 1)
    wavfile.write(path, FS, np.round(y * 32767).astype(np.int16))


def encode_opus(wav, out, kbps):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "libopus", "-b:a", f"{kbps}k",
                    "-application", "audio", "-frame_duration", "20", out], check=True)


def asset_bands(x):
    f, p = dsp.welch_psd(x, FS, seg_s=4.0)
    third = dsp.band_levels(f, p)
    octv = dsp.band_levels(f, p, centres=dsp.OCT.astype(float), frac=1)
    return [round(float(v), 2) for v in third], [round(float(v), 2) for v in octv]


def main(encode=True):
    prof = am.load_profile()
    os.makedirs(OUT, exist_ok=True)
    n = int(LOOP_S * FS)
    jet, jinfo = make_jet_layers(prof, n)
    # normalise rumble+jet+crackle with ONE gain so their relative levels stay physical
    tot = jet["rumble_lf"] + jet["jet_mix"]
    _, g_common = norm_rms(tot, -18.0)
    loops = {}
    for k in ("rumble_lf", "jet_mix", "crackle"):
        loops[k] = jet[k] * 10 ** (g_common / 20)
    loops["pad_interaction"], _ = norm_rms(make_pad(n))
    loops["onboard_structure"], _ = norm_rms(make_struct(n))
    loops["aero_buffet"], _ = norm_rms(make_aero(n))
    meta = {
        "rumble_lf": dict(band="3-160 Hz (power-complementary split at 80 Hz)", role="LF exhaust rumble, slow irregular modulation (0.08-3 Hz)",
                          group="jet", sync_group="jet"),
        "jet_mix": dict(band="40 Hz - 20 kHz", role="turbulent jet-mixing noise, main broadband body", group="jet", sync_group="jet"),
        "crackle": dict(band=">400 Hz (shock content)", role="shock HF of the steepened <2 kHz (energetic) jet waveform; amplitude = w (see profile crackle.layer_gain_rule) x jet EQ",
                        group="jet", sync_group="jet"),
        "pad_interaction": dict(band="12 Hz - 10 kHz", role="deflector/trench impingement + water/steam; weight w_pad", group="pad"),
        "onboard_structure": dict(band="6 Hz - 1.2 kHz", role="structure-borne vibration heard aboard; ~thrust", group="onboard"),
        "aero_buffet": dict(band="4 Hz - 2 kHz", role="hull boundary-layer/transonic buffet; ~w_aero", group="onboard"),
    }
    manifest = {"sample_rate_hz": FS, "seeds": SEEDS, "loop_seconds": LOOP_S, "synthesis_info": jinfo,
                "calibration": {
                    "note": "rumble_lf, jet_mix and crackle share ONE gain (their relative levels are physical): rumble_lf+jet_mix "
                            "power sum = -18 dBFS RMS and represents the SP-8072 source spectrum at the reference condition "
                            "(stage 1, sea level, full thrust, U=2786 m/s, De=3.096 m, f_peak=18 Hz, peak-directivity, no absorption). "
                            "Other loops are normalised to -20 dBFS RMS; their absolute level comes from the mapping formulas.",
                    "jet_group_rms_dbfs": -18.0,
                    "runtime_eq_rule": "EQ_dB[b] = D[b] - A_group[b]; D = digital target band level from the model, A_group = the "
                                       "group's asset band level (layers[].octave_db power-summed over the group)",
                }, "layers": [], "oneshots": []}
    for name, x in loops.items():
        p = os.path.join(OUT, name + ".wav")
        write_wav(p, x)
        third, octv = asset_bands(x)
        entry = dict(name=name, file=f"assets/{name}.wav", file_web=f"assets/{name}.ogg", loop=True,
                     samples=len(x), seconds=round(len(x) / FS, 4),
                     rms_dbfs=round(float(20 * np.log10(np.sqrt(np.mean(x * x)))), 2),
                     peak_dbfs=round(float(20 * np.log10(np.abs(x).max())), 2),
                     deriv_skewness=round(dsp.deriv_skewness(x, FS), 3),
                     third_oct_hz=[float(v) for v in dsp.THIRD_OCT], third_oct_db=third,
                     octave_hz=[float(v) for v in dsp.OCT], octave_db=octv, **meta[name])
        manifest["layers"].append(entry)
    shots = {"ignition": make_ignition(prof), "meco_tail": make_meco(prof), "stage_sep_structural": make_sep()}
    smeta = {"ignition": dict(event="IGNITION", offset_s=0.0, emit_at="pad (trench exit for the IOP)", tag="MEASURED shape + ARTISTIC level",
                              role="igniter tick, staggered engine light-ups, ignition overpressure pulse"),
             "meco_tail": dict(event="MECO", offset_s=0.0, emit_at="vehicle (airborne, propagated)", tag="ROM/ARTISTIC",
                               role="thrust tail-off: spectrum slides down with chamber pressure + plume-collapse rumble"),
             "stage_sep_structural": dict(event="STAGE_SEP", offset_s=0.0, emit_at="onboard/cabin only", tag="ARTISTIC",
                                          role="separation clunk through the structure")}
    for name, (x, extra) in shots.items():
        x, _ = norm_rms(x, -20.0)
        p = os.path.join(OUT, name + ".wav")
        write_wav(p, x)
        third, octv = asset_bands(x)
        manifest["oneshots"].append(dict(name=name, file=f"assets/{name}.wav", file_web=f"assets/{name}.ogg", loop=False,
                                         samples=len(x), seconds=round(len(x) / FS, 3),
                                         rms_dbfs=round(float(20 * np.log10(np.sqrt(np.mean(x * x)))), 2),
                                         peak_dbfs=round(float(20 * np.log10(np.abs(x).max())), 2),
                                         octave_hz=[float(v) for v in dsp.OCT], octave_db=octv, **smeta[name], **extra))
    for e in manifest["layers"] + manifest["oneshots"]:
        wav = os.path.join(ROOT, "data", "audio", e["file"])
        if encode:
            encode_opus(wav, os.path.join(ROOT, "data", "audio", e["file_web"]), 96 if e["loop"] else 112)
        e["bytes_wav"] = os.path.getsize(wav)
        ogg = os.path.join(ROOT, "data", "audio", e["file_web"])
        e["bytes_web"] = os.path.getsize(ogg) if os.path.exists(ogg) else None
    manifest["web_format_note"] = ("Ogg/Opus 96 kbit/s (loops) / 112 kbit/s (one-shots). Opus decoders apply the pre-skip; after "
                                   "decodeAudioData trim/pad the buffer to layers[].samples and loop the whole buffer (loopStart=0, "
                                   "loopEnd=samples/48000). Safari < 17 lacks Ogg/Opus: fall back to the WAV.")
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("synthesis info:", jinfo)
    for e in manifest["layers"] + manifest["oneshots"]:
        print(f"  {e['name']:22s} rms {e['rms_dbfs']:6.1f} dBFS  peak {e['peak_dbfs']:6.1f}  wav {e['bytes_wav']/1e6:5.2f} MB  web {(e['bytes_web'] or 0)/1e3:6.0f} kB")
    return manifest


if __name__ == "__main__":
    main(encode="--no-encode" not in sys.argv)
