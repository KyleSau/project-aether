"""Signal analysis used for both the real reference recordings and our own renders.

Every metric here is computed the same way for references and renders so the comparison
figures are like-for-like. Nothing here assumes a calibrated input: levels are in dB re
digital full scale (dBFS) unless a calibration is supplied.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# Nominal 1/3-octave centre frequencies (IEC 61260), 6.3 Hz .. 20 kHz
THIRD_OCT = np.array([6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
                      500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
                      12500, 16000, 20000])
OCT = np.array([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])


def exact_third_centres():
    """Base-10 exact 1/3-octave mid-band frequencies matching THIRD_OCT."""
    k = np.round(10 * np.log10(THIRD_OCT / 1000.0))
    return 1000.0 * 10 ** (k / 10.0)


def welch_psd(x, fs, seg_s=4.0):
    n = int(min(len(x), max(256, seg_s * fs)))
    f, p = signal.welch(x, fs, nperseg=n, noverlap=n // 2, window="hann", detrend="constant")
    return f, p


def band_levels(f, psd, centres=None, frac=3):
    """Integrate a one-sided PSD into fractional-octave band levels (dB re 1 unit^2)."""
    if centres is None:
        centres = exact_third_centres()
    out = []
    df = f[1] - f[0]
    for fc in centres:
        lo, hi = fc * 2 ** (-0.5 / frac), fc * 2 ** (0.5 / frac)
        m = (f >= lo) & (f < hi)
        e = psd[m].sum() * df if m.any() else np.nan
        if m.sum() < 2:  # band narrower than PSD resolution: interpolate density instead
            e = np.interp(fc, f, psd) * (hi - lo)
        out.append(10 * np.log10(e + 1e-30))
    return np.array(out)


def rms_envelope(x, fs, win_s=0.125, hop_s=0.0625):
    w, h = int(win_s * fs), int(hop_s * fs)
    n = 1 + max(0, (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    seg = x[idx]
    t = (np.arange(n) * h + w / 2) / fs
    return t, np.sqrt((seg ** 2).mean(1))


def skew(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    s = v.std()
    return float((v ** 3).mean() / (s ** 3 + 1e-30))


def kurt(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    s = v.std()
    return float((v ** 4).mean() / (s ** 4 + 1e-30))


def deriv_skewness(x, fs):
    """Skewness of the pressure time derivative (Gee et al.; Reichman et al.). The sign convention
    assumes positive = compression; recordings with unknown polarity are reported as |dSk| too."""
    d = np.diff(np.asarray(x, dtype=np.float64)) * fs
    return skew(d)


def clip_fraction(x, thresh=0.985):
    """Fraction of samples at/near digital full scale, and longest run of such samples."""
    a = np.abs(x) >= thresh * max(1e-12, np.abs(x).max()) if np.abs(x).max() >= 0.98 else np.zeros(len(x), bool)
    if not a.any():
        return 0.0, 0
    d = np.diff(np.concatenate([[0], a.astype(np.int8), [0]]))
    runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    return float(a.mean()), int(runs.max())


def crest_factor_db(x):
    r = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return float(20 * np.log10(np.abs(x).max() / (r + 1e-30) + 1e-30))


def transient_stats(x, fs, k=4.0):
    """Exceedance rate of |x| > k*sigma (events per second, counted as separate runs) and kurtosis."""
    s = np.std(x)
    a = np.abs(x - np.mean(x)) > k * s
    d = np.diff(np.concatenate([[0], a.astype(np.int8)]))
    n_events = int((d == 1).sum())
    gauss_rate = None
    return dict(kurtosis=kurt(x), exceed_rate_per_s=n_events / (len(x) / fs), k_sigma=k,
                exceed_fraction=float(a.mean()), gaussian_fraction=float(2 * 0.5 * (1 - _erf(k / np.sqrt(2)))))


def _erf(z):
    from math import erf
    return erf(z)


def lf_hf(f, psd, lf=100.0, hf=1000.0):
    df = f[1] - f[0]
    tot = psd.sum() * df
    e_lf = psd[f < lf].sum() * df
    e_mid = psd[(f >= lf) & (f < hf)].sum() * df
    e_hf = psd[f >= hf].sum() * df
    return dict(lf_frac=float(e_lf / tot), mid_frac=float(e_mid / tot), hf_frac=float(e_hf / tot),
                lf_over_hf_db=float(10 * np.log10(e_lf / (e_hf + 1e-30) + 1e-30)))


def spectral_centroid(f, psd):
    return float((f * psd).sum() / (psd.sum() + 1e-30))


def modulation_spectrum(x, fs, env_fs=200.0, band=None, fmax=40.0):
    """Envelope (modulation) spectrum: Hilbert envelope of the (optionally band-passed) signal,
    decimated to env_fs, normalised by its mean (so the result is modulation depth^2 per Hz)."""
    y = np.asarray(x, dtype=np.float64)
    if band is not None:
        sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, y)
    env = np.abs(signal.hilbert(y)) if len(y) < 2 ** 22 else np.sqrt(signal.sosfiltfilt(
        signal.butter(2, 100, fs=fs, output="sos"), y * y).clip(0))
    q = int(fs // env_fs)
    sos = signal.butter(4, 0.8 * env_fs / 2, fs=fs, output="sos")
    env = signal.sosfiltfilt(sos, env)[::q]
    env = env / (env.mean() + 1e-30) - 1.0
    n = int(min(len(env), 8 * env_fs))
    fm, pm = signal.welch(env, env_fs, nperseg=n, noverlap=n // 2)
    m = (fm > 0) & (fm <= fmax)
    return fm[m], pm[m]


def hf_cutoff(f, psd, drop_db=30.0):
    """Highest frequency before the PSD falls drop_db below its 2-8 kHz median (MP3 low-pass / mic limit)."""
    ref = np.median(10 * np.log10(psd[(f > 2000) & (f < 8000)] + 1e-30))
    L = 10 * np.log10(psd + 1e-30)
    above = np.where((f > 4000) & (L > ref - drop_db))[0]
    return float(f[above[-1]]) if len(above) else float("nan")


def lf_rolloff(f, psd):
    """Frequency below which the PSD is >10 dB under its 20-60 Hz level (high-pass / lost infrasound)."""
    L = 10 * np.log10(psd + 1e-30)
    ref = np.median(L[(f >= 20) & (f <= 60)])
    lo = np.where((f > 0.3) & (f < 20) & (L < ref - 10))[0]
    return float(f[lo[-1]]) if len(lo) else float(f[1])


def segment_metrics(x, fs, label=""):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch_psd(x, fs, seg_s=2.0 if len(x) / fs < 20 else 4.0)
    cf, run = clip_fraction(x)
    bl = band_levels(f, p)
    pk = int(np.nanargmax(np.where(THIRD_OCT >= 10, bl, -np.inf)))
    fm, pm = modulation_spectrum(x, fs)
    return dict(
        label=label, duration_s=round(len(x) / fs, 3),
        rms_dbfs=float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)),
        crest_db=crest_factor_db(x), skewness=skew(x), deriv_skewness=deriv_skewness(x, fs),
        **{k: v for k, v in transient_stats(x, fs).items()},
        clip_fraction=cf, clip_longest_run=run,
        **lf_hf(f, p), centroid_hz=spectral_centroid(f, p),
        third_oct_peak_hz=float(THIRD_OCT[pk]),
        third_oct_db=[round(float(v), 2) for v in bl],
        hf_cutoff_hz=hf_cutoff(f, p), lf_rolloff_hz=lf_rolloff(f, p),
        hf_slope_db_per_oct=hf_slope_db_per_oct(f, p),
        crackle_rate_per_s=crackle_events(x, fs)["rate_per_s"], crackle_interval_cv=crackle_events(x, fs)["interval_cv"],
        **{"bang_" + k: v for k, v in bang_stats(x, fs).items()},
        mod_index_0p5_4hz=float(np.sqrt(pm[(fm >= 0.5) & (fm <= 4)].sum() * (fm[1] - fm[0]))),
        mod_index_4_20hz=float(np.sqrt(pm[(fm > 4) & (fm <= 20)].sum() * (fm[1] - fm[0]))),
    )


# ------------------------------------------------------------------ plotting helpers
def plot_panel(x, fs, title, path, t0=0.0, events=None, fmax=12000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray(x, dtype=np.float64)
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
    t = t0 + np.arange(len(x)) / fs
    dec = max(1, len(x) // 200000)
    ax[0].plot(t[::dec], x[::dec], lw=0.3, color="#2a5d8f")
    ax[0].set_ylabel("amplitude (FS)")
    ax[0].set_title(title + " - waveform")
    nper = 4096
    # float32 and 50 % overlap keep a 5-minute render's spectrogram under ~60 MB
    f, tt, S = signal.spectrogram(x.astype(np.float32), fs, nperseg=nper, noverlap=nper // 2, window="hann")
    S = 10 * np.log10(S + 1e-14)
    vmax = np.percentile(S, 99.5)
    ax[1].pcolormesh(t0 + tt, f, S, shading="auto", cmap="magma", vmin=vmax - 80, vmax=vmax)
    ax[1].set_yscale("symlog", linthresh=100)
    ax[1].set_ylim(5, min(fmax, fs / 2))
    ax[1].set_ylabel("Hz")
    ax[1].set_title("STFT spectrogram (dB, 4096-pt Hann)")
    te, e = rms_envelope(x, fs)
    ax[2].plot(t0 + te, 20 * np.log10(e + 1e-9), color="#b5442c", lw=0.8)
    ax[2].set_ylabel("RMS dBFS (125 ms)")
    ax[2].set_title("RMS envelope")
    if events:
        for name, te_ in events:
            for a in ax[:3]:
                a.axvline(te_, color="k", lw=0.5, ls="--")
            ax[2].text(te_, ax[2].get_ylim()[1] - 3, name, fontsize=7, rotation=90, va="top")
    fw, pw = welch_psd(x, fs)
    ax[3].semilogx(fw[1:], 10 * np.log10(pw[1:] + 1e-20), color="#2a5d8f", lw=0.8, label="Welch PSD")
    bl = band_levels(fw, pw)
    ax[3].semilogx(THIRD_OCT, bl - 10 * np.log10(THIRD_OCT * 0.2316), "o-", ms=3, color="#d08a1e",
                   label="1/3-oct band level / bandwidth")
    ax[3].set_xlim(2, fs / 2)
    ax[3].set_xlabel("Hz")
    ax[3].set_ylabel("dB re FS^2/Hz")
    ax[3].legend(fontsize=8)
    ax[3].grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hf_slope_db_per_oct(f, psd, lo=250.0, hi=4000.0):
    """Least-squares slope of the PSD (dB per octave, per-Hz density) between lo and hi.
    Robust to the unknown LF response of broadcast microphones."""
    m = (f >= lo) & (f <= hi)
    return float(np.polyfit(np.log2(f[m]), 10 * np.log10(psd[m] + 1e-30), 1)[0])


def crackle_events(x, fs, k=5.0, min_sep_s=0.002):
    """Shock-like events: peaks of +dp/dt above k sigma. Returns rate and the coefficient of variation of
    the inter-event intervals (Poisson-like ~1, a 'machine-gun' periodic train -> 0)."""
    d = np.diff(np.asarray(x, dtype=np.float64)) * fs
    s = d.std()
    pk, _ = signal.find_peaks(d, height=k * s, distance=max(1, int(min_sep_s * fs)))
    if len(pk) < 3:
        return dict(rate_per_s=len(pk) / (len(x) / fs), interval_cv=float("nan"))
    iv = np.diff(pk) / fs
    return dict(rate_per_s=float(len(pk) / (len(x) / fs)), interval_cv=float(iv.std() / iv.mean()))


def k_weight(x, fs):
    """ITU-R BS.1770-4 K-weighting (pre-filter shelf + RLB high-pass), coefficients derived for any fs."""
    import math
    # stage 1: high shelf (f0 = 1681.97 Hz, G = +3.99984 dB, Q = 0.7071752)
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / fs)
    Vh, Vb = 10 ** (G / 20), (10 ** (G / 20)) ** 0.499666774155
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    # stage 2: high-pass (f0 = 38.135 Hz, Q = 0.5003270)
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / fs)
    a0 = 1 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x))


def lufs_integrated(x, fs):
    """Integrated loudness (LUFS, mono) per ITU-R BS.1770-4: 400 ms blocks, 75 % overlap, -70 LUFS absolute and
    -10 LU relative gates. Also returns the max momentary (400 ms) loudness."""
    y = k_weight(np.asarray(x, dtype=np.float64), fs)
    n, h = int(0.4 * fs), int(0.1 * fs)
    if len(y) < n:
        return float("nan"), float("nan")
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    starts = np.arange(0, len(y) - n + 1, h)
    z = (c[starts + n] - c[starts]) / n
    L = -0.691 + 10 * np.log10(z + 1e-30)
    g = z[L > -70]
    if not len(g):
        return float("-inf"), float(L.max())
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = z[(L > -70) & (L > rel)]
    return float(-0.691 + 10 * np.log10(g2.mean())), float(L.max())


def bang_stats(x, fs, hp_hz=1000.0, prominence_db=12.0, min_sep_s=0.01):
    """Distinct-bang detector (how many crackle events stand out as individual pops): >hp_hz band, 2 ms RMS envelope,
    events where it exceeds its 0.5 s running median by >= prominence_db, at least min_sep_s apart. Returns the rate,
    the inter-event CV (periodic 'machine gun' -> 0) and the median prominence of the detected bangs."""
    from scipy.ndimage import median_filter
    y = signal.sosfilt(signal.butter(4, hp_hz, btype="high", fs=fs, output="sos"), np.asarray(x, dtype=np.float64))
    h = int(0.001 * fs)
    e = np.sqrt(np.convolve(y * y, np.ones(2 * h) / (2 * h), mode="same")[::h] + 1e-24)
    L = 20 * np.log10(e)
    med = median_filter(L, size=int(0.5 * fs / h) | 1, mode="nearest")
    prom = L - med
    pk, _ = signal.find_peaks(prom, height=prominence_db, distance=max(1, int(min_sep_s * fs / h)))
    dur = len(x) / fs
    if len(pk) < 3:
        return dict(rate_per_s=len(pk) / dur, interval_cv=float("nan"), median_prominence_db=float("nan"))
    iv = np.diff(pk) * h / fs
    return dict(rate_per_s=float(len(pk) / dur), interval_cv=float(iv.std() / iv.mean()),
                median_prominence_db=float(np.median(prom[pk])))
