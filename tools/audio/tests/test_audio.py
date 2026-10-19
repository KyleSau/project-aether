"""Objective tests for the launch-audio core: python -m pytest tools/audio/tests -q"""
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(TOOLS, "..", ".."))
sys.path.insert(0, TOOLS)

import acoustic_model as am  # noqa: E402
import dsp  # noqa: E402
import render_launch as rl  # noqa: E402

FS = 48000


@pytest.fixture(scope="module")
def prof():
    return am.load_profile()


# ------------------------------------------------------------------ physics pieces
@pytest.mark.parametrize("f,expected", [(1000, 4.98), (2000, 9.02), (4000, 22.9), (8000, 76.6)])
def test_iso9613_standard_table(f, expected):
    """ISO 9613-1 Table 1 at 20 degC, 70 % RH, 101.325 kPa (dB/km)."""
    T, P = 293.15, 101325.0
    a = am.iso9613_alpha(f, T, P, am.molar_h2o_from_rh(70.0, T, P)) * 1000
    assert a == pytest.approx(expected, rel=0.05)


def test_sp8072_spectrum_unit_area(prof):
    st = np.geomspace(1e-4, 50, 20000)
    area = np.trapezoid(10 ** (am.spectrum_level_db(prof, st) / 10) * st, np.log(st))
    assert area == pytest.approx(1.0, rel=0.03)


def test_peak_frequency_matches_falcon9_order(prof):
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    st = am.source_state(prof, veh, dict(phase=1, thrust=7.473e6, mdot=2682.0, P=101325.0, T=288.15, a=340.3, throttle=1.0))
    assert 15.0 < st["f_peak"] < 25.0          # Falcon 9 measured ~20 Hz (Mathews et al. 2020)
    assert 195.0 < st["Lw"] < 200.0            # eta = 0.5 % of mechanical power


def test_directivity_peak_at_falcon9_angle(prof):
    d = prof["directivity"]
    ang = np.arange(0, 181)
    # power-average over Strouhal curves ~ OASPL directivity
    tot = sum(10 ** (np.interp(ang, d["angles_deg"], d["di_db"][str(k)]) / 10) for k in d["strouhal"])
    assert 55 <= ang[np.argmax(tot)] <= 75


# ------------------------------------------------------------------ assets / loop seams
def _loops(prof):
    return [e for e in prof["layers"] if e["loop"]]


def test_assets_exist_and_match_manifest(prof):
    for e in prof["layers"] + prof["oneshots"]:
        fs, x = wavfile.read(os.path.join(ROOT, "data", "audio", e["file"]))
        assert fs == 48000 and x.dtype == np.int16 and x.ndim == 1
        assert len(x) == e["samples"]
        assert np.abs(x).max() < 32767, f"{e['name']} clips"
        assert os.path.exists(os.path.join(ROOT, "data", "audio", e["file_web"]))


def test_loop_seams_are_continuous(prof):
    """The wrap-around step must look like any interior step, and the seam frame must not stand out spectrally."""
    for e in _loops(prof):
        fs, x = wavfile.read(os.path.join(ROOT, "data", "audio", e["file"]))
        x = x.astype(np.float64)
        d = np.diff(x)
        seam = x[0] - x[-1]
        assert abs(seam - d.mean()) < 5 * d.std(), e["name"]
        # spectral flux: frames straddling the seam vs interior frames of the looped signal
        y = np.concatenate([x[-FS:], x[:FS]])
        f, t, S = signal.stft(y, FS, nperseg=1024, noverlap=768)
        P = np.abs(S) ** 2
        flux = np.sum(np.diff(np.log(P + 1e-9), axis=1) ** 2, axis=0)
        k = int(np.argmin(np.abs(t[1:] - 1.0)))
        seam_flux = flux[max(0, k - 2):k + 3].max()
        assert seam_flux <= np.percentile(flux, 99.5), e["name"]


def test_loops_are_not_short_repeats(prof):
    """No internal period shorter than the file: autocorrelation away from lag 0 stays small."""
    for e in _loops(prof):
        fs, x = wavfile.read(os.path.join(ROOT, "data", "audio", e["file"]))
        x = x[:8 * FS].astype(np.float64)
        x = signal.sosfilt(signal.butter(2, 200, btype="high", fs=FS, output="sos"), x)
        X = np.fft.rfft(x, 2 * len(x))
        ac = np.fft.irfft(X * np.conj(X))[:len(x)]
        ac /= ac[0]
        assert np.abs(ac[int(0.05 * FS):int(4 * FS)]).max() < 0.2, e["name"]


# ------------------------------------------------------------------ propagation / renderer
def _static_tel(thrust_on=0.0, dur=16.0, vz=0.0, start_h=0.0):
    t = np.arange(-1.0, dur, 0.05)
    n = len(t)
    on = t >= thrust_on
    h = start_h + vz * np.clip(t, 0, None)
    tel = dict(t=t, phase=np.ones(n), thrust=np.where(on, 7.473e6, 0.0), mdot=np.where(on, 2682.0, 0.0), throttle=on * 1.0,
               P=np.full(n, 101325.0), T=np.full(n, 288.15), rho=np.full(n, 1.225), a=np.full(n, 340.294), h=h,
               mach=np.zeros(n), qbar=np.zeros(n), noz_e=np.zeros(n), noz_n=np.zeros(n), noz_u=h,
               ax_e=np.zeros(n), ax_n=np.zeros(n), ax_u=-np.ones(n))
    return tel


def test_propagation_delay_matches_distance():
    tel = _static_tel()
    lis = (3400.0, 0.0, 1.7)
    y, info, grid, _ = rl.render("custom", pos=lis, mode="ground", t0=-1.0, t1=15.0, tel=tel, events=[], verbose=False,
                                 return_debug=True)
    i0 = int(np.searchsorted(tel["t"], 1.0))
    src = np.array([0, 0, 1.0])
    expected = np.linalg.norm(np.array(lis) - src) / am.c_path(src, lis)
    assert grid["delay"][i0] == pytest.approx(expected, abs=0.02)
    # onset of the rendered sound (pad + jet start at t=0) arrives at ~ r/c
    tt, e = dsp.rms_envelope(y, FS, 0.01, 0.005)
    L = 20 * np.log10(e + 1e-12)
    onset = info["t0"] + tt[np.argmax(L > L.max() - 30)]
    assert onset == pytest.approx(min(grid["delay"][i0], grid["delay_pad"][i0]), abs=0.1)


def test_doppler_only_from_emission_time():
    """Receding source at constant speed v: the loop read-rate d(tau)/d(t_r) must equal c/(c+v) (no extra pitch)."""
    v = 100.0
    tel = _static_tel(vz=v, dur=20.0, start_h=200.0)
    tau = tel["t"]
    lis = np.array([0.0, 0.0, 1.7])
    c = 340.294
    d = np.abs(tel["noz_u"] - lis[2])
    tr = tau + d / c
    t_r = np.linspace(5, 15, 1000)
    tau_r = np.interp(t_r, tr, tau)
    rate = np.gradient(tau_r, t_r)[100:-100].mean()
    assert rate == pytest.approx(c / (c + v), rel=1e-3)


def test_event_sync_ignition(prof):
    tel = _static_tel(thrust_on=0.0)
    lis = (1000.0, 0.0, 1.7)
    y, info, grid, _ = rl.render("custom", pos=lis, mode="ground", t0=-1.0, t1=8.0, tel=tel, events=[("IGNITION", 0.0)],
                                 verbose=False, return_debug=True)
    name, t_arr = info["oneshots"][0]
    assert name == "ignition"
    pad_src = np.array([0.0, 60.0, 1.0])
    expected = np.linalg.norm(np.array(lis) - pad_src) / am.c_path(pad_src, lis)
    assert t_arr == pytest.approx(expected, abs=0.06)


def test_distance_dependent_filtering(prof):
    """HF/LF balance must fall monotonically with distance (ISO 9613-1 absorption + spreading)."""
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    st = am.source_state(prof, veh, dict(phase=1, thrust=7.473e6, mdot=2682.0, P=101325.0, T=288.15, a=340.3, throttle=1.0))
    ratios, oas = [], []
    for r in (500.0, 2000.0, 5000.0, 20000.0):
        L, g = am.band_spl(prof, st, np.array([0, 0, 200.0]), np.array([0, 0, -1.0]),
                           np.array([r, 0, 1.7]), rl.F_LO, rl.F_HI, rl.F_C)
        i63, i4k = int(np.argmin(np.abs(rl.F_C - 63))), int(np.argmin(np.abs(rl.F_C - 4000)))
        ratios.append(L[i4k] - L[i63])
        oas.append(10 * np.log10(np.sum(10 ** (L / 10))))
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert all(a > b for a, b in zip(oas, oas[1:]))


def test_vacuum_listener_is_silent():
    assert am.listener_impedance_db(200e3) == -np.inf
    assert am.onboard_air_weight(5.0, 1e-6, 300.0) < 1e-3


def test_anti_repetition_breaks_loop_period(prof):
    """A naive loop repeats its loudness pattern every N samples; the two-voice reader must not."""
    fs, x = wavfile.read(os.path.join(ROOT, "data", "audio", "assets", "onboard_structure.wav"))
    x = x.astype(np.float64) / 32768
    n = len(x)
    pos = np.arange(int(150 * FS), dtype=np.float64)

    def env_corr(y):
        _, e = dsp.rms_envelope(y, FS, 0.5, 0.25)
        L = 20 * np.log10(e)
        L -= L.mean()
        lag = int(round(n / FS / 0.25))
        return float(np.corrcoef(L[:-lag], L[lag:])[0, 1])

    assert env_corr(rl.read_loop(x, pos)) > 0.95
    assert env_corr(rl.read_loop_ar(x, pos, prof)) < 0.5
