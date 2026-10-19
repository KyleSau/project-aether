"""Write data/audio/acoustic_profile.json: the output contract shared by the browser and Unreal runtimes.

Every number below carries a provenance tag:
  MEASURED   - taken from a published measurement (citation given) or from our own analysis of a
               downloaded public-domain recording (tools/audio/references.py, numbers in
               data/audio/reference_analysis.json)
  LITERATURE - an empirical/standard model from the literature (NASA SP-8072, ISO 9613-1, ...)
  ROM        - reduced-order model chosen here, physically motivated but not validated against data
  ARTISTIC   - a listening/design choice with no physical claim

Run:  python tools/audio/make_profile.py   (build_all.py does this for you)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "data", "audio", "acoustic_profile.json")
ASSET_MANIFEST = os.path.join(ROOT, "data", "audio", "assets", "manifest.json")

# --------------------------------------------------------------------------------------------
# NASA SP-8072 (Eldred 1971) Fig. 5: normalised relative sound-power spectrum
#   10 log10( W(f) Ue / (W_OA de) ) vs Strouhal fde/Ue, faired curve for standard chemical rockets.
# Digitised BY EYE from the scanned figure (NTRS 19710023719, p.8), estimated reading error +-1.5 dB.
# Renormalised numerically below so that the integral over St is exactly 1 (the figure's definition).
SP8072_SPECTRUM_ST = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
SP8072_SPECTRUM_DB = [-0.5, 7.3, 10.3, 11.3, 11.2, 8.8, 4.0, -1.0, -7.0, -12.5, -19.0, -27.0]

# NASA SP-8072 Fig. 10: directivity index (dB) of far-field noise vs angle from the exhaust-flow
# direction, for five Strouhal numbers. Digitised by eye (p.12), +-1 dB / +-3 deg. The figure starts
# at 20 deg; values below 20 deg are an EXTRAPOLATION into the cone of relative silence (ROM).
SP8072_DI_ANGLES = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180]
SP8072_DI = {
    0.004: [-6.0, -1.0, 4.0, 7.6, 6.0, 3.8, 1.5, -3.3, -7.0, -9.8, -12.0, -14.0, -16.7],
    0.0125: [-9.0, -4.5, 0.0, 3.5, 5.9, 4.8, 3.0, -1.7, -6.0, -9.0, -11.5, -13.5, -15.8],
    0.04: [-10.0, -5.5, -1.0, 2.2, 4.6, 5.4, 4.0, -0.5, -5.0, -8.0, -10.5, -12.5, -14.5],
    0.125: [-10.5, -6.0, -1.5, 1.5, 3.8, 5.0, 4.6, 0.5, -4.0, -7.5, -9.5, -11.5, -13.5],
    0.4: [-11.0, -6.6, -2.2, 0.6, 2.5, 4.2, 4.4, 1.8, -2.5, -6.5, -8.5, -10.5, -12.5],
}

# Falcon 9 measured OASPL directivity (Mathews et al. 2020, J. Acoust. Soc. Korea 39(4) 322,
# doi:10.7776/ASK.2020.39.4.322): peak ~65 deg from the plume axis, 6-dB-down region ~45-80 deg.
F9_PEAK_DEG = 65.0
F9_6DB_DEG = (45.0, 80.0)


def _normalised_sp8072():
    st = np.array(SP8072_SPECTRUM_ST)
    db = np.array(SP8072_SPECTRUM_DB)
    lst = np.log(st)
    grid = np.linspace(lst[0], lst[-1], 4000)
    w = 10 ** (np.interp(grid, lst, db) / 10)
    area = np.trapezoid(w * np.exp(grid), grid)  # integral over St (d St = St d lnSt)
    corr = -10 * math.log10(area)
    return (db + corr).tolist(), corr


def _oaspl_directivity(spec_db, De=3.1, U=2790.0):
    """Power-weighted OASPL directivity implied by SP-8072 Figs 5+10 for a given De/U."""
    angles = np.arange(0, 181, 1.0)
    f = np.geomspace(2, 20000, 400)
    st = f * De / U
    w = 10 ** (np.interp(np.log(st), np.log(SP8072_SPECTRUM_ST), spec_db) / 10) / f  # per-Hz density shape
    keys = sorted(SP8072_DI)
    tab = np.array([np.interp(angles, SP8072_DI_ANGLES, SP8072_DI[k]) for k in keys])
    lk = np.log(keys)
    out = []
    for i, a in enumerate(angles):
        di = np.interp(np.log(np.clip(st, keys[0], keys[-1])), lk, tab[:, i])
        out.append(10 * math.log10(np.trapezoid(w * 10 ** (di / 10) * f, np.log(f)) / np.trapezoid(w * f, np.log(f))))
    return angles, np.array(out)


def _warp_angles(peak_src, lo_src, hi_src):
    """Piecewise-linear angle warp mapping SP-8072's OASPL peak and 6-dB points onto the Falcon 9
    measured ones, keeping 0 and 180 fixed. Returns (src_knots, dst_knots)."""
    return [0.0, lo_src, peak_src, hi_src, 180.0], [0.0, F9_6DB_DEG[0], F9_PEAK_DEG, F9_6DB_DEG[1], 180.0]


def build_directivity(spec_db):
    angles, oa = _oaspl_directivity(spec_db)
    ip = int(np.argmax(oa))
    pk = oa[ip]
    lo = angles[:ip][oa[:ip] < pk - 6][-1] if (oa[:ip] < pk - 6).any() else 20.0
    hi = angles[ip:][oa[ip:] < pk - 6][0] if (oa[ip:] < pk - 6).any() else 120.0
    src, dst = _warp_angles(float(angles[ip]), float(lo), float(hi))
    out_angles = list(range(0, 181, 10))
    adj = {}
    for k, v in SP8072_DI.items():
        # value at destination angle a = SP-8072 value at the pre-image angle
        pre = np.interp(out_angles, dst, src)
        adj[str(k)] = [round(float(x), 2) for x in np.interp(pre, SP8072_DI_ANGLES, v)]
    _, oa_adj = (None, None)
    return dict(
        sp8072_oaspl_peak_deg=float(angles[ip]), sp8072_oaspl_6db=[float(lo), float(hi)],
        warp_src_deg=src, warp_dst_deg=dst, angles_deg=out_angles, strouhal=[float(k) for k in sorted(SP8072_DI)],
        di_db={str(k): adj[str(k)] for k in sorted(SP8072_DI)},
        sp8072_raw={"angles_deg": SP8072_DI_ANGLES, "di_db": {str(k): v for k, v in sorted(SP8072_DI.items())}},
        oaspl_sp8072_db=[round(float(x), 2) for x in np.interp(out_angles, angles, oa)],
    )


def iso9613_alpha_table(T=293.15, P=101325.0, RH=70.0):
    import acoustic_model as am
    f = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    return f, [round(float(am.iso9613_alpha(fi, T, P, am.molar_h2o_from_rh(RH, T, P)) * 1000), 4) for fi in f]


def main():
    spec_db, corr = _normalised_sp8072()
    dirx = build_directivity(spec_db)
    with open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8") as fh:
        veh = json.load(fh)
    e1, e2 = veh["engines"]["stage1"], veh["engines"]["stage2"]
    assets = {}
    if os.path.exists(ASSET_MANIFEST):
        with open(ASSET_MANIFEST, encoding="utf-8") as fh:
            assets = json.load(fh)
    ref_stats = {}
    p_ref = os.path.join(ROOT, "data", "audio", "reference_analysis.json")
    if os.path.exists(p_ref):
        with open(p_ref, encoding="utf-8") as fh:
            ref_stats = json.load(fh).get("summary", {})
    fa, aa = iso9613_alpha_table()

    def de_cluster(e):
        return math.sqrt(e["count"]) * 2 * math.sqrt(e["ae_m2"] / math.pi)

    prof = {
        "schema": "aether-acoustic-profile/1",
        "generator": "tools/audio/make_profile.py",
        "units": "SI; levels in dB re 20 uPa (SPL) or 1 pW (sound power); angles in degrees",
        "provenance_tags": {
            "MEASURED": "published measurement or our analysis of a downloaded public-domain recording",
            "LITERATURE": "empirical/standard model from the literature",
            "ROM": "reduced-order model chosen here; physically motivated, not validated",
            "ARTISTIC": "listening/design choice, no physical claim",
        },
        "sample_rate_hz": 48000,
        # ---------------------------------------------------------------- telemetry contract
        "telemetry": {
            "_note": "Fields each runtime must supply every audio control tick (>= 30 Hz). 'emission' fields are "
                     "looked up at the EMISSION time t_e (see propagation), so keep >= 20 s of history.",
            "per_tick": {
                "t": {"unit": "s", "ts": "Telemetry.t", "unreal": "Aether::Telemetry::T", "python": "tel['t']"},
                "phase": {"unit": "enum", "ts": "Telemetry.phase", "unreal": "Telemetry::Phase", "python": "tel['phase']",
                          "values": "0 PAD,1 S1_BURN,2 COAST_SEP,3 COAST_S2,4 S2_BURN,5 ORBIT,6 S2_DEPLETED"},
                "thrust": {"unit": "N", "ts": "Telemetry.thrust", "unreal": "Telemetry::Thrust", "python": "tel['thrust']",
                           "note": "total thrust of the ACTIVE stage"},
                "mdot": {"unit": "kg/s", "ts": "Telemetry.mdot", "unreal": "Telemetry::Mdot", "python": "tel['mdot']",
                         "note": "total mass flow of the active stage"},
                "throttle": {"unit": "0..1", "ts": "Telemetry.throttle", "unreal": "Telemetry::Throttle", "python": "tel['throttle']"},
                "pa": {"unit": "Pa", "ts": "Telemetry.P", "unreal": "Telemetry::P", "python": "tel['P']",
                       "note": "ambient pressure at the VEHICLE"},
                "T_amb": {"unit": "K", "ts": "Telemetry.T", "unreal": "Telemetry::AtmT", "python": "tel['T']"},
                "rho": {"unit": "kg/m3", "ts": "Telemetry.rho", "unreal": "Telemetry::Rho", "python": "tel['rho']"},
                "a": {"unit": "m/s", "ts": "Telemetry.a", "unreal": "Telemetry::A", "python": "tel['a']"},
                "h": {"unit": "m", "ts": "Telemetry.h", "unreal": "Telemetry::H", "python": "tel['h']"},
                "mach": {"unit": "-", "ts": "Telemetry.mach", "unreal": "Telemetry::Mach", "python": "tel['mach']"},
                "qbar": {"unit": "Pa", "ts": "Telemetry.qbar", "unreal": "Telemetry::QBar", "python": "tel['qbar']"},
                "nozzle_pos": {"unit": "m (listener frame)", "ts": "Effects input base (render world, float64)",
                               "unreal": "vehicle actor nozzle-exit world location (cm -> m)",
                               "python": "ECI y[0:3] - pad, rotated into local ENU (render_launch.py)",
                               "note": "active stage nozzle exit plane centre"},
                "plume_axis": {"unit": "unit vector", "ts": "-axis (EffectInput.axis negated)",
                               "unreal": "-(body x axis) from the ascent quaternion", "python": "-R[:,0] from y[6:10]",
                               "note": "direction the EXHAUST flows (opposite the nose)"},
                "listener_pos": {"unit": "m", "note": "camera/listener position in the same frame"},
                "listener_pa": {"unit": "Pa", "note": "ambient pressure at the LISTENER (USSA-1976 at listener altitude)"},
                "listener_mode": {"unit": "enum", "values": ["ground", "onboard", "cabin", "chase", "vacuum"]},
            },
            "from_vehicle_json": {
                "engine_count": "engines.stageN.count", "ae_m2": "engines.stageN.ae_m2", "ve_m_s": "engines.stageN.ve_m_s",
                "pe_pa": "engines.stageN.pe_pa", "chamber_pressure_pa": "engines.stageN.chamber_pressure_pa",
                "gamma_exit": "engines.stageN.gamma_exit",
            },
            "events": {"_note": "names as emitted by sim.ts / AetherSim.cpp / sim.py",
                       "IGNITION": "start ignition one-shot", "LIFTOFF": "pad clamp release (pad layer keeps its height law)",
                       "MECO": "start meco_tail one-shot (airborne + onboard)", "STAGE_SEP": "start stage_sep_structural (onboard/cabin only)",
                       "SES": "stage-2 start (vacuum engine: onboard structure only)", "SECO": "onboard structure fades with thrust"},
        },
        # ---------------------------------------------------------------- source model
        "source": {
            "acoustic_efficiency": {"value": 0.005, "tag": "LITERATURE",
                                    "source": "NASA SP-8072 s2.1.1: 0.2-1.0 % for most data, 0.5 % most probable, 1 % conservative",
                                    "measured_alternatives": {"SLS_Artemis_I": 0.0033,
                                                              "_src": "Kellison & Gee 2023, JASA-EL, doi:10.1121/10.0022538 (OAPWL 202.4 dB, ~0.33 %)"}},
            "jet_velocity": {"formula": "U = thrust / mdot  (effective exhaust velocity incl. pressure thrust)", "tag": "ROM",
                             "note": "SP-8072 uses the fully expanded exit velocity; thrust/mdot is within ~3 % of it for this vehicle and is "
                                     "available identically in all three runtimes."},
            "mechanical_power": {"formula": "P_mech = 0.5 * mdot * U^2", "unit": "W"},
            "sound_power": {"formula": "W = eta * P_mech * w_src(pa)", "Lw_formula": "Lw = 10*log10(W) + 120  [dB re 1 pW]",
                            "throttle_and_engines": "mdot and thrust already include throttle and engine count; W scales ~ throttle * U^2"},
            "source_continuum_cutoff": {"formula": "w_src = clamp((log10(pa) - log10(pa_lo)) / (log10(pa_hi) - log10(pa_lo)), 0, 1)",
                                        "pa_lo": 0.1, "pa_hi": 10.0, "tag": "ARTISTIC",
                                        "note": "keeps W from being radiated into a near-vacuum; irrelevant below ~65 km where 1-stage noise IS heard "
                                                "on the ground (Mathews et al. 2020 measured the MECO level drop at 8.3 km)"},
            "spectrum": {
                "tag": "LITERATURE",
                "source": "NASA SP-8072 Fig. 5, faired curve, digitised by eye (+-1.5 dB), renormalised to unit area",
                "strouhal": SP8072_SPECTRUM_ST, "level_db": [round(x, 2) for x in spec_db],
                "normalisation_correction_db": round(corr, 2),
                "definition": "10*log10( W(f) * U / (W * De) ) at St = f*De/U ; band power W_b = W * integral over band of 10^(level/10) dSt",
                "peak_strouhal": 0.02,
                "cluster_diameter": {"formula": "De = sqrt(n_engines) * d_e_single", "source": "SP-8072 s2.1.2 / eq.(3) (equivalent area)"},
                "under_expanded_diameter": {"formula": "D_spec = max(De, sqrt(n) * Dj)  with Dj from plume ROM (physics/plume_rom.py)",
                                            "tag": "ROM", "note": "lowers the spectrum as the plume balloons at altitude; not validated"},
                "reference_condition": {"stage": 1, "U_m_s": 2786.0, "De_m": round(de_cluster(e1), 3),
                                        "f_peak_hz": round(0.02 * 2786.0 / de_cluster(e1), 2),
                                        "note": "stage 1, sea level, full thrust: f_peak = 0.02*U/De"},
                "check_falcon9": {"tag": "MEASURED", "value_hz": 20,
                                  "source": "Mathews et al. 2020 (peak ~20 Hz at peak directivity, 8.3 km); Gee et al. 2023 Artemis-I mean 1/3-oct peak 20 Hz"},
            },
            "core_length": {"formula": "x_t = 3.45 * De * (1 + 0.38 * M_e)^2  (De = cluster-equivalent diameter, merged flow)", "tag": "LITERATURE",
                            "source": "curve fit commonly used for SP-8072 Fig. 11 (core length vs exit Mach; the report gives the plot, "
                                      "not the formula - fit gives 19.5 De at M_e=3.63 (plume ROM, stage 1), inside the figure's rocket data band)",
                            "apparent_source_offset": "x_src = 1.5 * x_t along the plume axis from the nozzle exit (SP-8072 Fig. 12: maxima at 1-2 core lengths)",
                            "M_e_source": "plume ROM Me (physics/plume_rom.py) at the current throttle"},
        },
        # ---------------------------------------------------------------- directivity
        "directivity": dict(
            tag="LITERATURE+MEASURED",
            source="SP-8072 Fig. 10 shapes (digitised), angle-warped so the implied OASPL peak (and 6-dB points) sit at the Falcon 9 "
                   "measured 65 deg (45-80 deg) of Mathews et al. 2020/2021. Angle measured from the EXHAUST-FLOW direction.",
            usage="DI(f) = bilinear interp in (log St, angle) with St = f*De/U clamped to [0.004, 0.4]; add to band SPL",
            **dirx,
        ),
        # ---------------------------------------------------------------- propagation
        "propagation": {
            "geometric_spreading": {
                "formula": "Lp_band = Lw_band + DI - 10*log10(4*pi*r_eff^2) + 10*log10(rho_L*c_L/(rho0*c0)) - A_abs + G_ground",
                "r_eff": "sqrt(r^2 + r0^2), r0 = x_t (near-field limit: source extent ~ supersonic core length)",
                "rho0c0": 413.3, "tag": "LITERATURE (SP-8072 eq.5) + ROM near-field limit",
                "note": "rho_L*c_L at the listener (intensity is conserved along the ray through the stratified atmosphere)"},
            "emission_time": {
                "formula": "t_r = t_e + |x_src(t_e) - x_L| / c_path ; solve for t_e each tick (fixed-point, 3 iterations from last t_e)",
                "c_path": "harmonic mean of USSA-1976 sound speed along the straight ray (wind ignored)",
                "doppler_policy": "Doppler comes ONLY from reading the source at t_e (variable delay line / control-rate time warp of the "
                                  "source playback). Do NOT add a separate pitch shift. Broadband loops: playbackRate = dt_e/dt_r "
                                  "clamped to [0.5, 1.5].", "tag": "LITERATURE"},
            "atmospheric_absorption": {
                "model": "ISO 9613-1:1993 pure-tone absorption, evaluated at band centre, integrated along the ray with local T, p, humidity",
                "tag": "LITERATURE",
                "formula": {
                    "alpha_dB_per_m": "8.686*f^2*[1.84e-11*(pr/pa)*(T/T0)^0.5 + (T/T0)^-2.5*(0.01275*exp(-2239.1/T)/(frO+f^2/frO) + 0.1068*exp(-3352/T)/(frN+f^2/frN))]",
                    "frO": "(pa/pr)*(24 + 4.04e4*h*(0.02+h)/(0.391+h))",
                    "frN": "(pa/pr)*(T/T0)^-0.5*(9 + 280*h*exp(-4.170*((T/T0)^(-1/3) - 1)))",
                    "h": "molar concentration of water vapour in %: h = RH*(psat/pr)/(pa/pr), psat/pr = 10^(-6.8346*(273.16/T)^1.261 + 4.6151)",
                    "constants": {"pr": 101325.0, "T0": 293.15},
                },
                "humidity_profile": {"surface_RH_percent": 70.0, "h2o_scale_height_m": 2500.0, "tag": "ROM",
                                     "note": "constant-RH surface, water-vapour molar fraction decays exp(-z/2500 m)"},
                "table_standard_conditions": {"T_K": 293.15, "p_Pa": 101325.0, "RH_percent": 70.0,
                                              "octave_hz": fa, "alpha_dB_per_km": aa},
                "crackle_band_policy": {"evaluate_at_hz": 250.0, "exponent": 0.2, "nonlinear_range_m": 20000.0, "formula": "A_crackle(f) = A_iso(f) - (A_iso(f) - A_iso(f_ref*(f/f_ref)^exponent)) * min(1, nonlinear_range_m / r)   for f > f_ref", "tag": "ROM",
                                        "note": "weak-shock theory: while the wave stays nonlinear, the shock's HF content is regenerated by "
                                                "steepening, so the crackle layer is absorbed as if each band f > f_ref were at f_ref*(f/f_ref)^exponent "
                                                "(jet_mix keeps full ISO 9613-1 per band). "
                                                "Beyond nonlinear_range_m (old-age region) normal absorption resumes for the remaining path fraction."},
            },
            "ground_reflection": {
                "formula": "G = +6 dB for f < fc, +3 dB for f > 4*fc (log-interp between); fc = c/(4*dPath), dPath = 2*h_L*sin(psi)",
                "tag": "ROM", "h_listener_m": 1.7,
                "note": "coherent pressure doubling at LF over hard ground, incoherent (+3 dB) at HF; avoids an audible comb filter. "
                        "For listeners > 30 m above ground (aerial cameras) use G = +3 dB at LF only below 40 Hz."},
            "terrain_shielding": {"tag": "MEASURED", "note": "Mathews et al. 2021 report terrain shielding effects; not modelled (flat ground)."},
        },
        # ---------------------------------------------------------------- crackle
        "crackle": {
            "tag": "ROM",
            "convective_mach": "Mc = 0.65 * U / a_amb",
            "weight": "w_c = smoothstep((Mc - 1.5)/2.0) * clamp(pa/101325, 0, 1)^0.5 ; layer amplitude = clamp(w_c * reference_weight * w_dir(theta) * g_nl(r), 0, max_weight)",
            "note": "Crackle (shock content) needs supersonic convection of the large-scale structures (Ffowcs Williams et al. 1975; "
                    "Petitjean et al. 2006) - always satisfied by rocket plumes, so w_c is driven mainly by ambient density "
                    "(plume shear-layer intensity). See layer_gain_rule.",
            "reference_weight": 0.5,
            "max_weight": 4.0,
            "nonlinear_growth": {"formula": "g_nl = 1 + max_gain * min(1, log2(1 + ln(max(r, x_t)/x_t)) / log2_saturation)",
                                 "max_gain": 4.0, "log2_saturation": 2.5, "tag": "ROM",
                                 "note": "Mathews et al. 2020 measured Falcon 9 SPL above the noise floor up to 10 kHz at 8.3 km and "
                                         "attribute it to nonlinear propagation; spherical-wave distortion grows ~ln(r/r0)."},
            "directivity": {"peak_deg": 65.0, "width_deg": 25.0, "formula": "w_dir = exp(-((theta - peak_deg)/width_deg)^2)",
                            "tag": "LITERATURE (qualitative) + ROM",
                            "note": "crackle is Mach-wave radiation, strongest around the peak-radiation angle and weak toward the plume "
                                    "axis (derivative skewness peaks near max directivity in Gee et al.'s rocket/jet measurements); "
                                    "peak taken from the Falcon 9 OASPL peak (65 deg)"},
            "layer_gain_rule": "crackle amplitude w = clamp(w_c * reference_weight * w_dir * g_nl, 0, max_weight) (LINEAR) x the jet-group EQ gain, plus the "
                               "crackle absorption policy (propagation.atmospheric_absorption.crackle_band_policy). The crackle asset is "
                               "normalised so that at amplitude 1 it carries the same >400 Hz energy as the linear jet. reference_weight 0.5 and the "
                               "absorption exponent 0.2 were tuned (tools/audio/validate.py) so the pad render gives |dSk| 0.7-3 at "
                               "ignition/liftoff (unclipped IMAP Blast-1: 1.0-2.8) and the 5 km render 1.5-10 (IMAP Blast-2 at ~6 km: 3.1-7.6). Play it sample-aligned with jet_mix when possible.",
            "propagation_note": "No nonlinear propagation SOLVER: distance growth of shock content is represented by g_nl and the "
                                "reduced crackle absorption (both ROM). Real waveform steepening over km is NOT simulated.",
            "targets": {"deriv_skewness_shock_formation": {"value": 5.0, "tag": "LITERATURE",
                                                           "source": "Reichman et al. (cited in Gee et al. 2018 JASA 144, EL40): dSk~5 <-> significant shocks, needs fs >~ 100 f_peak"},
                        "deriv_skewness_contains_crackle": {"value": 3.0, "tag": "LITERATURE", "source": "Downing & James 2019 criterion (dSk > 3)"},
                        "measured_references": ref_stats.get("deriv_skewness", {})},
        },
        # ---------------------------------------------------------------- pad interaction
        "pad_interaction": {
            "tag": "ROM",
            "weight": "w_pad = exp(-max(h_nozzle, 0) / h_e),  h_e = 10 * De  (~31 m for stage 1)",
            "level_db_re_jet": -3.0,
            "source_position": "flame-trench exit: pad + trench_dir * 60 m at ground level (trench_dir = horizontal, away from the camera "
                               "is fine if unknown)",
            "deflected_efficiency_note": "SP-8072 Fig. 4: deflected exhausts radiate LESS total power (0.05-0.2 %), so the pad layer adds "
                                         "low-mid roar + water/steam texture while w_pad also lowers the free-jet term: "
                                         "jet term x (1 - 0.5*w_pad)",
            "ignition_overpressure": {
                "tag": "MEASURED (shape) + ARTISTIC (amplitude)",
                "source": "Mathews et al. 2020: IOP waveform period ~0.8 s, spectral peaks ~3 and 16 Hz at 8.3 km (Falcon 9). "
                          "Casiano 2009 (NTRS 20090032104): IOP from rapid exhaust acceleration during engine start.",
                "level_db_re_full_thrust_jet": 0.0, "directivity": "strongest along the flame-trench exit (Gee et al. 2023 Artemis-I)"},
        },
        # ---------------------------------------------------------------- onboard / cabin
        "onboard": {
            "airborne_plume_reaches_vehicle": {"formula": "w_air = (1 - smoothstep((mach - 0.8)/0.4)) * sqrt(rho*a/(1.225*340.3))",
                                              "tag": "ROM",
                                              "note": "once the vehicle outruns its own exhaust noise (M > ~1) plume sound cannot propagate upstream to the "
                                                      "forward vehicle; liftoff reflection from the pad is the dominant external load"},
            "structure_borne": {"formula": "L_struct = 20*log10(thrust / thrust_ref_stage1) + struct_ref_db ; continues in vacuum",
                                "thrust_ref_stage1_N": 7.473e6, "tag": "ROM",
                                "spectrum_source": "shape of NASA GSFC-STD-7000 (GEVS) random-vibration qualification spectrum (20-2000 Hz), "
                                                   "plus damped structural modes - SHAPE ONLY, no level claim"},
            "digital_reference_levels_dbfs": {"tag": "ARTISTIC",
                                              "onboard": {"structure_at_stage1_full_thrust": -22.0, "aero_at_w_aero_1": -20.0, "stage_sep": -26.0},
                                              "cabin": {"structure_at_stage1_full_thrust": -16.0, "aero_at_w_aero_1": -22.0, "stage_sep": -24.0},
                                              "note": "RMS dBFS; structure scales 20*log10(thrust/thrust_ref), aero 20*log10(w_aero); cabin aero "
                                                      "also gets 0.5 x mass-law TL as spectral tilt"},
            "aero_buffet": {"formula": "w_aero = (qbar/30000) * (0.35 + 0.65*exp(-((mach-1)/0.25)^2))", "tag": "ROM",
                            "source": "same law Aether already uses for camera shake (frontend/src/engine/Effects.ts)"},
            "cabin_transmission_loss": {"formula": "TL(f) = max(0, 20*log10(m_s * f) - 47)  [field-incidence mass law]",
                                        "m_s_kg_m2": 20.0, "tag": "LITERATURE (mass law) + ROM (surface mass)"},
            "vacuum_cutoff": "external airborne gain = 0 when listener pa < 1 Pa (and continuously via the rho*c factor above that)",
        },
        # ---------------------------------------------------------------- levels -> digital
        "level_mapping": {
            "tag": "ARTISTIC",
            "note": "Real launch levels span >80 dB across viewpoints; a microphone engineer sets gain per position. Each listener preset "
                    "has a fixed mic gain chosen from its predicted maximum SPL; levels then follow physics with an optional "
                    "loudness compression k (1 = physical, 0.6 recommended for games).",
            "dbfs": "RMS dBFS = target_peak_dbfs + k * (Lp - Lp_max_preset)  (target_peak_dbfs is the RMS level at the loudest moment; leaves ~15 dB crest headroom)",
            "target_peak_dbfs": -18.0, "k_default": 1.0, "k_game": 0.6,
            "safety_limiter": "soft limiter, threshold -1 dBFS, 5 ms attack, 200 ms release (never hard-clip)",
        },
        "anti_repetition": {
            "tag": "ARTISTIC (standard game-audio practice)",
            "voice_offset_fraction": 0.6180339887, "crossfade_cycle_s": 37.3,
            "formula": "out(t) = cos(phi)*loop(p) + sin(phi)*loop(p + 0.618*N),  phi = pi/4 + pi/4*sin(2*pi*t/37.3)",
            "note": "Each loop plays as two voices (golden-ratio offset) with an equal-power crossfade on an incommensurate "
                    "period, so the 24 s loop pattern does not audibly repeat. Use the SAME p and phi for rumble_lf, jet_mix and "
                    "crackle (sync group 'jet'). Different groups may use different start offsets (e.g. aero +7.3 s).",
        },
        "time_warp": {
            "tag": "ARTISTIC",
            "policy": [
                "warp <= 1 (incl. slow motion): control parameters and propagation delays in SIM time; layer playback stays at real-time "
                "rate (no pitch change); one-shots play at normal speed.",
                "1 < warp <= 4: parameters in sim time, delays shrink in wall-clock (consistent ordering); suppress one-shots that would "
                "overlap; airborne gain -6 dB*log2(warp)/2.",
                "warp > 4: fade airborne to -24 dB over 1 s and keep only rumble_lf + onboard_structure (time-lapse ambience).",
            ],
        },
        "listeners": {
            "_note": "positions relative to the pad, metres, local ENU (x east = downrange, y north, z up). Aether render-world uses "
                     "y-up: map (east, north, up) -> render (x, -z, y) as CameraRig does.",
            "aether": {
                "Pad": {"mode": "ground", "pos_enu": [310.0, -260.0, 1.7], "Lp_max_db": 150, "notes": "405 m: near field, pad layer + crackle dominate first 8 s"},
                "Engines": {"mode": "chase", "notes": "rides with the vehicle 26 m off the engines: no delay change, no Doppler; airborne "
                                                      "plume level from source model with r = 30 m, theta = 100 deg; fades with w_air like onboard"},
                "Tracker": {"mode": "ground", "pos_enu": [-1800.0, -2600.0, 2.0], "Lp_max_db": 135, "notes": "3.2 km long-lens: full delay + absorption"},
                "Onboard": {"mode": "onboard", "notes": "structure_borne + w_air plume + aero_buffet; no external sound in vacuum"},
                "Aerial": {"mode": "ground", "pos_enu": [1400.0, 1700.0, 650.0], "Lp_max_db": 135, "notes": "helicopter 650 m up: ground G per note"},
                "Free": {"mode": "chase", "notes": "treat as chase at the user distance; vacuum rule applies above ~80 km"},
                "Orbital": {"mode": "vacuum", "notes": "silent externally; optional onboard_structure at -20 dB if the camera is attached"},
            },
            "misanthropic": {
                "outside_observer": {"mode": "ground", "notes": "any distance: full model; 5 km and 20 km presets validated in docs"},
                "cabin_passenger": {"mode": "cabin", "notes": "onboard + cabin_transmission_loss on the airborne part; structure_borne dominant"},
                "chase_camera": {"mode": "chase", "notes": "as Aether 'Engines' with r = camera distance; cap Lp at preset max"},
                "vacuum": {"mode": "vacuum", "notes": "external = 0; cabin passenger still hears structure_borne while engines burn"},
            },
        },
        "layers": assets.get("layers", []),
        "oneshots": assets.get("oneshots", []),
        "asset_calibration": assets.get("calibration", {}),
        "references_summary": ref_stats,
        "vehicle_reference": {"stage1": {"count": e1["count"], "ae_m2": e1["ae_m2"], "De_cluster_m": round(de_cluster(e1), 3)},
                              "stage2": {"count": e2["count"], "ae_m2": e2["ae_m2"], "De_cluster_m": round(de_cluster(e2), 3)}},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=1)
    print("wrote", OUT)
    return prof


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
