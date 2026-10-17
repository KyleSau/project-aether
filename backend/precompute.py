"""Offline precompute pipeline (run once; the browser only loads the resulting JSON files).

  1. Cantera equilibrium performance of LOX/RP-1 (+ reference check vs published values)
  2. Derive the vehicle engine parameters (ve, pe, gamma_exit) from (1) with stated efficiency factors
  3. Reduced-order plume table vs altitude for both stages
  4. Authoritative trajectory (solve_ivp) -> data/reference_trajectory.json
  5. Thermal estimate along that trajectory (Sutton-Graves stagnation heating; SciPy integration)

Usage:  python backend/precompute.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "physics"))
sys.path.insert(0, HERE)

import cantera as ct  # noqa: E402
import thermo  # noqa: E402
from aether_physics.atmosphere import atmosphere  # noqa: E402
from aether_physics import plume_rom  # noqa: E402

DATA = os.path.join(ROOT, "data")
G0 = 9.80665

# Stated assumptions: delivered/ideal Isp efficiency (c* efficiency x nozzle/divergence losses)
EFFICIENCY = {"stage1": 0.93, "stage2": 0.935}
DESIGN = {"stage1": dict(pc=9.7e6, of=2.36, eps=16.0), "stage2": dict(pc=9.7e6, of=2.36, eps=165.0)}


def step_thermo():
    t0 = time.time()
    ref_perf, rows = thermo.reference_check()
    stages = {}
    curves = {}
    for st, d in DESIGN.items():
        shf = thermo.performance(d["pc"], d["of"], eps=d["eps"], mode="shifting")
        frz = thermo.performance(d["pc"], d["of"], eps=d["eps"], mode="frozen")
        stages[st] = dict(
            pc_bar=shf.pc_bar, of_ratio=shf.of_ratio, eps=shf.eps, Tc_K=shf.Tc_K, gamma_c=shf.gamma_c,
            gamma_c_eq=shf.gamma_c_eq, mw_c=shf.mw_c, cstar_m_s=shf.cstar_m_s, Isp_vac_s=shf.Isp_vac_s,
            Isp_sl_s=shf.Isp_sl_s, Te_K=shf.Te_K, pe_bar=shf.pe_bar, Me=shf.Me, gamma_e=shf.gamma_e,
            ve_m_s=shf.ve_m_s, cf_vac=shf.cf_vac, mode="shifting equilibrium",
            frozen=dict(Isp_vac_s=frz.Isp_vac_s, Isp_sl_s=frz.Isp_sl_s, pe_bar=frz.pe_bar, Te_K=frz.Te_K, Me=frz.Me,
                        cstar_m_s=frz.cstar_m_s, gamma_e=frz.gamma_e),
            chamber_mole_fractions=shf.x_c,
        )
        # Isp vs expansion ratio (vacuum and sea level), shifting
        eps_list = [4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80, 120, 165, 200]
        rowsE = []
        for e in eps_list:
            p = thermo.performance(d["pc"], d["of"], eps=float(e), mode="shifting")
            rowsE.append(dict(eps=e, Isp_vac_s=p.Isp_vac_s, Isp_sl_s=p.Isp_sl_s, pe_bar=p.pe_bar))
        curves[st] = rowsE
    # O/F sweep at S1 design (shifting)
    of_sweep = []
    for of in np.arange(1.8, 3.21, 0.1):
        p = thermo.performance(9.7e6, float(of), eps=16.0, mode="shifting")
        of_sweep.append(dict(of=round(float(of), 2), Tc_K=p.Tc_K, cstar_m_s=p.cstar_m_s, Isp_vac_s=p.Isp_vac_s, mw_c=p.mw_c))
    out = dict(
        schema="aether-thermo/1",
        propellant="LOX / RP-1 (RP-1 as CH1.9532, hf=-24717.7 J/mol; O2(L) h=-12979 J/mol at 90.17 K)",
        method="Cantera chemical equilibrium (HP chamber; isentropic SP expansion, shifting and frozen); species: all C/H/O gases in nasa_gas.yaml",
        cantera_version=ct.__version__,
        stages=stages,
        isp_vs_eps=curves,
        of_sweep=of_sweep,
        reference_check=[{**{k: v for k, v in r.items() if k != "pass_"}, "pass": r["pass_"]} for r in rows],
        efficiency=EFFICIENCY,
        compute_s=time.time() - t0,
    )
    with open(os.path.join(DATA, "thermo.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def step_vehicle(th):
    p = os.path.join(DATA, "vehicle.json")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    for st in ("stage1", "stage2"):
        e = cfg["engines"][st]
        s = th["stages"][st]
        eta = EFFICIENCY[st]
        isp_vac = eta * s["Isp_vac_s"]
        pe = s["pe_bar"] * 1e5
        ve = isp_vac * G0 - pe * e["ae_m2"] / e["mdot0_kg_s"]
        e["ve_m_s"] = round(ve, 2)
        e["pe_pa"] = round(pe, 1)
        e["gamma_exit"] = round(s["gamma_e"], 4)
        e["chamber_pressure_pa"] = s["pc_bar"] * 1e5
        e["expansion_ratio"] = s["eps"]
        e["of_ratio"] = s["of_ratio"]
        e["derivation"] = (f"Cantera shifting-equilibrium ideal Isp_vac={s['Isp_vac_s']:.1f} s x efficiency {eta} "
                           f"= {isp_vac:.1f} s; pe from ideal expansion to eps={s['eps']:g}")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def step_plume(cfg):
    rows = {}
    for st in ("stage1", "stage2"):
        e = cfg["engines"][st]
        out = []
        for h in range(0, 120001, 1000):
            pa = atmosphere(float(h))[1]
            ps = plume_rom.plume_state(e["chamber_pressure_pa"], e["pe_pa"], e["ae_m2"], e["gamma_exit"], pa, 1.0, e.get("nozzle_lip_angle_deg", 10.0))
            out.append(dict(h=h, **ps))
        rows[st] = out
    nozzle = {st: plume_rom.nozzle_profile(cfg["engines"][st]["expansion_ratio"], cfg["engines"][st]["gamma_exit"]) for st in rows}
    with open(os.path.join(DATA, "plume_table.json"), "w", encoding="utf-8") as f:
        json.dump(dict(schema="aether-plume/1", label="REDUCED-ORDER MODEL", model=plume_rom.__doc__.strip(), stages=rows, nozzle=nozzle), f)
    return rows


def step_reference():
    import gen_reference  # noqa: E402  (physics/gen_reference.py)
    gen_reference.main()


def step_thermal():
    """Sutton-Graves stagnation-point convective heating on the fairing nose (r_n = 1.0 m, assumed)."""
    with open(os.path.join(DATA, "reference_trajectory.json"), encoding="utf-8") as f:
        ref = json.load(f)
    k = 1.7415e-4  # Earth air, SI (Sutton & Graves 1971)
    rn = 1.0
    t = np.array([s["t"] for s in ref["samples"]])
    h = np.array([s["h"] for s in ref["samples"]])
    v = np.array([s["v_rel"] for s in ref["samples"]])
    rho = np.array([atmosphere(float(x))[2] for x in h])
    q = k * np.sqrt(rho / rn) * v ** 3  # W/m^2
    Q = cumulative_trapezoid(q, t, initial=0.0)
    i = int(np.argmax(q))
    out = dict(schema="aether-thermal/1", model="Sutton-Graves stagnation heating, nose radius 1.0 m (estimate)",
               peak_heat_flux_kW_m2=float(q[i] / 1e3), peak_t=float(t[i]), peak_h_m=float(h[i]),
               total_heat_load_MJ_m2=float(Q[-1] / 1e6),
               series=[dict(t=float(a), q_kW_m2=float(b / 1e3)) for a, b in zip(t[::5], q[::5])])
    with open(os.path.join(DATA, "thermal.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "physics"))
    th = step_thermo()
    for r in th["reference_check"]:
        print(f"  ref check: {r['quantity']}: {r['computed']:.2f} vs {r['reference']:.2f} ({r['err_pct']:+.2f}%) {'PASS' if r['pass'] else 'FAIL'}")
    for st, s in th["stages"].items():
        print(f"  {st}: Tc={s['Tc_K']:.0f} K c*={s['cstar_m_s']:.0f} Isp_vac={s['Isp_vac_s']:.1f} (frozen {s['frozen']['Isp_vac_s']:.1f}) pe={s['pe_bar']:.4f} bar Me={s['Me']:.2f}")
    cfg = step_vehicle(th)
    for st in ("stage1", "stage2"):
        print("  vehicle", st, cfg["engines"][st]["derivation"], "ve=", cfg["engines"][st]["ve_m_s"])
    step_plume(cfg)
    step_reference()
    tm = step_thermal()
    print(f"  thermal: peak {tm['peak_heat_flux_kW_m2']:.1f} kW/m2 at t={tm['peak_t']:.0f}s h={tm['peak_h_m']/1e3:.1f} km; load {tm['total_heat_load_MJ_m2']:.2f} MJ/m2")
