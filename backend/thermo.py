"""Rocket-propellant performance from chemical equilibrium (Cantera).

Standard textbook performance analysis (as in NASA CEA "rocket" problems):
  * chamber: adiabatic, isobaric equilibrium (HP) of the reactants at Pc
  * nozzle: isentropic expansion with either SHIFTING equilibrium (re-equilibrate at each
    pressure, 'SP') or FROZEN composition (chamber composition held fixed)
  * throat located at maximum mass flux; area ratio / exit state from mass-flux ratio
Outputs: Tc, gamma, molecular weight, c*, exit velocity, Isp(vac/SL), pe, Te, Me.

Reactant enthalpies (NASA CEA thermo.inp, Gordon & McBride):
  RP-1  (CH1.9532, liquid, 298.15 K): hf = -24717.7 J/mol  (MW 13.9668)
  O2(L) (90.17 K):                    h  = -12979.0 J/mol  (MW 31.9988)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cantera as ct
import numpy as np
from scipy.optimize import brentq, minimize_scalar

G0 = 9.80665
RP1_H_PER_MOL = -24717.7
RP1_MW = 12.011 + 1.9532 * 1.00794
RP1_H_RATIO = 1.9532
LOX_H_PER_MOL = -12979.0
LOX_MW = 31.9988

_GAS = None


def gas() -> ct.Solution:
    global _GAS
    if _GAS is None:
        species = [s for s in ct.Species.list_from_file("nasa_gas.yaml") if set(s.composition) <= {"C", "H", "O"}]
        _GAS = ct.Solution(thermo="ideal-gas", species=species)
    return _GAS


def reactant_state(of_ratio: float):
    """Elemental composition (mole basis) and specific enthalpy [J/kg] of RP-1 + LOX at O/F (mass)."""
    m_f, m_o = 1.0, of_ratio  # kg
    n_f = m_f / (RP1_MW / 1000.0)  # mol of CH1.9532
    n_o2 = m_o / (LOX_MW / 1000.0)
    h = (n_f * RP1_H_PER_MOL + n_o2 * LOX_H_PER_MOL) / (m_f + m_o)
    # element-equivalent starting composition of stable species (C as CO, H as H2, rest of O as O2);
    # equilibrium does not depend on this choice, only on element totals, H and P.
    n_c, n_h, n_o = n_f, n_f * RP1_H_RATIO, 2.0 * n_o2
    comp = {"CO": n_c, "H2": n_h / 2.0, "O2": max(n_o - n_c, 0.0) / 2.0}
    return comp, h


@dataclass
class Perf:
    pc_bar: float
    of_ratio: float
    eps: float
    mode: str
    Tc_K: float
    gamma_c: float          # frozen cp/cv at chamber
    gamma_c_eq: float       # equilibrium isentropic exponent at chamber
    mw_c: float
    cstar_m_s: float
    Te_K: float
    pe_bar: float
    Me: float
    gamma_e: float
    ve_m_s: float
    Isp_vac_s: float
    Isp_sl_s: float
    cf_vac: float
    mw_e: float
    x_c: dict


def _chamber(pc: float, of: float):
    g = gas()
    comp, h = reactant_state(of)
    g.TPX = 3000.0, pc, comp
    g.HP = h, pc
    g.equilibrate("HP")
    return g


def _equilibrium_gamma(g: ct.Solution) -> float:
    """Isentropic exponent (dlnP/dlnrho)_s with re-equilibration (finite difference)."""
    s0, p0, r0 = g.entropy_mass, g.P, g.density
    X = g.X.copy()
    T = g.T
    g.SP = s0, p0 * 1.001
    g.equilibrate("SP")
    r1 = g.density
    g.TPX = T, p0, X
    return math.log(1.001) / math.log(r1 / r0)


def performance(pc_pa: float, of: float, eps: float | None = None, pe_target: float | None = None,
                mode: str = "shifting", p_amb_sl: float = 101325.0) -> Perf:
    g = _chamber(pc_pa, of)
    Tc, hc, sc, Xc = g.T, g.enthalpy_mass, g.entropy_mass, g.X.copy()
    gamma_c = g.cp_mass / g.cv_mass
    mw_c = g.mean_molecular_weight
    gamma_eq = _equilibrium_gamma(g)
    g.TPX = Tc, pc_pa, Xc

    def state_at(p):
        if mode == "shifting":
            g.SP = sc, p
            g.equilibrate("SP")
        else:
            g.TPX = Tc, pc_pa, Xc
            g.SP = sc, p
        v = math.sqrt(max(2.0 * (hc - g.enthalpy_mass), 0.0))
        return g.density * v, v, g.T, g.cp_mass / g.cv_mass, g.mean_molecular_weight, g.density

    # throat: maximise mass flux over pressure ratio
    res = minimize_scalar(lambda r: -state_at(pc_pa * r)[0], bounds=(0.4, 0.75), method="bounded", options={"xatol": 1e-7})
    pt = pc_pa * res.x
    G_t = state_at(pt)[0]
    cstar = pc_pa / G_t
    # exit pressure from area ratio (supersonic branch) or vice versa
    if eps is None:
        pe = pe_target
        G_e = state_at(pe)[0]
        eps = G_t / G_e
    else:
        f = lambda lnp: G_t / state_at(math.exp(lnp))[0] - eps
        pe = math.exp(brentq(f, math.log(pt * 0.999), math.log(pc_pa * 1e-7), xtol=1e-10))
    G_e, ve, Te, gamma_e, mw_e, rho_e = state_at(pe)
    a_e = math.sqrt(gamma_e * ct.gas_constant / mw_e * Te)
    Me = ve / a_e
    ae_over_mdot = eps / G_t
    isp_vac = (ve + pe * ae_over_mdot) / G0
    isp_sl = (ve + (pe - p_amb_sl) * ae_over_mdot) / G0
    cf_vac = isp_vac * G0 / cstar
    return Perf(pc_pa / 1e5, of, eps, mode, Tc, gamma_c, gamma_eq, mw_c, cstar, Te, pe / 1e5, Me, gamma_e, ve,
                isp_vac, isp_sl, cf_vac, mw_e,
                {k: float(v) for k, v in zip(g.species_names, Xc) if v > 1e-3})


def optimum_sl(pc_pa: float, of: float, mode="shifting") -> Perf:
    """Expansion to 1 atm (optimum sea-level expansion)."""
    return performance(pc_pa, of, pe_target=101325.0, mode=mode)


# Published reference: Sutton & Biblarz, Rocket Propulsion Elements, Table 5-5
# (LOX/RP-1, Pc = 1000 psia, optimum expansion to 14.7 psia, shifting equilibrium,
#  mixture ratio 2.56): Tc = 3676 K, c* = 1799 m/s, MW = 23.3, k = 1.24, Isp = 300 s.
SUTTON_REF = dict(pc_pa=1000 * 6894.757, of=2.56, Tc=3676.0, cstar=1799.0, mw=23.3, gamma=1.24, isp_sl=300.0)


def reference_check():
    r = SUTTON_REF
    p = optimum_sl(r["pc_pa"], r["of"], "shifting")
    rows = [
        ("Chamber temperature [K]", p.Tc_K, r["Tc"], 1.5),
        ("Characteristic velocity c* [m/s]", p.cstar_m_s, r["cstar"], 1.5),
        ("Chamber molecular weight [kg/kmol]", p.mw_c, r["mw"], 3.0),
        ("Isp, opt. SL expansion [s]", p.Isp_sl_s, r["isp_sl"], 2.0),
        ("gamma (chamber, frozen cp/cv; ref. k is a nozzle-average value)", p.gamma_c, r["gamma"], 4.0),
    ]
    out = []
    for name, c, ref, tol in rows:
        out.append(dict(quantity=name, computed=float(c), reference=float(ref), tol_pct=tol,
                        err_pct=float(100 * (c - ref) / ref), pass_=bool(abs(c - ref) / ref * 100 <= tol),
                        source="Sutton & Biblarz Table 5-5 (CEA-derived), LOX/RP-1 Pc=1000 psia, O/F=2.56, shifting"))
    return p, out


if __name__ == "__main__":
    p, rows = reference_check()
    for r in rows:
        print(f"{r['quantity']:40s} {r['computed']:9.2f} vs {r['reference']:9.2f}  ({r['err_pct']:+.2f}%, tol {r['tol_pct']}%)  {'PASS' if r['pass_'] else 'FAIL'}")
    for mode in ("shifting", "frozen"):
        q = performance(9.7e6, 2.36, eps=16.0, mode=mode)
        print(mode, f"S1 eps16: Tc={q.Tc_K:.0f} c*={q.cstar_m_s:.0f} Isp_vac={q.Isp_vac_s:.1f} Isp_sl={q.Isp_sl_s:.1f} pe={q.pe_bar:.3f} Me={q.Me:.2f} ge={q.gamma_e:.3f}")
        q = performance(9.7e6, 2.36, eps=165.0, mode=mode)
        print(mode, f"S2 eps165: Isp_vac={q.Isp_vac_s:.1f} pe={q.pe_bar:.4f} Me={q.Me:.2f} Te={q.Te_K:.0f}")
