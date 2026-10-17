"""Cantera equilibrium performance vs published reference values (Sutton & Biblarz Table 5-5)."""
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))
import thermo  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return thermo.reference_check()


def test_reference_values_within_tolerance(ref):
    _, rows = ref
    for r in rows:
        assert r["pass_"], f"{r['quantity']}: {r['computed']:.2f} vs {r['reference']:.2f}"


def test_shifting_exceeds_frozen():
    s = thermo.performance(9.7e6, 2.36, eps=16.0, mode="shifting")
    f = thermo.performance(9.7e6, 2.36, eps=16.0, mode="frozen")
    assert s.Isp_vac_s > f.Isp_vac_s
    assert s.Tc_K == pytest.approx(f.Tc_K)
    # shifting equilibrium recombination keeps the exit hotter
    assert s.Te_K > f.Te_K


def test_isp_increases_with_expansion_ratio_in_vacuum():
    a = thermo.performance(9.7e6, 2.36, eps=10.0)
    b = thermo.performance(9.7e6, 2.36, eps=40.0)
    c = thermo.performance(9.7e6, 2.36, eps=165.0)
    assert a.Isp_vac_s < b.Isp_vac_s < c.Isp_vac_s
    assert a.pe_bar > b.pe_bar > c.pe_bar


def test_vehicle_engine_parameters_derived_from_cantera():
    cfg = json.load(open(os.path.join(ROOT, "data", "vehicle.json"), encoding="utf-8"))
    th = json.load(open(os.path.join(ROOT, "data", "thermo.json"), encoding="utf-8"))
    for st in ("stage1", "stage2"):
        e = cfg["engines"][st]
        isp_vac = (e["ve_m_s"] + e["pe_pa"] * e["ae_m2"] / e["mdot0_kg_s"]) / 9.80665
        assert isp_vac == pytest.approx(th["efficiency"][st] * th["stages"][st]["Isp_vac_s"], rel=1e-4)
        assert e["pe_pa"] == pytest.approx(th["stages"][st]["pe_bar"] * 1e5, rel=1e-4)
