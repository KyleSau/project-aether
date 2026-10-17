"""Generate data/reference_trajectory.json from the authoritative solve_ivp simulation.

Used by the TypeScript / WASM agreement tests and shown in the app's engineering mode.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from aether_physics.sim import run_mission, orbit_check  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "reference_trajectory.json")


def main():
    r = run_mission(t_orbit_coast=300.0, sample_dt=1.0)
    keep = ("t", "phase", "h", "mach", "qbar", "thrust", "mass", "throttle", "g_load", "v_inertial", "v_rel", "x_cg", "alpha")
    samples = []
    for s in r["samples"]:
        d = {k: float(s[k]) if k != "phase" else int(s[k]) for k in keep}
        d["y"] = [round(v, 6) for v in s["y"]]
        samples.append(d)
    out = dict(
        schema="aether-reference/1",
        generator="physics/gen_reference.py (scipy.integrate.solve_ivp DOP853, rtol=1e-9, exact event location)",
        events=[[n, float(t)] for n, t in r["events"]],
        max_q={k: float(v) for k, v in r["max_q"].items()},
        elements_seco={k: float(v) for k, v in r["elements_seco"].items()},
        orbit_achieved=orbit_check(r["elements_seco"]),
        samples=samples,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", os.path.abspath(OUT), len(samples), "samples")
    for n, t in r["events"]:
        print(f"  {n:12s} t={t:8.2f}")
    e = r["elements_seco"]
    print(f"  SECO orbit: {e['periapsis_alt']/1e3:.1f} x {e['apoapsis_alt']/1e3:.1f} km, i={e['i_deg']:.2f} deg, e={e['e']:.5f}")
    print(f"  max-Q {r['max_q']['qbar']:.0f} Pa at t={r['max_q']['t']:.1f}s, M={r['max_q']['mach']:.2f}")


if __name__ == "__main__":
    main()
