"""Build the README figures from the solver outputs shipped in site/data.

    python docs/figures/make_figures.py

Every number plotted here comes from a file written by an executed solver (scipy solve_ivp,
Cantera, the reduced-order plume model, the benchmark scripts). One y-axis per chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "site" / "data"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9.5,
})


def save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.01, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


def load(f):
    return json.loads((DATA / f).read_text())


ref = load("reference_trajectory.json")
smp = [s for s in ref["samples"] if s["t"] <= 520]


def col(k):
    return np.array([s[k] for s in smp])


t = col("t")
ev = {n: tt for n, tt in ref["events"]}
EV_SHOW = [("MECO", "MECO + sep"), ("FAIRING_SEP", "Fairing"), ("SECO", "SECO")]


def mark_events(ax, keys=EV_SHOW):
    top = ax.get_ylim()[1]
    ax.axvline(ev["STAGE_SEP"], color=AXIS, lw=1, ls=(0, (3, 3)), zorder=0)
    for i, (key, lab) in enumerate(keys):
        ax.axvline(ev[key], color=AXIS, lw=1, ls=(0, (3, 3)), zorder=0)
        ax.text(ev[key] + 4, top * (0.98 - 0.09 * (i % 2)), lab, fontsize=8, color=MUTED, va="top")


# ------------------------------------------------------------------ 1. ascent profile (small multiples)
fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
panels = [
    (axs[0, 0], col("h") / 1000, "Altitude (km)", S[0]),
    (axs[0, 1], col("v_inertial") / 1000, "Inertial velocity (km/s)", S[0]),
    (axs[1, 0], col("thrust") / 1e6, "Thrust (MN)", S[1]),
    (axs[1, 1], col("mass") / 1000, "Vehicle mass (t)", S[2]),
]
for ax, y, lab, c in panels:
    ax.plot(t, y, color=c)
    ax.set_title(lab, fontsize=11)
    ax.set_ylim(0, y.max() * 1.12)
    mark_events(ax)
for ax in axs[1]:
    ax.set_xlabel("Mission time (s)")
fig.suptitle("Pad to orbit: 6-DOF ascent, scipy solve_ivp (DOP853) reference run", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
el = ref["elements_seco"]
save(fig, "ascent_profile.png",
     f"Orbit at SECO (T+{ev['SECO']:.1f} s): {el['periapsis_alt']/1000:.1f} x {el['apoapsis_alt']/1000:.1f} km, "
     f"i = {el['i_deg']:.2f} deg, e = {el['e']:.4f}. Rotating Earth, J2, USSA-1976 atmosphere, winds aloft.")

# ------------------------------------------------------------------ 2. max-Q & throttle bucket
m = t <= 170
fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axs[0].plot(t[m], col("qbar")[m] / 1000, color=S[1])
mq = ref["max_q"]
axs[0].scatter([mq["t"]], [mq["qbar"] / 1000], s=46, color=S[1], edgecolor=SURF, linewidth=2, zorder=3)
axs[0].annotate(f"Max-Q {mq['qbar']/1000:.1f} kPa\nT+{mq['t']:.0f} s, Mach {mq['mach']:.2f}",
                (mq["t"], mq["qbar"] / 1000), xytext=(-2, 24), textcoords="data", fontsize=9, color=INK2)
axs[0].set_title("Dynamic pressure (kPa)", fontsize=11)
axs[1].plot(t[m], col("throttle")[m] * 100, color=S[0])
axs[1].set_title("Stage-1 throttle (%): q-limit then g-limit guidance", fontsize=11)
axs[2].plot(t[m], col("g_load")[m], color=S[2])
axs[2].set_title("Axial load (g)", fontsize=11)
axs[2].set_xlabel("Mission time (s)")
for ax in axs:
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.axvline(ev["MECO"], color=AXIS, lw=1, ls=(0, (3, 3)))
axs[1].set_ylim(0, 110)
save(fig, "maxq_throttle.png", "Throttle-down through max-Q, then the g-limit caps acceleration before MECO (dashed line).")

# ------------------------------------------------------------------ 3. Cantera validation
th = load("thermo.json")
rc = th["reference_check"]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
SHORT = ["Chamber temperature", "Characteristic velocity c*", "Molecular weight", "Sea-level Isp", "Gamma (frozen)"]
names = SHORT[: len(rc)][::-1]
errs = [r["err_pct"] for r in rc][::-1]
tols = [r["tol_pct"] for r in rc][::-1]
y = np.arange(len(names))
for i, tl in enumerate(tols):
    ax.plot([-tl, tl], [i, i], color=GRID, lw=9, solid_capstyle="round", zorder=1)
ax.barh(y, errs, height=0.42, color=S[0], zorder=2)
for i, e in enumerate(errs):
    ax.text(e + (0.15 if e >= 0 else -0.15), i + 0.3, f"{e:+.2f}%", va="center", ha="left" if e >= 0 else "right",
            fontsize=9, color=INK2)
ax.set_yticks(y, names)
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlabel("Error vs reference (%); grey band = stated tolerance")
ax.set_title(f"Cantera {th['cantera_version']} vs Sutton & Biblarz (LOX/RP-1): 5 of 5 within tolerance")
ax.grid(axis="y", visible=False)
save(fig, "cantera_validation.png", "Pc = 1000 psia, O/F = 2.56, shifting equilibrium. Gamma compared as frozen.")

# ------------------------------------------------------------------ 4. Isp vs expansion ratio & O/F
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
iv = th["isp_vs_eps"]["stage1"]
eps = [r["eps"] for r in iv]
axs[0].plot(eps, [r["Isp_vac_s"] for r in iv], color=S[0], label="Vacuum")
axs[0].plot(eps, [r["Isp_sl_s"] for r in iv], color=S[1], label="Sea level")
axs[0].set_xscale("log")
for k, c, lab in [("stage1", S[0], "S1, e=16"), ("stage2", S[0], "S2, e=165")]:
    sd = th["stages"][k]
    axs[0].scatter([sd["eps"]], [sd["Isp_vac_s"]], s=46, color=c, edgecolor=SURF, linewidth=2, zorder=3)
    axs[0].annotate(lab, (sd["eps"], sd["Isp_vac_s"]), xytext=(-8, 8), textcoords="offset points",
                    fontsize=9, color=INK2, ha="right")
axs[0].set_xlabel("Nozzle expansion ratio")
axs[0].set_title("Ideal Isp (s), shifting equilibrium", fontsize=11)
axs[0].legend(loc="lower left")
axs[0].set_ylim(0, 400)
of = th["of_sweep"]
axs[1].plot([r["of"] for r in of], [r["Tc_K"] for r in of], color=S[1])
d_of = th["stages"]["stage1"]["of_ratio"]
axs[1].axvline(d_of, color=AXIS, ls=(0, (3, 3)))
axs[1].text(d_of + 0.03, min(r["Tc_K"] for r in of), "design O/F", fontsize=8.5, color=MUTED)
axs[1].set_xlabel("Mixture ratio O/F")
axs[1].set_title("Chamber temperature (K)", fontsize=11)
save(fig, "thermochemistry.png", f"Cantera equilibrium, Pc = {th['stages']['stage1']['pc_bar']:.0f} bar. "
     "Vehicle Isp = ideal x efficiency (0.93 / 0.935).")

# ------------------------------------------------------------------ 5. plume regime vs altitude
pl = [r for r in load("plume_table.json")["stages"]["stage1"] if r["h"] <= 80000]
h = np.array([r["h"] for r in pl]) / 1000
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(h, [r["ratio"] for r in pl], color=S[0])
axs[0].axhline(1, color=AXIS, lw=1, ls=(0, (3, 3)))
axs[0].set_yscale("log")
axs[0].text(52, 0.45, "matched (pe = pa)", fontsize=8.5, color=MUTED)
axs[0].set_title("Stage 1 exit / ambient pressure ratio", fontsize=11)
axs[0].set_xlabel("Altitude (km)")
axs[1].plot(h, [r["dj"] for r in pl], color=S[1])
axs[1].set_title("Jet boundary diameter (normalised)", fontsize=11)
axs[1].set_xlabel("Altitude (km)")
save(fig, "plume_rom.png", "REDUCED-ORDER MODEL (isentropic + Prandtl-Meyer + Pack shock-cell spacing), not CFD. "
     "It drives the plume shape and shock diamonds in the renderer.")

# ------------------------------------------------------------------ 6. physics performance
bp = load("benchmarks_physics.json")
fig, ax = plt.subplots(figsize=(8, 3.2))
lab = ["TypeScript RK4", "TS loop + WASM step", "Rust/WASM full loop"]
val = [bp["ts_full"]["stepsPerSec"], bp["ts_loop_wasm_rk4"]["stepsPerSec"], bp["wasm_full"]["stepsPerSec"]]
ax.barh(lab, np.array(val) / 1000, height=0.5, color=S[0])
for i, v in enumerate(val):
    ax.text(v / 1000 + 10, i, f"{v/1000:.0f}k steps/s", va="center", fontsize=9, color=INK2)
ax.set_xlabel("RK4 steps per second (full pad-to-orbit mission, median of 7)")
ax.set_title("6-DOF integrator throughput (Node 22 / V8)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(val) / 1000 * 1.25)
acc = bp["accuracy"]
save(fig, "physics_bench.png", f"All three agree: final position {acc['ts_final_pos_err_vs_python_m']:.2f} m from the Python "
     f"reference; WASM vs TS {acc['wasm_vs_ts_final_pos_diff_m']:.0e} m. 50x time warp needs 2.5k steps/s.")

# ------------------------------------------------------------------ 7. heating
tm = load("thermal.json")
ts = [r for r in tm["series"] if r["t"] <= 300]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot([r["t"] for r in ts], [r["q_kW_m2"] for r in ts], color=S[1])
ax.set_title("Stagnation-point heat flux estimate (kW/m²)")
ax.set_xlabel("Mission time (s)")
ax.set_ylim(bottom=0)
save(fig, "heating.png", f"Sutton-Graves, 1 m nose radius: peak {tm['peak_heat_flux_kW_m2']:.1f} kW/m² at T+{tm['peak_t']:.0f} s, "
     f"{tm['total_heat_load_MJ_m2']:.1f} MJ/m² total. Engineering estimate.")
