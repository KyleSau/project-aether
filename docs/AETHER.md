# Project Aether

Project Aether is a browser-based orbital launch simulator. The vehicle's pose comes only from a 6-DOF ascent simulation; nothing in the scene is driven by an animation curve. The vehicle is a generic, unbranded two-stage LOX/RP-1 medium-lift launcher. All of its parameters are assumptions, chosen within ranges that have been published for this class of vehicle (`data/vehicle.json`).

## Layout

| Path | Contents |
|---|---|
| `physics/` | The authoritative Python reference: USSA-1976 atmosphere, 6-DOF dynamics, guidance and control, `solve_ivp` mission driver, orbital elements and the reduced-order plume model. Tests are in `physics/tests`. |
| `backend/` | `thermo.py` (Cantera equilibrium), `precompute.py` (offline pipeline that writes `data/*.json`), `app.py` (FastAPI) and `openfoam/` (case generator and adapter; currently NOT CONNECTED). |
| `wasm/` | A Rust crate (wasm-pack) that ports the integrator line for line. |
| `frontend/` | React 19, TypeScript, Vite 8 and three.js 0.185.1 (`WebGPURenderer`). The TS physics port lives in `src/physics`; rendering is in `src/engine`. |
| `site/` | Static build (Vite, relative base). It needs no backend. |
| `data/` | Versioned precomputed data: the vehicle, Cantera results, the plume table, the reference trajectory, thermal results and benchmarks. |

Run it with `.\start.ps1`. This builds the app if needed and then serves both the app and the API at http://localhost:8300.

URL flags:

- `?renderer=webgl` uses the WebGL2 backend.
- `?quality=low|medium|high|ultra` sets the quality tier.
- `?pr=2` sets the pixel ratio.
- `?physics=wasm` runs the RK4 step in Rust/WASM.
- `?particles=analytic` forces the non-compute particle system.
- `?mblur=1` turns on motion blur (experimental).

## Physics model

The state has 15 elements, all in an ECI frame: r, v, the quaternion, body rates, and two propellant masses.

- **Gravity** is inverse-square plus J2. Earth rotation can be switched on (the pad starts with velocity ω×r).
- **Atmosphere** is USSA-1976. From 0 to 86 km it uses the exact layer equations; from 86 to 1000 km it interpolates the published table log-linearly.
- **Aerodynamics** uses an axial coefficient CA(M) taken from a representative slender-body curve (an assumption, listed in `vehicle.json`). The normal force uses CNα = 2/rad and acts at the centre of pressure. The vehicle is statically unstable and is stabilised by thrust-vector control (TVC). Wind follows an assumed profile that peaks near 12 km.
- **Thrust** is `F = n(thr·ṁ0·ve + (thr·pe0 − pa)·Ae)`, so it varies with ambient pressure.
- **Throttle** is limited by dynamic pressure (q-limit 30 kPa) and by a 4.5 g acceleration limit.
- **Mass properties:** the CG and inertia come from cylinder components, and the propellant columns drain.
- **Rotational dynamics** use Euler's equations. The İω term and jet damping are neglected.
- **Guidance:** vertical rise, then a 6.5° pitch kick, then a blend into a gravity turn on the Earth-relative velocity. Stage 2 uses closed-loop altitude/vertical-speed feedback (ωn = 0.03 rad/s). SECO occurs when orbital energy reaches the energy of a circular orbit at 200 km.
- **Attitude:** a PD controller that commands angular acceleration. On engine power it is realised by gimbal (±5°); while coasting on stage 2 it uses ideal RCS.
- **Events:** MECO at propellant depletion, separation at +3 s, stage-2 ignition (SES) at +5 s, fairing jettison at 110 km, then SECO.

There are three implementations of the same model:

- **Python:** `solve_ivp` with DOP853 at rtol 1e-9 and exact event location.
- **TypeScript:** fixed-step RK4 at h = 0.02 s, with secant refinement for state events. This is what the app uses.
- **Rust/WASM:** an identical port of the TS integrator.

The spent stage and the fairing halves after separation are not simulated in 6-DOF. They use a reduced-fidelity 3-DOF ballistic model with drag, and their separation pushes are assumed values.

Nominal result (Python): MECO at T+151.8 s and 75 km; max-Q 31.3 kPa at T+51 s, Mach 1.07; SECO at T+460.1 s into a **196.6 × 203.4 km orbit, i = 28.64°, e = 0.0005**.

## Thermochemistry (Cantera 3.2.0)

The chamber is solved as HP equilibrium. The nozzle expansion is isentropic, computed both with shifting equilibrium and with frozen composition. The throat is placed at maximum mass flux. All C/H/O gas species from `nasa_gas.yaml` are included. RP-1 is modelled as CH1.9532 with hf = −24717.7 J/mol, and LOX with h = −12979 J/mol at 90 K (the CEA values).

The check against Sutton & Biblarz Table 5-5 (LOX/RP-1, Pc = 1000 psia, O/F = 2.56, shifting equilibrium, optimum sea-level expansion):

| Quantity | Cantera | Reference | Error | Tolerance |
|---|---|---|---|---|
| Tc | 3665.1 K | 3676 K | −0.30 % | 1.5 % |
| c* | 1797.0 m/s | 1799 m/s | −0.11 % | 1.5 % |
| MW | 23.34 | 23.3 | +0.18 % | 3 % |
| Isp (sea level) | 299.7 s | 300 s | −0.09 % | 2 % |
| γ (frozen chamber) | 1.211 | 1.24* | −2.3 % | 4 % |

\* The reference k is a single averaged value. The equilibrium isentropic exponent in the chamber is 1.138.

Design points: Pc = 97 bar, O/F = 2.36.

- **Stage 1** (ε = 16): ideal Isp_vac is 336.7 s with shifting equilibrium and 321.6 s frozen. pe = 0.691 bar, Me = 3.50.
- **Stage 2** (ε = 165): Isp_vac is 372.1 s shifting and 349.5 s frozen. pe = 0.034 bar.

The vehicle uses delivered values: the ideal shifting Isp multiplied by an assumed efficiency of 0.93 (stage 1) or 0.935 (stage 2), giving **313.1 s and 347.9 s**. `pe` and `γe` are taken directly from Cantera. `precompute.py` writes them into `vehicle.json`.

## What is solver data and what is approximation

| Element | Label |
|---|---|
| Trajectory, attitude, telemetry, orbital elements, max-Q, staging, engineering force vectors, CG | **SOLVER DATA** from the 6-DOF integrator (TS, validated against the Python solve_ivp run) |
| Exhaust Tc, γ, MW, c*, Isp, pe | **SOLVER DATA** from Cantera equilibrium, computed offline |
| Plume expansion angle, jet diameter, shock-cell spacing, diamond visibility, Mach disk | **REDUCED-ORDER MODEL**: isentropic relations, Prandtl–Meyer and Pack's formula applied to the simulated ambient pressure, validated against NACA 1135 values |
| Stagnation heating | **ESTIMATE** (Sutton–Graves, assuming a 1 m nose radius): peak 36.7 kW/m² at T+130 s, load 3.3 MJ/m² |
| Plume luminosity and colour; smoke, steam (sound-suppression water), clouds, the Earth surface, the vapour cone (assumed humidity profile × transonic Gaussian), camera shake (0.5 % acoustic efficiency, delayed by d/c), audio | **ARTISTIC APPROXIMATION** |
| CFD fields | **NOT CONNECTED**. No CFD has been run and none is displayed. |

## Validation (all passing at commit time)

- **pytest, `physics/tests`: 32 tests.**
  - The USSA-1976 table at 10 altitudes, including the speed of sound.
  - Radial free fall against the closed-form Kepler fall time (|Δr| < 5 cm).
  - A ballistic arc against a universal-variable Kepler propagation (< 5 cm).
  - A circular orbit over one period: it closes to within 1 m, and energy and angular momentum drift by less than 1e-10.
  - J2 nodal regression against first-order theory.
  - Thrust against ambient pressure.
  - The Tsiolkovsky equation against the integrated |F|/m over the stage-2 burn (to 1e-6).
  - Staging mass bookkeeping, CG and inertia trends.
  - Orbit insertion, max-Q and the g-limit.
  - Cantera against the reference values, shifting vs frozen, Isp vs ε, and engine parameters derived from Cantera.
  - The plume ROM against NACA 1135 area–Mach and Prandtl–Meyer values, and its regimes.
- **pytest, `backend/test_api.py`: 3 tests.**
- **vitest: 15 tests.**
  - The TS atmosphere against USSA.
  - The thrust model.
  - **TS RK4 against Python solve_ivp over 763 s of flight:** identical event sequence, event times within 0.05 s, maximum |Δr| = 0.92 m, maximum |Δv| = 0.010 m/s, orbit within 500 m.
  - **WASM against TS:** |Δr| = 4.7e-9 m at T+760 s.
  - The TS plume ROM against the Python table (relative difference < 1e-6).
- **cargo test: 1 test** (the mission reaches orbit natively).

## Benchmarks

**Physics** (`npm run bench`; Node 22/V8, i7-14700KF; full mission, ~38k RK4 steps, median of 7 runs):

| Engine | Steps/s | Time per mission |
|---|---|---|
| TS | 572k | 66.7 ms |
| WASM (whole loop in Rust) | 855k | 44.6 ms (1.49×) |
| TS loop with WASM RK4 step | 614k | 1.07× |

Both are about 0.84 m from Python at T+760 s, and they differ from each other by 4e-9 m.

At 50× time warp the simulation needs 2,500 steps per second, so TS already has about 229× headroom. **WASM is not the default**: it does not clearly win in practice, because the per-step boundary overhead eats most of its advantage. It ships as an option (`?physics=wasm`, verified in Chrome: same events and orbit). Particle emission math runs on the GPU, in compute shaders or vertex shaders, so it was not ported to WASM.

**Rendering:** Chrome 155 on an RTX 4070 Ti, viewport 1084×815, in-app `benchmark()` at T+3 s with full lift-off effects. Display vsync caps the rate at about 131 Hz.

| Configuration | FPS | p95 (ms) | CPU (ms/frame) |
|---|---|---|---|
| WebGPU, high, 48k compute particles | 131 (vsync) | 8.1 | 1.4 |
| WebGPU, ultra, 120k particles, 2168×1630 | 124 | 11.8 | — |
| WebGL2, medium | 131 | 8.2 | — |
| WebGL2, high, 2168×1630 | 117 | 14.8 | — |
| Low tier (phone settings) | 131 | — | — |

JS heap ranged from 67 to 213 MB. Estimated GPU memory is 21–213 MB. Full data is in `data/benchmarks_render.json`. No phone was tested.

## OpenFOAM: NOT CONNECTED

`backend/openfoam/case_generator.py` writes a complete case:

- an axisymmetric 5° wedge;
- `rhoCentralFoam` with Kurganov fluxes;
- inflow at the nozzle exit set from the Cantera exit state;
- co-flow at the trajectory's air-relative speed;
- ambient conditions from USSA;
- `waveTransmissive` far field;
- about 86k cells;
- an axis sampling function object;
- an `Allrun` script.

An example case is in `backend/openfoam/cases/s1_h10km`. `adapter.py` detects an installation read-only; on this machine it reports NOT CONNECTED, because there is no `blockMesh` and the only WSL distro is `docker-desktop`, with the Docker engine stopped. It also parses results, but only from a completed run that carries an `AETHER_RUN_COMPLETE` marker.

What is needed, with the user's permission:

- **Option A:** WSL2 Ubuntu with OpenFOAM (`openfoam11` or ESI `openfoam2406`).
- **Option B:** a running Docker engine plus an OpenFOAM image such as `opencfd/openfoam-run:2406`.

The case has never been run, so the blockMesh vertex ordering is unverified.

Limitations of the case: it models a single perfect gas (the exhaust γ) everywhere, it is inviscid, it has no afterburning chemistry, and the nozzle interior is not meshed.

## Known limitations and next steps

- Rotational dynamics neglect İω and jet damping. Gimbal actuation is algebraic, with no actuator dynamics. There is no roll control.
- Stage-2 guidance is planar (no yaw steering), so inclination follows from the launch azimuth.
- The atmosphere above 86 km is interpolated from the table rather than computed with the full 1976 upper-atmosphere model.
- Plume luminosity is not radiometric. There is no soft-particle depth fade. Particle sorting is unsorted alpha blending.
- Motion blur is off by default: the velocity MRT smeared transparent sprites.
- The sun direction is fixed in the pad frame.
- The spent stage and fairings use the 3-DOF ballistic model described above.
- Not done yet: Monte Carlo dispersions, a powered-explicit-guidance (PEG) style upper-stage guidance, running the OpenFOAM case, a phone performance check, and HDR environment capture from the sky model.
