# Launch acoustics: the shared audio core

This document describes the physically grounded launch-audio core shared by the Aether browser simulator and the
Misanthropic Unreal game (which runs a C++ port of the same physics). The core covers the research, the measurements,
the offline synthesis, the physics→audio contract and its validation. The browser front-end is described in section 11; the Unreal
front-end is built separately from the outputs listed below.

Every number carries one of four labels, in Aether's usual spirit:

| Tag | Meaning |
|---|---|
| **MEASURED** | A published measurement (cited), or our own computation on a downloaded public-domain recording |
| **LITERATURE** | An empirical or standard model (NASA SP-8072, ISO 9613-1, mass law, ...) |
| **ROM** | A reduced-order model chosen here. Physically motivated, but not validated |
| **ARTISTIC** | A listening or design choice with no physical claim |

We claim **no absolute SPL accuracy**. None of the recordings we could obtain are calibrated. The only calibrated
numbers are the published ones quoted below.

## 1. Outputs (the contract)

| Path | What it is |
|---|---|
| `data/audio/acoustic_profile.json` | Machine-readable profile: telemetry contract, source model, spectrum, directivity, propagation, crackle, pad, onboard, level mapping, time-warp and anti-repetition policies, listener presets, layer list with calibration |
| `data/audio/assets/*.wav` | Source textures: 48 kHz, mono, 16-bit. Loops are 24 s and seamless by construction |
| `data/audio/assets/*.ogg` | Ogg/Opus copies for the web (96 kbit/s loops, 112 kbit/s one-shots; 2.5 MB in total) |
| `data/audio/assets/manifest.json` | Per-asset band levels (1/3-octave and octave), RMS, peak, seeds and sizes (merged into the profile) |
| `data/audio/reference_analysis.json` | Every metric computed on every real-recording segment, plus a summary |
| `data/audio/validation/validation_metrics.json` | The same metrics on our renders, plus the "before" proxy |
| `data/audio/validation/comparison_pairs.json` | Render-vs-real pairs used in the figures |
| `data/audio/validation/telemetry_reference.csv` | Reference-sim telemetry in the renderer's CSV format: a ground truth for runtime checks |
| `docs/audio/launch_*.ogg` | Validation renders to listen to: pad 405 m, 5 km, 20 km, onboard, cabin |
| `data/audio/validation/browser_runtime_comparison.json` | Browser runtime vs Python renders (section 11.5) |
| `tools/audio/` | Analysis (`dsp.py`, `references.py`), model (`acoustic_model.py`, `make_profile.py`), synthesis (`synth_layers.py`), reference renderer (`render_launch.py`), validation (`validate.py`), tests, `build_all.py` |

**Regenerate everything** with `python tools/audio/build_all.py`. Use `--offline` to reuse the committed reference
analysis instead of re-downloading the NASA recordings. You need numpy, scipy, matplotlib, pytest and ffmpeg built
with libopus. A full run takes about 5 minutes after the first download, and every random draw uses a fixed seed.

## 2. Telemetry the runtimes must supply

The renderer needs these fields at the control rate (30 Hz or faster). Because sound is evaluated at the **emission
time**, keep at least 20 s of history (more for distant listeners: the delay reaches 60 s at 20 km).

| Quantity | Unit | TypeScript (`physics/dynamics.ts` `Telemetry`) | Unreal (`Aether::Telemetry`) | Python (`tel` dict) |
|---|---|---|---|---|
| time, phase | s, enum | `t`, `phase` | `T`, `Phase` | `t`, `phase` |
| active-stage thrust | N | `thrust` | `Thrust` | `thrust` |
| active-stage mass flow | kg/s | `mdot` | `Mdot` | `mdot` |
| throttle | 0..1 | `throttle` | `Throttle` | `throttle` |
| ambient p, T, ρ, a at the vehicle | Pa, K, kg/m³, m/s | `P`, `T`, `rho`, `a` | `P`, `AtmT`, `Rho`, `A` | `P`, `T`, `rho`, `a` |
| altitude, Mach, q̄ | m, -, Pa | `h`, `mach`, `qbar` | `H`, `Mach`, `QBar` | `h`, `mach`, `qbar` |
| nozzle-exit position | m | Effects `base` (render world, float64) | from `GetAscentState` R and Q | ECI `y[0:3]`, then pad-local ENU |
| plume axis (exhaust direction) | unit vector | `-axis` | `-(body x)` from Q | `-R[:,0]` |
| engine count, `ae_m2`, `chamber_pressure_pa`, `pe_pa`, `gamma_exit` | | `cfg.engines.stageN` | `Engine::Count`, `Ae`, `Pe`; **Pc and γ are not in the C++ `Engine` struct** (read them from vehicle.json or hard-code them) | `vehicle.json` |
| events | | `MissionSim.events` (`IGNITION`, `LIFTOFF`, `MECO`, `STAGE_SEP`, `SES`, `FAIRING_SEP`, `SECO`) | same names | `run_mission()['events']` |
| listener position, its ambient pressure, its mode | | camera | camera/pawn | CLI |

The effective exhaust velocity is `U = thrust / mdot`, the same in all three runtimes. At sea level it is 2786 m/s,
within 3 % of the configured 2877 m/s; the difference is the over-expanded pressure thrust.

## 3. Sources researched and analysed

### 3.1 Literature (numbers quoted, never invented)

| Source | Type | What we use |
|---|---|---|
| Eldred, **NASA SP-8072** (1971), *Acoustic Loads Generated by the Propulsion System*, NTRS 19710023719. Downloaded; text extracted; figures 5 and 10 digitised by eye from the scan | LITERATURE | Acoustic efficiency 0.2-1.0 % ("0.5 % most probable, 1 % conservative upper bound"); deflected exhausts radiate less (Fig. 4); normalised power spectrum vs Strouhal `f·de/Ue`, peaking at St ≈ 0.02 (Fig. 5); cluster rule `De = √n·de` (eq. 3); directivity vs Strouhal (Fig. 10); SPL = Lw − 10 log r² − 11 + DI (eq. 5); core-length plot (Fig. 11) and source distribution at 1-2 core lengths (Fig. 12) |
| Mathews, Gee, Hart et al. 2020, *Comparative analysis of noise from three Falcon 9 launches*, J. Acoust. Soc. Korea 39(4) 322, doi:10.7776/ASK.2020.39.4.322 (the jask.or.kr link) | MEASURED (Vandenberg SLC-4E, 8.3 km) | Max OASPL 112-114 dB near T+40 s; about 106 dB at T+10-11 s; peak directivity about 65° (6-dB-down 45-80°); peak frequency about 20 Hz; SPL above the noise floor up to 10 kHz, attributed to nonlinear propagation; IOP waveform period about 0.8 s with spectral peaks at about 3 and 16 Hz; late-launch bandwidth about 50 Hz; about 15 dB drop at MECO; 0.5 Hz high-pass used |
| Mathews, Gee & Hart 2021, *Characterization of Falcon 9 launch vehicle noise from far-field measurements*, JASA 150, 620, doi:10.1121/10.0005658 (PubMed 34340507; abstract read via Europe PMC) | MEASURED | Qualitative: directivity between the historical ranges and explained by convective Mach; radiation efficiency "consistent with the literature"; two OASPL-max predictors within 2 dB at several km; terrain shielding matters. Full text not accessed, so no numbers are quoted from it |
| Kellison & Gee 2023, JASA-EL, doi:10.1121/10.0022538 | MEASURED | SLS OAPWL 202.4 dB re 1 pW, acoustic efficiency about 0.33 % |
| Gee et al. 2023, JASA-EL, doi:10.1121/10.0016878 (Artemis I) | MEASURED | 127-136 dB maximum OASPL at 1.5-5.2 km; mean maximum 1/3-octave peak 20 Hz; IOP strongest toward the flame-trench exit |
| Gee et al. 2018, JASA 144 EL40 (crackle percept); Swift, Gee & Neilsen 2017; Reichman et al. (as cited); Downing & James 2019 (as cited) | LITERATURE | Crackle correlates log-linearly with the skewness of the pressure derivative (dSk); dSk ≈ 5 marks significant shocks when fs ≳ 100 × f_peak; dSk > 3 is used as a "contains crackle" criterion. Pressure skewness alone is not sufficient |
| Casiano 2009, NTRS 20090032104 (IOP / lift-off acoustics prediction software) | LITERATURE (qualitative) | IOP is caused by rapid exhaust acceleration during engine start. The NTRS page has no numbers |
| NASA NAS SC16 demo 17 (LAVA/ASMAT ignition-overpressure CFD+DG) | LITERATURE (qualitative) | Sources: plume shear layers and plume-pad impingement; reflection and refraction on pad structures |
| ISO 9613-1:1993 | LITERATURE | Atmospheric absorption. Our implementation matches the standard's table at 20 °C, 70 % RH within 1.3 % (unit-tested) |
| NASA GSFC-STD-7000 (GEVS) random-vibration qualification spectrum | LITERATURE (shape only) | Shape of the structure-borne layer |

### 3.2 Recordings actually downloaded and computed on

All of these are NASA-produced media. NASA media are generally not subject to US copyright (NASA Images and Media Usage
Guidelines); NASA endorsement must not be implied. The Falcon 9 feeds were recorded by NASA/KSC staff (asset credit
"NASA/Lorne Mathre"). **Nothing from them is shipped**: they were used for statistics only, and option 4h ("use
processed NASA textures") was not needed. Only the launch windows were transferred, with ffmpeg seeking inside the
remote CBR MP3 files.

| ID | Launch / vehicle | Item | Type | Usable segments | Artefacts flagged |
|---|---|---|---|---|---|
| `imap` ch0 "Blast 1" | IMAP, Falcon 9, LC-39A, 2025-09-24 | images.nasa.gov `KSC-20250924-...BLAST_1_BLAST_2-M16776` | broadcast technical feed, unmixed blast mics, MP3 192k/48k | startup ramp; receding; decay | clipping 86-96 s (up to 21 % of samples); fader/routing steps at 118 s and 202 s; LF roll-off below about 12 Hz |
| `imap` ch1 "Blast 2" | same | same | same | arrival, peak, ascent decay, tail | crosstalk from Blast 1 during 86-97 s; LF below about 30 Hz depressed. **Distance inferred** at about 5.5-7.5 km from the about 20 s arrival lag (not documented) |
| `crew12` ch0 "Beach" | Crew-12, Falcon 9, SLC-40, 2026-02-13 | `KSC-20260213-...Beach_Causeway-M19013` | broadcast feed | ascent decay A/B | onset is a 40 dB gate step in 0.1 s (not acoustic); 136-157 s fully clipped; periodic 0.5-1.5 kHz bursts every ~30 s (surf or PA), excluded |
| `crew12blast` ch1 "Blast" | same | `...Tech_Feed_Blast-M19013` | broadcast feed | liftoff (clipped) | gated/switched channel; unrelated programme after 155 s |
| `atlas_sdo` | SDO, Atlas V 401 (single RD-180 LOX/RP-1, no SRBs) | nasa.gov `590329main_ringtone_SDO_launchNats.mp3` | edited 20 s clip, MP3 128k | whole clip | fades, 16 kHz MP3 low-pass, nothing below 20 Hz, distance unknown |
| `sts131` | STS-131, Shuttle | `590189main_ringtone_131_launchNats.mp3` | edited clip | whole | probably limited (6 dB crest factor); SRB-dominated, so secondary |
| `sls_test` | SLS-programme static test | `663784main_SLS_Audio_D.mp3` | clip | whole | static, not a launch |

We looked at but did not use the NASA Wallops "Sounds of a Launch" mix (a narrated podcast mix), STS-26 (commentary)
and the Crew-11 feed (two Falcon 9 feeds were enough).

### 3.3 Analysis pipeline (`tools/audio/dsp.py`, applied identically to references and renders)

For each segment we compute:

- waveform; STFT spectrogram (4096-point Hann); Welch PSD; 1/3-octave band levels
- RMS envelope (125 ms); crest factor; kurtosis and >4σ exceedance rate
- skewness of p and of dp/dt (crackle)
- crackle events: peaks of dp/dt > 5σ, their rate, and the **coefficient of variation of the inter-event intervals** (≈1 for Poisson, →0 for a "machine-gun" train)
- LF/mid/HF energy fractions; spectral centroid; least-squares PSD slope over 250-4000 Hz (robust to the unknown LF response of broadcast mics)
- envelope (modulation) spectrum, as a modulation index over 0.5-4 Hz and 4-20 Hz
- clipping fraction and longest run; MP3 HF cut-off; LF roll-off

Per-reference panels are in `docs/figures/audio/ref_*.png`; `ref_comparison.png` overlays the clean spectra and dSk.

**Measured on the recordings** (uncalibrated MP3 broadcast audio; dBFS is meaningless across files):

| Segment | PSD slope 250-4000 Hz | dSk | Crackle CV | Modulation 0.5-4 / 4-20 Hz (250-2000 Hz band) |
|---|---|---|---|---|
| IMAP Blast-1 startup ramp (unclipped) | −4.0 dB/oct | −2.8 (polarity?) | 0.34 | − |
| IMAP Blast-1 receding | −6.9 | 1.0 | 3.9 | 0.23 / 0.24 |
| IMAP Blast-2 peak (~6 km) | −6.5 | **7.6** | 2.0 | 0.17 / 0.27 |
| IMAP Blast-2 ascent decay | −7.0 | 3.1 | 2.3 | 0.15 / 0.42 |
| Crew-12 Beach ascent decay | −17 / −12 | ≈0 | 2.6-2.9 | 0.12 / 0.18 |
| Atlas V SDO (distant) | −10.6 | **7.7** | 1.9 | 0.14 / 0.28 |
| STS-131 | −8.8 | 4.7 | 1.1 | − |

Other measured facts:

- **Engine-start rise at a near-pad mic:** 48.5 dB in 1.02 s (IMAP Blast-1, before clipping). It is preceded by an isolated igniter/valve "tick" about 2 s earlier, about 40 dB below full level.
- **Envelope at about 6 km** (Blast-2): roughly a 20 s arrival lag, a peak about 40 s after the Blast-1 onset, then a roughly 25 dB decay over 80 s. This matches Mathews' "maximum near T+40 s".
- **Spectra:** every broadcast mic loses the band below 20-40 Hz, so the measured LF shape is **not** usable. LF comes from SP-8072 and the published 20 Hz peak instead.
- **Crackle is bursty:** interval CV of about 2 is clustered, more irregular than Poisson. That is the target that keeps our crackle from sounding like a machine gun.

## 4. Source model (stage 1 at sea level: U = 2786 m/s, ṁ = 2682 kg/s, De = 3.096 m)

- Mechanical power: `P = ½ṁU²` = 1.04×10¹⁰ W. Acoustic power: `W = η P`, η = 0.5 % → **Lw = 197.2 dB re 1 pW** [LITERATURE]. With the SLS-measured η of 0.33 % it would be 1.8 dB lower.
- Spectrum: SP-8072 Fig. 5, digitised and renormalised to unit area (a −0.29 dB correction), with **f_peak = 0.02·U/De = 18.0 Hz** [LITERATURE]. Measured Falcon 9 peaks are about 20 Hz.
- Spectral shift with exhaust velocity, nozzle size and ambient pressure: `f_peak ∝ U/D_spec` [ROM].
  - `D_spec = max(De, √n·Dj)`, where Dj is the fully expanded jet diameter from `physics/plume_rom.py`. The plume balloons at altitude, so the spectrum moves down (5.8 Hz at 24 km, 1.7 Hz at 45 km).
  - Throttle enters through ṁ and U.
- Apparent source position: `x_src = 1.5·x_t` down the plume, with core length `x_t = 3.45·De·(1+0.38Me)²` (about 60 m). This is a commonly used fit to SP-8072 Fig. 11; the report itself gives only the plot. The source is clipped at the ground, and whatever lies below it becomes the pad term.
- Directivity [LITERATURE + MEASURED]: the SP-8072 Fig. 10 curves are angle-warped so that the OASPL peak they imply (SP-8072: 40°, 6 dB points 17-78°) maps to the Falcon 9 measured **65° (45-80°)**. Angles below 20° are an extrapolation into the cone of relative silence. The table is in the profile (`directivity.di_db`, 0-180° every 10°, five Strouhal curves).
- Source cut-off: `w_src` ramps from 0 to 1 between 0.1 and 10 Pa ambient [ARTISTIC]. Nothing is cut below MECO altitude, because Mathews measured the MECO level drop on the ground.

## 5. Propagation

| Effect | Model | Tag |
|---|---|---|
| Spreading | `Lp = Lw_b + DI − 10log(4πr_eff²) + 10log(ρ_L c_L/413.3) − A + G`. Near-field limit `r_eff = √(r² + x_t²)` | LITERATURE / ROM |
| Delay | `t_r = t_e + r(t_e)/c_path`. `c_path` is the harmonic mean of USSA-1976 sound speed along the straight ray. Wind and refraction are ignored | LITERATURE |
| Doppler | **Only** from reading the source at its emission time (a variable delay line; playback rate `dτ/dt_r`). **No extra pitch shift.** Unit-tested: the read rate equals c/(c+v) | LITERATURE |
| Absorption | ISO 9613-1 per band, integrated along the ray with local T and p and a humidity profile (70 % RH at the surface, water-vapour scale height 2.5 km). Standard table in the profile: 1 kHz 4.98, 2 kHz 9.04, 4 kHz 23.1, 8 kHz 77.6 dB/km | LITERATURE (humidity profile: ROM) |
| Ground | +6 dB below `fc = c/(4·2h_L·sinψ)`, +3 dB above 4fc. This avoids a comb filter. For elevated listeners: +3 dB below 40 Hz | ROM |
| Vacuum | `ρc` at the listener → 0; external gain is 0 when the listener's ambient pressure is below 1 Pa | LITERATURE |

## 6. Crackle (the hardest part, and the least certain)

The waveform's **energetic part (< 2 kHz) is steepened** with a circular Earnshaw/Burgers (Pestorius-style)
advection, with a thermo-viscous low-pass applied at every step. The shock train's HF content (above 400 Hz, rise-time
roll-off at 5 kHz, softly limited at 1.4σ) forms the `crackle` layer. The shocks sit on the jet's own large
compressions, because they come from the same random realisation. The layer adds burst intermittency (0.2-3 Hz,
σ = 0.3): its event-interval CV is about 1.4-2, against a measured value of about 2.

At run time [ROM]:

- Source term: `w_c = smoothstep((Mc−1.5)/2)·√(pa/p0)`, with `Mc = 0.65U/a`.
- Mach-wave directivity: `w_dir = exp(−((θ−65°)/25°)²)`.
- Distance growth (nonlinear steepening): `g_nl = 1 + 4·min(1, log₂(1+ln(r/x_t))/2.5)`.
- Layer amplitude: `w = clamp(w_c · 0.5 · w_dir · g_nl, 0, 4)`, a linear factor on the jet EQ gain.
- Absorption: the crackle layer is absorbed as if each band f were at `250·(f/250)^0.2` Hz, over the first 20 km of the path (weak-shock HF regeneration).

The constants were tuned against the measured dSk and HF levels (section 8).

## 7. Layers, mappings and runtime rules

| Layer | Loop | Band | Role | Gain law |
|---|---|---|---|---|
| `rumble_lf` | yes | 3-160 Hz | LF exhaust rumble with slow modulation | jet-group EQ (below) |
| `jet_mix` | yes | 40 Hz-20 kHz | Turbulent mixing noise, the main body. Has 3-20 Hz roughness driven by its own LF waveform | jet-group EQ |
| `crackle` | yes | >400 Hz | Shock HF, bursty | `w` × jet EQ + crackle absorption |
| `pad_interaction` | yes | 12 Hz-10 kHz | Deflector/trench roar plus water/steam texture | pad EQ × `w_pad = exp(−h/(10De))`; jet term × (1 − 0.5 w_pad) |
| `onboard_structure` | yes | 6 Hz-1.2 kHz | Structure-borne: GEVS shape → panel velocity, with modes | `20log(F/F_ref)` + reference level |
| `aero_buffet` | yes | 4 Hz-2 kHz | Boundary layer and transonic buffet | `20log(w_aero)` + reference level |
| `ignition` | one-shot | | Igniter tick, nine staggered engine light-ups (crackly swells, not a bang), IOP pulse (0.8 s Gaussian-derivative plus 16 Hz component) | Jet EQ at liftoff − 2 dB, placed at IGNITION + pad delay |
| `meco_tail` | one-shot | | Thrust tail-off with a falling spectrum, then plume-collapse rumble | Jet EQ just before MECO, delayed |
| `stage_sep_structural` | one-shot | | Muffled structural clunk and ring-down (onboard/cabin only) | Reference level |

**EQ rule** (all groups): `EQ_dB[b] = D[b] − A_group[b]`.

- `A_group` is the asset band level from the profile, power-summed over the group's layers.
- `D` is the digital target: `D[b] = Lp[b] − Lp_OA + (−18 dBFS) + k·(Lp_OA − Lp_max_preset)`. Use k = 1 for physical levels, 0.6 for games.
- Runtimes may use 11 octave bands; the reference renderer uses 1/3-octave bands from 6.3 Hz to 16 kHz with STFT gains.

**Anti-repetition** (ARTISTIC): each loop plays as two voices, offset by 0.618 of its length, with an equal-power
crossfade `φ = π/4 + π/4·sin(2πt/37.3)`. Unit-tested: the 24 s envelope correlation falls from 0.99 to below 0.5.
Layers in sync group `jet` share position and φ.

**Time warp:**

- warp ≤ 1: parameters and delays run in sim time; playback stays at real-time rate.
- warp up to 4×: delays shrink with sim time; overlapping one-shots are suppressed; −3 dB per doubling.
- warp > 4×: keep rumble and structure only, at −24 dB.

**Cameras and listeners.** Presets in the profile (`listeners`):

| Listener | Preset |
|---|---|
| Aether Pad | ground, 405 m, Lp_max 143 dB |
| Aether Engines | chase |
| Aether Tracker | ground, 3.2 km |
| Aether Onboard | onboard |
| Aether Aerial | elevated ground |
| Aether Free | chase, vacuum rule applies |
| Aether Orbital | vacuum |
| Misanthropic outside observer | ground |
| Misanthropic cabin passenger | cabin: mass-law TL with m_s = 20 kg/m², structure dominant |
| Misanthropic chase camera | chase |
| Misanthropic vacuum | vacuum |

**Onboard**: external plume sound reaches the hull only while M ≲ 1 (`w_air = (1 − smoothstep((M−0.8)/0.4))·√(ρa/ρ0a0)`).
Structure-borne sound continues in vacuum while the engines burn; coasting is silent.

## 8. Validation (figures in `docs/figures/audio/`)

Renders, produced by `render_launch.py` from the Python reference trajectory:

| Listener | Opus file (listen) | Full-quality WAV (local, git-ignored) | Predicted max OASPL |
|---|---|---|---|
| Pad camera, 405 m | `docs/audio/launch_pad.ogg` | `tools/audio/_cache/renders/launch_pad.wav` | 143 dB |
| 5 km NW | `docs/audio/launch_obs5km.ogg` | `.../launch_obs5km.wav` | 121 dB |
| 20 km NW | `docs/audio/launch_obs20km.ogg` | `.../launch_obs20km.wav` | 108 dB |
| Onboard (hull) | `docs/audio/launch_onboard.ogg` | `.../launch_onboard.wav` | 146 dB external |
| Cabin | `docs/audio/launch_cabin.ogg` | `.../launch_cabin.wav` | 129 dB after TL |

**Where it agrees**

- **OASPL vs the only calibrated Falcon 9 data** (`val_oaspl.png`): the model gives about 110 dB at T+10 s and a maximum of about 116.5 dB at 8.3 km. Mathews measured about 106 dB and 112-114 dB. That is **2.5-4 dB high**, which is consistent with η = 0.5 % against the about 0.33 % now measured for SLS. The model's maximum arrives about 12 s earlier than measured, because our trajectory and observer geometry are generic. The throttle-down near max-Q shows up as a kink.
- **Envelope at about 5 km** (`val_envelopes.png`): rise, directivity peak and an 80 s decay overlay IMAP Blast-2 within a few dB.
- **Spectra above 250 Hz** (`val_third_octave.png`):
  - Near pad: slope −7.3 vs −6.9 dB/oct.
  - About 5 km: −8.7 vs −6.5; 2 kHz and 4 kHz within about 3-5 dB of Blast-2.
  - 20 km: smooth, between the Atlas V and Crew-12 beach shapes.
- **Crackle** (`val_metrics.png`): pad at full thrust, dSk 1.1 vs 1.0 (Blast-1, unclipped); 5 km early ascent, 8.3 vs 7.6 (Blast-2 peak). Event-interval CV is 1.1-2.7 (real: 1.7-3.9), never periodic.
- **Modulation:** in the 250-2000 Hz band, the assets give 0.21 (0.5-4 Hz) and 0.33 (4-20 Hz), against 0.12-0.23 and 0.18-0.42 measured.

**Where it differs** (honestly)

- **Below 40 Hz** our renders are much stronger than every recording. That is expected: the broadcast mics roll off there, and the physics (SP-8072, Artemis I and Falcon 9 peaks at about 20 Hz) says the energy is there. Laptop and phone speakers cannot reproduce it either: below about 80 Hz they rely on the harmonics of the 40-160 Hz content.
- **dSk is far higher than any reference in two render windows**: pad during early ascent (22) and 5 km mid-ascent (19). In both, ISO absorption has removed the linear jet HF and the remaining HF is pure shock content. No unclipped reference covers those geometries. dSk is also extremely sensitive to content above 4 kHz.
- **20 km has no distinct crackle** (dSk 0.5); the Atlas V clip has 7.7, but its distance is unknown and probably much closer. Mathews reports content to 10 kHz at 8.3 km but gives no level, so we cannot calibrate the distant crackle.
- **Slow modulation (0.5-4 Hz) in renders** is 0.2-0.3 against 0.1-0.2 measured. Part of this is real level change from the trajectory inside the analysis windows.
- **Not modelled:** refraction and wind, terrain shielding (Mathews 2021 show it matters), plume-ground reflection geometry beyond the ground-gain rule, and trench orientation (a fixed 60 m trench exit is assumed).

**Before/after.** `old_audio_ts_proxy` re-implements the previous `frontend/src/engine/Audio.ts`: a 4-s looped
leaky-integrated noise through a 220-420 Hz low-pass, plus 4-s looped sparse clicks band-passed at 900 Hz.

| | Before (proxy) | After |
|---|---|---|
| Loop period | 4 s (audible) | 24 s with anti-repetition |
| dSk | 0.0 (no shock structure; clicks are symmetric) | as in the table above |
| Crackle interval CV | 0.97 (Poisson-like clicks) | 1.1-2.7, clustered like real recordings |
| Spectrum | LP at 420 Hz, nothing physical | follows Strouhal/directivity/absorption |
| Delay, Doppler, directivity, pad term, IOP, onboard vacuum behaviour | none | all modelled |

The Unreal game's layered thunder/burner/waterfall stand-ins (Tools/prep_rocket_audio.py) were judged "horrible" by
the user. They are replaced by these assets plus the profile.

**Tests** (`python -m pytest tools/audio/tests -q`, 16 tests):

- ISO 9613-1 table values
- SP-8072 spectrum has unit area
- f_peak and Lw in the expected ranges
- directivity peak at 55-75°
- assets match the manifest and never clip
- **loop seams**: wrap step < 5σ of the interior steps, and seam spectral flux ≤ the 99.5th percentile
- no internal short repeats
- **propagation delay** (model and rendered onset)
- **Doppler** read rate = c/(c+v)
- **event sync** (ignition arrival)
- **distance-dependent filtering** (HF/LF and OASPL fall monotonically with distance)
- vacuum silence
- anti-repetition

## 9. Performance notes for the runtimes

- Per tick: one source state, one ray (16-24 atmosphere samples for `c_path` and absorption), about 11 octave gains per group. Cache absorption and `c_path` at 4-10 Hz; they change slowly.
- Audio graph: 4 loop groups × 2 anti-repetition voices, plus one-shots. EQ is either 11 peaking/shelf biquads per group or one FFT convolver updated at 10-20 Hz. WebAudio: `BiquadFilterNode` chains with `setTargetAtTime`. Unreal: MetaSounds or a Submix EQ chain.
- Emission time: keep a ring buffer of (t, nozzle position, axis, source state). Solve `t_e` by 2-3 fixed-point iterations from the previous value. Feed `dτ/dt_r` to the playback rate of all voices in a sync group (clamped to 0.5-1.5).
- Decoded memory: 6 loops × 24 s × 48 kHz × 4 bytes ≈ 28 MB of float32. Halve it by decoding at 24 kHz for `rumble_lf`, `onboard_structure` and `aero_buffet`.

## 10. Limitations and uncertainties (summary)

- Absolute levels are η-dependent (±2 dB from the 0.33-0.5 % spread) and uncalibrated at the speaker.
- Digitising the SP-8072 figures by eye adds about ±1.5 dB in the spectrum and about ±1 dB / ±3° in directivity.
- Crackle is a reduced-order texture plus heuristics (section 6), not a nonlinear propagation solver.
- The pad term (level, weight height, trench position), the onboard levels and the cabin surface mass are ROM or artistic.
- The reference recordings are broadcast MP3s with unknown mic positions, unknown frequency responses, clipping and gating. Everything measured on them is relative.
- Falcon 9-specific numbers are applied to Aether's generic AETHER-M2 vehicle. They match only to the extent that the vehicle is Falcon 9-class (9 × LOX/RP-1, 7.5 MN).

## 11. Browser runtime (`frontend/src/audio/`)

The browser engine implements this model live from the simulator's telemetry. It replaces the old
`frontend/src/engine/Audio.ts` (a filtered-noise stand-in), which is deleted.

### 11.1 Architecture

| File | Role |
|---|---|
| `profile.ts` | Typed view of `acoustic_profile.json`; octave-band tables (`A_jet` = rumble_lf ⊕ jet_mix, `A_pad`, pad shape, asset overall levels, directivity and spectrum tables) |
| `acoustics.ts` | Port of `acoustic_model.py` in octave bands: source state (W, U, D_spec via the TS plume ROM, x_t, crackle source weight), SP-8072 band power, directivity, ISO 9613-1 along a 24-point ray with the humidity profile, `c_path`, ground gain, ρc at the listener (−∞ below 1 Pa), and the pad, onboard, aero and cabin-TL laws |
| `telemetryRing.ts` | `TelemetryRing`: 8192 frames in column storage (819 s of the sim's 0.1 s history; MECO reaches the Pad camera 441 s late). `solveEmissionTime`: Newton form of the fixed point `t_e = t − r(t_e)/c` (3 steps from the previous t_e, bisection fallback). The plain fixed point diverges once the radial speed exceeds c (about T+60 s at 5 km). `c_path` is iterated to self-consistency, so a seek gives the same t_e as continuous play |
| `simFeed.ts` | Telemetry adapter: `MissionSim.history` frames (deterministic; also what replay plays) → `Telemetry` via `evaluate()` → nozzle exit (stage 2: body x = `geometry.stage2.x0` = 45 m) and exhaust axis in pad-local ENU, the frame of `render_launch.py` |
| `listeners.ts` | Camera → listener preset, taken from `listeners.aether` |
| `mixModel.ts` | `AcousticMixer` (pure, no Web Audio). For each control tick it gives the per-group octave EQ targets `EQ_dB = D − A_group`, the read rate and read position (emission time) per sync group, the onboard levels, the warp policy and the one-shot plans. Also holds `OneShotScheduler` |
| `eqDesign.ts` | Realises an octave target with Web Audio nodes: a gain, 10 high-shelf biquads (S = 1) at the octave mid-points, and a 4th-order Butterworth low-pass (two biquads). Solved with the exact Web Audio biquad formulas: a linear solve on the dB sensitivity matrix, then 5 fixed-Jacobian Newton passes |
| `AudioEngine.ts` | The Web Audio graph on any `BaseAudioContext` (live or offline) |
| `assets.ts` | Opus first, WAV fallback, trim to the exact length, half-rate storage of rumble_lf and onboard_structure |
| `LaunchAudio.ts` | Browser driver: unlock on the first gesture, a 30 Hz `setInterval` tick (outside the three.js loop), suspend on pause or hidden tab, discontinuity handling (seek, restart), volume and mute kept in `localStorage` |

Graph (one EQ chain per group; two anti-repetition voices per layer):

```
rumble_lf A/B ─ xfade ─ layer gain ┐
jet_mix   A/B ─ xfade ─ layer gain ┴─ jet EQ ─────┐
crackle   A/B ─ xfade ─ layer gain ── crackle EQ ─┼─ airborne (warp gain) ─┐
pad_int.  A/B ─ xfade ─ layer gain ── pad EQ ─────┘                        │
onboard_structure A/B ─ xfade ─ level ─────────────────────────────────────┼─ mix ─ ¼ ─ WaveShaper (soft limit) ─ master ─ out
aero_buffet       A/B ─ xfade ─ level ─────────────────────────────────────┤
one-shots ─ own static EQ chain ─ airborne (ignition, MECO tail) or mix (structural)
```

The graph holds 50 nodes when idle (3 EQ chains of 13 nodes, the buses and 6 layer gains) and 81 at liftoff
(8 loop voices plus the ignition one-shot). A layer creates its voices only while it is audible and releases
them 1.2 s after its output falls below −110 dBFS. One-shot chains are disconnected on `ended`. After a
restart the count returns to the idle 50 (checked in Chrome).

### 11.2 Mappings as implemented

- **Telemetry**: `thrust`, `mdot`, `throttle`, `P`, `T`, `rho`, `a`, `h`, `mach`, `qbar` and `phase` come from
  `evaluate()` on each history frame. Events come from `MissionSim.events`, keeping only those with `t ≤` the
  displayed time, so replay never plays the future. Listener positions: Pad, Tracker and Aerial use the profile's
  `pos_enu`. Free maps the camera's render position to ENU as (x, −z, y − GROUND_Y).
- **Delay and Doppler**: the jet group is read at `t_e`. Its `playbackRate` is `1/(1 + (dr/dt_e)/c)`, the
  analytic read rate of the delay line, clamped to [0.5, 1.5] as the profile says. Nothing else changes pitch.
  The pad layer is a static source at the trench exit (0, 60, 1) m, with its own constant delay and rate 1.
- **Loop positions**: every voice starts at `(τ − τ0) mod 24 s` (the second voice adds 0.618·24 s), as in
  `read_loop_ar`. Here τ is the group's emission time and τ0 the first history sample; the cross-fade phase φ
  uses the same τ. A replay or a seek therefore reproduces the same textures. On a discontinuity the voices
  restart at the new positions with a 20 ms cross-fade.
- **Levels**: `D = Lp − Lp_OA − 18 + k(Lp_OA − Lp_max)` with `k = k_default = 1`. Pad, Tracker and Aerial take
  `Lp_max` from their presets (150, 135 and 135 dB). Onboard and chase have no preset, so they use the model at a
  synthetic full-thrust liftoff state: 145.69 dB onboard (the Python render gives 145.71 dB), and r = 30 m,
  θ = 100° for Engines. Chase (Engines, Free) is capped at that maximum.
- **Groups off**: a group whose loudest output band is below −110 dBFS is reported off, and its voices stop.
- **EQ conditioning**: an output band more than 50 dB below the loudest output band of its group is treated as
  "don't care". Its target may be raised to keep the fall at 24 dB/oct or less, and it is down-weighted in the
  fit. The low-pass corner is a continuous function of the target: the point where the target's local slope
  first passes −18 dB/oct. The filter structure therefore never switches between ticks. An early version did
  switch, and that made 1500σ clicks.
- **Parameter smoothing**: group gains use `setTargetAtTime` with τ = 50 ms; cross-fade gains and playback
  rate use τ = 100 ms. Biquad gains and frequencies are stepped with `setValueAtTime` once per tick, because
  automated biquad parameters make Chrome recompute coefficients for every sample. On a camera switch the mix
  dips (12 ms), the new listener's EQ, voices and one-shots are set underneath, and the mix returns 70 ms later.
- **One-shots**: planned per (event, listener):
  - Ignition at `IGNITION + r_pad/c` (ground) with the liftoff jet EQ − 2 dB.
  - MECO tail at `MECO + r(t_MECO − 0.1 s)/c` with the jet EQ just before MECO.
  - Onboard: the external tail × w_air (dropped when inaudible) plus the structural tail (ref − 6 dB − 0.3·TL).
  - The stage-separation clunk plays onboard only.

  They are scheduled in audio time at `now + (t_arrive − t_sim)/warp` with a 0.12 s look-ahead. A one-shot seen
  one tick late (≤ 0.15 s) starts from its beginning. One joined later (after a seek or a camera switch) starts
  part-way, with a 6 ms fade-in.
- **Time warp** (`warpPolicy`):
  - Up to 1×: normal. Delays run in sim time and playback stays at real-time rate.
  - 2×: airborne −3 dB, and one-shots do not overlap.
  - Above 4× (5, 10, 50): airborne reaches −24 dB in about 1 s. Only rumble_lf and onboard_structure play,
    there are no one-shots, and the read rate is 1 (no Doppler at time-lapse rates).
- **Pause or hidden tab**: the master fades out in 12 ms, then `AudioContext.suspend()` runs; resuming
  reverses it. Restart stops and disconnects every voice and one-shot and clears the history and the plans.
- **Cameras**:
  - Pad, Tracker and Aerial are ground listeners: full delay, Doppler, absorption and ground gain (Aerial uses
    the elevated rule).
  - Engines is a chase listener: r = 30 m, θ = 100°, no delay, fading with w_air.
  - Free is a chase listener at the actual geometry.
  - Onboard hears structure, aero and the w_air plume; the external sound is killed in vacuum.
  - Orbital is a vacuum listener and is silent.
- **Not implemented in the browser**: the cabin aero spectral tilt. Aether has no cabin camera; the cabin mode
  exists in the model for the jet and pad transmission loss only.

### 11.3 Browser realities

- The `AudioContext` is created on the first `pointerdown`, `keydown` or `touchend` (the LAUNCH click counts),
  at 48 kHz with `latencyHint: 'playback'`, falling back to the default rate. The profile JSON is fetched at
  boot and the assets after unlock.
- Opus is used when `canPlayType('audio/ogg; codecs="opus"')` is non-empty and `decodeAudioData` succeeds.
  Otherwise that asset falls back to its WAV (Safari < 17). Decoded buffers are trimmed to the manifest length,
  rescaled to the context rate. Network: 1.9 MB with Opus, 14.2 MB with the WAV fallback.
- Memory: 24.1 MB of decoded float32 PCM plus a 1.2 MB ring buffer. rumble_lf and onboard_structure are stored
  at 24 kHz. aero_buffet is not halved, because it carries real energy at 8-16 kHz.
- Static hosting: `npm run sync-data` copies `data/audio/acoustic_profile.json` and `data/audio/assets/*` to
  `public/data/audio/`. The build uses relative URLs (`base: './'`). This was checked under a `/site/` sub-path:
  every asset returned 200 and there were no console errors. `post-site.mjs` accepts `.ogg` and `.wav` and
  budgets the WAV fallback separately (≤ 20 MB).
- The bottom bar has a master volume slider and a Sound/mute button.

### 11.4 Performance

| | Value |
|---|---|
| Control tick (model + 3 EQ designs + node updates), desktop Chrome | 0.24-0.36 ms at 30 Hz (about 1 % of a core) |
| Same, offline run (node) | mean 0.08 ms onboard and 0.29 ms ground; max 2.7 ms |
| EQ design | about 10 µs for a flat target, about 120 µs with a low-pass corner |
| Audio graph | 3 EQ chains × 12 biquads, up to 12 loop voices; the offline render runs at 43-87× real time |

### 11.5 Validation against the Python reference

`npm run audio:offline` renders the real graph (`AudioEngine` + `AcousticMixer`) on an `OfflineAudioContext`.
It uses node-web-audio-api (an implementation of the Web Audio spec), drives the graph from
`data/audio/validation/telemetry_reference.csv` with 30 Hz ticks, and applies the same mic gain as the Python
render. `python tools/audio/compare_browser.py` then compares the result with
`tools/audio/_cache/renders/launch_*.wav` using `dsp.py`. Results are in
`data/audio/validation/browser_runtime_comparison.json`:

| Render | Envelope diff (1 s, >20 Hz): mean / median abs / p95 | Octave bands per window, mean abs diff | Onset |
|---|---|---|---|
| Pad 405 m | −0.21 / 0.27 / 1.67 dB (r = 0.999) | 0.40-0.79 dB (max 2.6 dB, at 250 Hz during the crackle peak) | −1.50 s in both |
| 5 km NW | +0.11 / 1.43 / 5.05 dB (r = 0.977) | 0.36-0.45 dB up to T+150 s; 1.7 dB in the late tail | 11.50 s in both |
| Onboard | +0.05 / 0.05 / 0.13 dB (r = 1.000) | 0.06-0.61 dB | −2.50 s in both |

- Ignition arrival matches Python to within 2 ms at 405 m, 5 km and 20 km. MECO-tail arrival matches to within
  0.3 s on a 441 s delay.
- Crackle dSk per window is 80-85 % of the Python value (15.8 vs 18.6 at the pad, 13.5 vs 16.3 at 5 km). The
  octave EQ cannot reproduce the 1/3-octave detail of the crackle undo-absorption.
- Unit tests (`tests/audio.test.ts`, 27 tests) check the following:
  - the band SPL against `tools/audio/export_runtime_fixture.py`: within 0.02 dB in every octave band, over
    35 state × listener cases;
  - delays (within 0.1 ms) and the reception → emission map;
  - the ISO table, spreading and the near-field limit, and absorption vs distance;
  - the vacuum mute and the onboard rules;
  - event delays and one-shot scheduling;
  - the warp policy, the EQ fit and seek determinism.

### 11.6 Limits

- **Infrasound**: late in flight, the whole-band RMS at the ground listeners is 4-6 dB lower than Python's, and
  all of the difference lies below 20 Hz. The runtime's octave grid is flat below the 16 Hz band, while Python's
  1/3-octave grid extends to 6.3 Hz, and at high Doppler the rumble is shifted into that range. Above 20 Hz the
  windows agree within 0.4 dB.
- **Read-rate clamp**: the profile clamps the read rate to [0.5, 1.5]; `render_launch.py` does not. After about
  T+80 s the vehicle recedes faster than sound, and Python pitch-shifts by up to 1/7. A diagnostic render with
  the clamp relaxed (`obs5km_noclamp`) cuts the octave difference in the late 5 km window from 1.74 to 0.59 dB.
- The runtime uses octave resolution (11 bands), not 1/3 octave. Extreme absorption cliffs are fitted where they
  are audible (within 50 dB of the loudest band) and only bounded below that.
- For the attached cameras (onboard, chase), the parameters use history frames up to 0.1 s old.
- The profile's limiter has attack and release times; the runtime uses the reference renderer's static soft
  clip instead.
- Mobile CPU has not been measured on a real device. The per-tick cost and node count above are the budget.
