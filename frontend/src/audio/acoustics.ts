// Executable form of the acoustic profile for the browser: a line-by-line port of
// tools/audio/acoustic_model.py (the reference), evaluated in octave bands (the profile's runtime grid).
// Pure functions, no DOM: unit-tested in tests/audio.test.ts against the profile tables.
import { atmosphere, type AtmoOut } from '../physics/atmosphere';
import type { VehicleConfig } from '../physics/config';
import { plumeState } from '../engine/plumeModel';
import { NB, type PreparedProfile } from './profile';

export const P_REF = 101325.0;
const T0 = 293.15;
const T01 = 273.16;

export type V3 = [number, number, number];

/** One telemetry sample in the listener frame (pad-local ENU, metres). Same columns as
 *  data/audio/validation/telemetry_reference.csv / render_launch.py TEL_COLS. */
export interface SourceFrame {
  t: number; phase: number; thrust: number; mdot: number; throttle: number;
  P: number; T: number; rho: number; a: number; h: number; mach: number; qbar: number;
  noz: V3; // active-stage nozzle exit
  ax: V3; // exhaust-flow direction (unit)
}

export const newFrame = (): SourceFrame => ({
  t: 0, phase: 0, thrust: 0, mdot: 0, throttle: 0, P: P_REF, T: 288.15, rho: 1.225, a: 340.294, h: 0, mach: 0, qbar: 0,
  noz: [0, 0, 0], ax: [0, 0, -1],
});

export const smoothstep = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);

// ------------------------------------------------------------------ ISO 9613-1
export function molarH2oFromRh(rhPercent: number, T: number, P: number) {
  const psatPr = Math.pow(10, -6.8346 * Math.pow(T01 / T, 1.261) + 4.6151);
  return (rhPercent * psatPr) / (P / P_REF);
}

/** pure-tone absorption coefficient [dB/m], ISO 9613-1:1993 eqs. 3-5 */
export function iso9613Alpha(f: number, T: number, P: number, h: number) {
  const pr = P / P_REF;
  const tr = T / T0;
  const frO = pr * (24 + (4.04e4 * h * (0.02 + h)) / (0.391 + h));
  const frN = pr * Math.pow(tr, -0.5) * (9 + 280 * h * Math.exp(-4.17 * (Math.pow(tr, -1 / 3) - 1)));
  return 8.686 * f * f * ((1.84e-11 / pr) * Math.sqrt(tr) + Math.pow(tr, -2.5) * (
    (0.01275 * Math.exp(-2239.1 / T)) / (frO + (f * f) / frO) + (0.1068 * Math.exp(-3352.0 / T)) / (frN + (f * f) / frN)));
}

const atmTmp: AtmoOut = { T: 0, P: 0, rho: 0, a: 0 };
let h0Cache = -1;
let h0Key = '';
/** water-vapour molar concentration [%] vs altitude (profile humidity_profile, never above saturation) */
export function h2oAt(z: number, p: PreparedProfile) {
  const hp = p.raw.propagation.atmospheric_absorption.humidity_profile;
  const key = `${hp.surface_RH_percent}/${hp.h2o_scale_height_m}`;
  if (key !== h0Key) {
    const g = atmosphere(0);
    h0Cache = molarH2oFromRh(hp.surface_RH_percent, g.T, g.P);
    h0Key = key;
  }
  const zz = Math.max(z, 0);
  atmosphere(zz, atmTmp);
  const hsat = molarH2oFromRh(100, atmTmp.T, atmTmp.P);
  return Math.min(h0Cache * Math.exp(-zz / hp.h2o_scale_height_m), hsat);
}

/** Straight-ray atmosphere samples (24 points, as acoustic_model.path_absorption_db). */
export class Ray {
  n: number;
  r = 0;
  T: Float64Array; P: Float64Array; H: Float64Array; invA: Float64Array;
  constructor(n = 24) {
    this.n = n;
    this.T = new Float64Array(n); this.P = new Float64Array(n); this.H = new Float64Array(n); this.invA = new Float64Array(n);
  }
  set(src: ArrayLike<number>, lis: ArrayLike<number>, p: PreparedProfile) {
    const n = this.n;
    this.r = Math.hypot(src[0] - lis[0], src[1] - lis[1], src[2] - lis[2]);
    for (let i = 0; i < n; i++) {
      const s = i / (n - 1);
      const z = Math.max(lis[2] + (src[2] - lis[2]) * s, 0);
      atmosphere(z, atmTmp);
      this.T[i] = atmTmp.T;
      this.P[i] = Math.max(atmTmp.P, 1e-6);
      this.H[i] = h2oAt(z, p);
      this.invA[i] = 1 / atmTmp.a;
    }
    return this;
  }
  /** total absorption [dB] along the ray at each frequency (trapezoid in s, times r) */
  absorption(freqs: ArrayLike<number>, out: Float64Array) {
    const n = this.n;
    if (this.r < 1) { out.fill(0); return out; }
    for (let b = 0; b < freqs.length; b++) {
      let acc = 0;
      for (let i = 0; i < n; i++) {
        const w = i === 0 || i === n - 1 ? 0.5 : 1;
        acc += w * iso9613Alpha(freqs[b], this.T[i], this.P[i], this.H[i]);
      }
      out[b] = (acc / (n - 1)) * this.r;
    }
    return out;
  }
}

const cRay = new Ray(16);
/** harmonic mean of the USSA-1976 sound speed along the ray (travel time = r / c_path) */
export function cPath(src: ArrayLike<number>, lis: ArrayLike<number>, p: PreparedProfile) {
  // only the sound speed is needed; Ray.set also evaluates humidity, which is cheap
  cRay.set(src, lis, p);
  const n = cRay.n;
  let acc = 0;
  for (let i = 0; i < n; i++) acc += (i === 0 || i === n - 1 ? 0.5 : 1) * cRay.invA[i];
  return 1 / (acc / (n - 1));
}

// ------------------------------------------------------------------ source
export interface SourceState {
  U: number; W: number; Lw: number; De: number; D_spec: number; n: number; xt: number; x_src: number;
  Me: number; Mc: number; w_crackle: number; f_peak: number; pa: number; w_src: number;
}

export function sourceState(p: PreparedProfile, veh: VehicleConfig, f: SourceFrame): SourceState | null {
  const eng = f.phase <= 2 ? veh.engines.stage1 : veh.engines.stage2;
  const n = eng.count;
  const F = f.thrust, mdot = f.mdot, pa = f.P;
  if (F <= 1 || mdot <= 1e-6) return null;
  const U = F / mdot;
  const src = p.raw.source;
  const co = src.source_continuum_cutoff;
  const w_src = clamp((Math.log10(Math.max(pa, 1e-9)) - Math.log10(co.pa_lo)) / (Math.log10(co.pa_hi) - Math.log10(co.pa_lo)), 0, 1);
  const W = src.acoustic_efficiency.value * 0.5 * mdot * U * U * w_src;
  const de1 = 2 * Math.sqrt(eng.ae_m2 / Math.PI);
  const De = Math.sqrt(n) * de1;
  const thr = Math.max(f.throttle, 0.05);
  const ps = plumeState(eng, thr, pa);
  const D_spec = pa < ps.pe ? Math.max(De, Math.sqrt(n) * ps.dj) : De;
  const Me = ps.Me;
  const xt = 3.45 * De * (1 + 0.38 * Me) ** 2;
  const Mc = (0.65 * U) / f.a;
  const w_c = smoothstep((Mc - 1.5) / 2.0) * Math.sqrt(clamp(pa / P_REF, 0, 1));
  return {
    U, W, Lw: 10 * Math.log10(Math.max(W, 1e-12)) + 120, De, D_spec, n, xt, x_src: 1.5 * xt, Me, Mc, w_crackle: w_c,
    f_peak: (0.02 * U) / D_spec, pa, w_src,
  };
}

/** SP-8072 normalised spectrum level at Strouhal st (end slopes continued log-linearly) */
export function spectrumLevelDb(p: PreparedProfile, st: number) {
  const x = p.specLogSt, y = p.specDb, n = x.length;
  const ls = Math.log(st);
  if (ls <= x[0]) return y[0] + ((y[1] - y[0]) / (x[1] - x[0])) * (ls - x[0]);
  if (ls >= x[n - 1]) return y[n - 1] + ((y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])) * (ls - x[n - 1]);
  let i = 0;
  while (x[i + 1] < ls) i++;
  return y[i] + ((y[i + 1] - y[i]) * (ls - x[i])) / (x[i + 1] - x[i]);
}

/** sound-power level per band [dB re 1 pW] (trapezoid in log St over 8 points, as band_power_db) */
export function bandPowerDb(p: PreparedProfile, Lw: number, U: number, D: number, out: Float64Array, nPts = 8) {
  for (let b = 0; b < NB; b++) {
    const a = (p.octLo[b] * D) / U, c = (p.octHi[b] * D) / U;
    const la = Math.log(a), lc = Math.log(c);
    const dl = (lc - la) / (nPts - 1);
    let frac = 0;
    for (let i = 0; i < nPts; i++) {
      const ls = la + i * dl;
      const st = Math.exp(ls);
      const v = Math.pow(10, spectrumLevelDb(p, st) / 10) * st;
      frac += (i === 0 || i === nPts - 1 ? 0.5 : 1) * v * dl;
    }
    out[b] = Lw + 10 * Math.log10(Math.max(frac, 1e-30));
  }
  return out;
}

function interp1(xs: ArrayLike<number>, ys: ArrayLike<number>, x: number) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (xs[i + 1] < x) i++;
  return ys[i] + ((ys[i + 1] - ys[i]) * (x - xs[i])) / (xs[i + 1] - xs[i]);
}

/** directivity index [dB]: angle interpolation per Strouhal row, then log-Strouhal interpolation (clamped) */
export function directivityDb(p: PreparedProfile, thetaDeg: number, st: number) {
  const ks = p.dirSt;
  const col = new Array<number>(ks.length);
  for (let k = 0; k < ks.length; k++) col[k] = interp1(p.dirAng, p.dirTab[k], thetaDeg);
  const lk = ks.map(Math.log);
  return interp1(lk, col, Math.log(clamp(st, ks[0], ks[ks.length - 1])));
}

/** ground reflection [dB]: +6 below fc, +3 above 4 fc (log interpolation); elevated: +3 below 40 Hz */
export function groundGainDb(f: number, c: number, hL: number, psiRad: number, elevated: boolean) {
  if (elevated) return f < 40 ? 3 : 0;
  const dpath = Math.max(2 * hL * Math.sin(Math.max(psiRad, 1e-3)), 1e-3);
  const fc = c / (4 * dpath);
  const x = clamp(Math.log(Math.max(f, 1e-3) / fc) / Math.log(4), 0, 1);
  return 6 - 3 * x;
}

export const padWeight = (hNozzle: number, De: number) => Math.exp(-Math.max(hNozzle, 0) / (10 * De));
export const onboardAirWeight = (mach: number, rho: number, a: number) =>
  (1 - smoothstep((mach - 0.8) / 0.4)) * Math.sqrt(Math.max(rho * a, 0) / (1.225 * 340.3));
export const aeroWeight = (qbar: number, mach: number) => (qbar / 30000) * (0.35 + 0.65 * Math.exp(-(((mach - 1) / 0.25) ** 2)));
export const cabinTlDb = (f: number, ms = 20) => Math.max(0, 20 * Math.log10(ms * Math.max(f, 1)) - 47);

/** 10 log10(rho c / rho0c0) at the listener altitude; -Infinity in vacuum (P < 1 Pa) */
export function listenerImpedanceDb(alt: number, p: PreparedProfile) {
  const a = atmosphere(Math.max(alt, 0));
  if (a.P < 1) return -Infinity;
  return 10 * Math.log10((a.rho * a.a) / p.raw.propagation.geometric_spreading.rho0c0);
}

/** crackle layer LINEAR amplitude weight (multiplies the jet-group EQ gain) */
export function crackleWeight(p: PreparedProfile, wSource: number, thetaDeg: number, r: number, xt: number) {
  const c = p.raw.crackle;
  const wDir = Math.exp(-(((thetaDeg - c.directivity.peak_deg) / c.directivity.width_deg) ** 2));
  const nl = c.nonlinear_growth;
  const g = 1 + nl.max_gain * Math.min(1, Math.log2(1 + Math.log(Math.max(r, xt) / xt)) / nl.log2_saturation);
  return clamp(wSource * c.reference_weight * wDir * g, 0, c.max_weight);
}

/** apparent jet source: x_src down the plume, never below ground (the rest is the pad term) */
export function apparentSource(st: SourceState | null, noz: V3, ax: V3, out: V3 = [0, 0, 0]): V3 {
  let x = st ? st.x_src : 0;
  const al = Math.hypot(ax[0], ax[1], ax[2]) + 1e-12;
  const a0 = ax[0] / al, a1 = ax[1] / al, a2 = ax[2] / al;
  if (a2 < -1e-3 && noz[2] > 0) x = Math.min(x, Math.max(noz[2], 0) / -a2);
  else if (noz[2] <= 0.5) x = 0;
  out[0] = noz[0] + a0 * x; out[1] = noz[1] + a1 * x; out[2] = Math.max(noz[2] + a2 * x, 1);
  return out;
}

export interface SplGeom { r: number; theta: number; src: V3; c: number; delay: number }

const tmpPow = new Float64Array(NB);
const tmpAbs = new Float64Array(NB);
const rayJet = new Ray(24);
/**
 * Free-jet band SPL at the listener (acoustic_model.band_spl): spreading with the near-field limit,
 * directivity, listener rho*c, ISO 9613-1 absorption along the ray, ground reflection.
 * Writes L (dB per octave band); returns geometry. `ray` (optional) receives the ray samples for reuse.
 */
export function bandSpl(p: PreparedProfile, st: SourceState, noz: V3, ax: V3, lis: V3, L: Float64Array,
  opts: { ground: boolean; absorb: boolean; thetaOverride?: number; rOverride?: number }, ray: Ray = rayJet): SplGeom {
  const src = apparentSource(st, noz, ax);
  const al = Math.hypot(ax[0], ax[1], ax[2]) + 1e-12;
  const v0 = lis[0] - src[0], v1 = lis[1] - src[1], v2 = lis[2] - src[2];
  let r = Math.hypot(v0, v1, v2);
  let theta = (Math.acos(clamp((v0 * ax[0] + v1 * ax[1] + v2 * ax[2]) / ((r + 1e-9) * al), -1, 1)) * 180) / Math.PI;
  if (opts.thetaOverride !== undefined) theta = opts.thetaOverride;
  const rGeom = r;
  if (opts.rOverride !== undefined) r = opts.rOverride;
  const rEff = Math.sqrt(r * r + st.xt * st.xt);
  bandPowerDb(p, st.Lw, st.U, st.D_spec, tmpPow);
  const imp = listenerImpedanceDb(lis[2], p);
  const spread = 10 * Math.log10(4 * Math.PI * rEff * rEff);
  for (let b = 0; b < NB; b++) L[b] = tmpPow[b] + directivityDb(p, theta, (p.oct[b] * st.D_spec) / st.U) - spread + imp;
  const c = cPath(src, lis, p);
  if (opts.absorb) {
    ray.set(src, lis, p);
    if (opts.rOverride !== undefined) ray.r = r;
    ray.absorption(p.oct, tmpAbs);
    for (let b = 0; b < NB; b++) L[b] -= tmpAbs[b];
  }
  if (opts.ground) {
    const hL = p.raw.propagation.ground_reflection.h_listener_m;
    const psi = Math.asin(clamp((src[2] - lis[2]) / (rGeom + 1e-9), 0, 1));
    const elevated = lis[2] > 30;
    for (let b = 0; b < NB; b++) L[b] += groundGainDb(p.oct[b], c, hL, psi, elevated);
  }
  return { r, theta, src, c, delay: rGeom / c };
}

export const oaDb = (L: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < L.length; i++) s += Math.pow(10, L[i] / 10);
  return 10 * Math.log10(s + 1e-30);
};
