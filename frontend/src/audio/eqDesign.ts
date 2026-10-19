// Octave-band EQ realised with Web Audio BiquadFilterNodes:
//   gain g0 -> 10 high-shelf biquads (S = 1) at the geometric mid-points between the 11 octave centres
//           -> a 4th-order Butterworth low-pass (2 biquads, the "air" stage; parked at Nyquist when not needed).
// A shelf cascade reproduces broad tilts (spectral balance, directivity, ground gain) and interpolates smoothly
// between band centres like the reference renderer's log-frequency interpolation. Distance absorption is a
// cliff (ISO 9613-1 grows ~f^2: 1 km of air removes 5 dB at 1 kHz but 78 dB at 8 kHz), steeper than the
// ~12 dB/octave a single shelf can make, so the low-pass (24 dB/octave) takes the cliff and the shelves fit the rest.
// Gains are solved with the exact Web Audio biquad formulas: a linear solve with the dB sensitivity matrix
// refined by fixed-Jacobian Newton passes. Deterministic and continuous: the result depends only on the
// target, and moves smoothly with it (no structure switching between control ticks, hence no clicks).
import { NB } from './profile';

export const EQ_MIN_DB = -120;
export const EQ_MAX_DB = 40;
const STEP_MAX = 100;
/** output bands more than this below the loudest output band are "don't care": their target may be raised */
export const DONT_CARE_DB = 50;
/** the steepest per-octave fall asked for inside the don't-care region */
const SLOPE_MAX = 24;
const DAMP = 0.8;
/** 4th-order Butterworth low-pass as two biquads; Web Audio lowpass Q is in dB (Q 0.5412 and 1.3066) */
export const LP_Q_DB = [20 * Math.log10(0.5411961), 20 * Math.log10(1.3065630)];
const lpMag = (f0: number, f: number, fs: number) =>
  biquadMagDb(lowPass(f0, LP_Q_DB[0], fs, tmpQ), f, fs) + biquadMagDb(lowPass(f0, LP_Q_DB[1], fs, tmpQ), f, fs);

export interface Biquad { b0: number; b1: number; b2: number; a0: number; a1: number; a2: number }

/** Web Audio spec (Audio EQ Cookbook) high-shelf coefficients, S = 1 */
export function highShelf(f0: number, gainDb: number, fs: number, out: Biquad = { b0: 0, b1: 0, b2: 0, a0: 1, a1: 0, a2: 0 }): Biquad {
  const A = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * f0) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alphaS = (sw / 2) * Math.SQRT2; // sqrt((A + 1/A)(1/S - 1) + 2) with S = 1
  const k = 2 * Math.sqrt(A) * alphaS;
  out.b0 = A * ((A + 1) + (A - 1) * cw + k);
  out.b1 = -2 * A * ((A - 1) + (A + 1) * cw);
  out.b2 = A * ((A + 1) + (A - 1) * cw - k);
  out.a0 = (A + 1) - (A - 1) * cw + k;
  out.a1 = 2 * ((A - 1) - (A + 1) * cw);
  out.a2 = (A + 1) - (A - 1) * cw - k;
  return out;
}

/** Web Audio spec low-pass coefficients (Q in dB) */
export function lowPass(f0: number, qDb: number, fs: number, out: Biquad = { b0: 0, b1: 0, b2: 0, a0: 1, a1: 0, a2: 0 }): Biquad {
  const w0 = (2 * Math.PI * Math.min(f0, fs / 2)) / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * Math.pow(10, qDb / 20));
  out.b0 = (1 - cw) / 2; out.b1 = 1 - cw; out.b2 = (1 - cw) / 2;
  out.a0 = 1 + alpha; out.a1 = -2 * cw; out.a2 = 1 - alpha;
  return out;
}

/** |H(e^jw)| in dB */
export function biquadMagDb(q: Biquad, f: number, fs: number) {
  const w = (2 * Math.PI * f) / fs;
  const c1 = Math.cos(w), s1 = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
  const nr = q.b0 + q.b1 * c1 + q.b2 * c2, ni = -(q.b1 * s1 + q.b2 * s2);
  const dr = q.a0 + q.a1 * c1 + q.a2 * c2, di = -(q.a1 * s1 + q.a2 * s2);
  return 10 * Math.log10((nr * nr + ni * ni) / (dr * dr + di * di) + 1e-300);
}

export interface ShelfEq { g0: number; shelfDb: Float64Array; shelfHz: Float64Array; lpHz: number /* 0 = off */ }

export function newShelfEq(oct: number[]): ShelfEq {
  const shelfHz = new Float64Array(NB - 1);
  for (let k = 0; k < NB - 1; k++) shelfHz[k] = Math.sqrt(oct[k] * oct[k + 1]);
  return { g0: 0, shelfDb: new Float64Array(NB - 1), shelfHz, lpHz: 0 };
}

const tmpQ: Biquad = { b0: 0, b1: 0, b2: 0, a0: 1, a1: 0, a2: 0 };
const tgt = new Float64Array(NB);
const wgt = new Float64Array(NB);
const outLvl = new Float64Array(NB);
const fAt = (f: number, fs: number) => Math.min(f, fs * 0.499);

/** cascade response (dB) at frequency f */
export function shelfEqResponse(eq: ShelfEq, f: number, fs: number) {
  let r = eq.g0;
  const ff = fAt(f, fs);
  for (let k = 0; k < eq.shelfDb.length; k++) r += biquadMagDb(highShelf(eq.shelfHz[k], eq.shelfDb[k], fs, tmpQ), ff, fs);
  if (eq.lpHz > 0) r += lpMag(eq.lpHz, ff, fs);
  return r;
}

/** precomputed trig at the band centres and shelf corners, per (fs, grid) */
interface Trig { c1: Float64Array; s1: Float64Array; c2: Float64Array; s2: Float64Array; cw: Float64Array; sw: Float64Array; lp: Map<number, Float64Array> }
const trigCache = new Map<string, Trig>();
function trig(oct: number[], fs: number, shelfHz: Float64Array): Trig {
  const key = `${fs}|${oct.join(',')}`;
  let t = trigCache.get(key);
  if (t) return t;
  const n = oct.length;
  t = { c1: new Float64Array(n), s1: new Float64Array(n), c2: new Float64Array(n), s2: new Float64Array(n),
    cw: new Float64Array(n - 1), sw: new Float64Array(n - 1), lp: new Map() };
  for (let b = 0; b < n; b++) {
    const w = (2 * Math.PI * fAt(oct[b], fs)) / fs;
    t.c1[b] = Math.cos(w); t.s1[b] = Math.sin(w); t.c2[b] = Math.cos(2 * w); t.s2[b] = Math.sin(2 * w);
  }
  for (let k = 0; k < n - 1; k++) { const w0 = (2 * Math.PI * shelfHz[k]) / fs; t.cw[k] = Math.cos(w0); t.sw[k] = Math.sin(w0); }
  trigCache.set(key, t);
  return t;
}
/** response (dB) of the whole chain at every band centre */
function responseAll(eq: ShelfEq, oct: number[], fs: number, out: Float64Array) {
  const T = trig(oct, fs, eq.shelfHz);
  const n = oct.length;
  out.fill(eq.g0);
  for (let k = 0; k < n - 1; k++) {
    const A = Math.pow(10, eq.shelfDb[k] / 40), cw = T.cw[k];
    const kk = 2 * Math.sqrt(A) * (T.sw[k] / 2) * Math.SQRT2;
    const b0 = A * ((A + 1) + (A - 1) * cw + kk), b1 = -2 * A * ((A - 1) + (A + 1) * cw), b2 = A * ((A + 1) + (A - 1) * cw - kk);
    const a0 = (A + 1) - (A - 1) * cw + kk, a1 = 2 * ((A - 1) - (A + 1) * cw), a2 = (A + 1) - (A - 1) * cw - kk;
    for (let b = 0; b < n; b++) {
      const c1 = T.c1[b], s1 = T.s1[b], c2 = T.c2[b], s2 = T.s2[b];
      const nr = b0 + b1 * c1 + b2 * c2, ni = b1 * s1 + b2 * s2, dr = a0 + a1 * c1 + a2 * c2, di = a1 * s1 + a2 * s2;
      out[b] += 10 * Math.log10((nr * nr + ni * ni) / (dr * dr + di * di) + 1e-300);
    }
  }
  if (eq.lpHz > 0) {
    let lr = T.lp.get(eq.lpHz);
    if (!lr) { lr = Float64Array.from(oct, (f) => lpMag(eq.lpHz, fAt(f, fs), fs)); T.lp.set(eq.lpHz, lr); }
    for (let b = 0; b < n; b++) out[b] += lr[b];
  }
  return out;
}

/**
 * Bands whose OUTPUT level (asset band level + EQ) is more than DONT_CARE_DB below the loudest output band are
 * inaudible next to it. Raise such targets just enough to keep the fall <= SLOPE_MAX per octave, never above
 * (loudest - DONT_CARE_DB); they are down-weighted in the fit.
 */
function conditionTargets(assetDb: ArrayLike<number>) {
  let omax = -Infinity;
  for (let b = 0; b < NB; b++) { outLvl[b] = assetDb[b] + tgt[b]; omax = Math.max(omax, outLvl[b]); }
  const ceil = omax - DONT_CARE_DB;
  // weight ramps from 1 (audible) to 0.05 over 10 dB below the ceiling: continuous in the target
  for (let b = 0; b < NB; b++) wgt[b] = Math.min(1, Math.max(0.05, (outLvl[b] - (ceil - 10)) / 10));
  for (let pass = 0; pass < 2; pass++) {
    for (let b = 1; b < NB; b++) raise(b, b - 1, ceil, assetDb);
    for (let b = NB - 2; b >= 0; b--) raise(b, b + 1, ceil, assetDb);
  }
}
function raise(b: number, nb: number, ceil: number, assetDb: ArrayLike<number>) {
  if (assetDb[b] + tgt[b] >= ceil) return;
  const want = tgt[nb] - SLOPE_MAX;
  if (want > tgt[b]) tgt[b] = Math.min(want, ceil - assetDb[b]);
}

/** inverse of the dB-per-dB sensitivity matrix (col 0: g0, cols 1..10: shelves) at the band centres */
const jacCache = new Map<string, Float64Array>();
function inverseJacobian(oct: number[], fs: number, shelfHz: Float64Array): Float64Array {
  const key = `${fs}|${oct.join(',')}`;
  const hit = jacCache.get(key);
  if (hit) return hit;
  const n = NB;
  const A = new Float64Array(n * n), I = new Float64Array(n * n);
  for (let b = 0; b < n; b++) {
    A[b * n] = 1;
    for (let k = 0; k < n - 1; k++) A[b * n + k + 1] = biquadMagDb(highShelf(shelfHz[k], 12, fs, tmpQ), fAt(oct[b], fs), fs) / 12;
  }
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let c = 0; c < n; c++) { // Gauss-Jordan, partial pivoting
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[piv * n + c])) piv = r;
    if (piv !== c) for (let j = 0; j < n; j++) {
      let t = A[c * n + j]; A[c * n + j] = A[piv * n + j]; A[piv * n + j] = t;
      t = I[c * n + j]; I[c * n + j] = I[piv * n + j]; I[piv * n + j] = t;
    }
    const d = A[c * n + c];
    for (let j = 0; j < n; j++) { A[c * n + j] /= d; I[c * n + j] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r * n + c];
      if (f !== 0) for (let j = 0; j < n; j++) { A[r * n + j] -= f * A[c * n + j]; I[r * n + j] -= f * I[c * n + j]; }
    }
  }
  jacCache.set(key, I);
  return I;
}

const xv = new Float64Array(NB);
const ev = new Float64Array(NB);
const lpR = new Float64Array(NB);
const rsp = new Float64Array(NB);

/** fit shelves (+ fixed low-pass lpHz) to tgt; returns the weighted max error, leaves the solution in eq */
function fit(oct: number[], fs: number, eq: ShelfEq, lpHz: number, passes: number): number {
  const n = NB, Inv = inverseJacobian(oct, fs, eq.shelfHz);
  eq.lpHz = lpHz;
  eq.g0 = 0; eq.shelfDb.fill(0);
  responseAll(eq, oct, fs, lpR); // = low-pass alone
  for (let i = 0; i < n; i++) { let a = 0; for (let j = 0; j < n; j++) a += Inv[i * n + j] * (tgt[j] - lpR[j]); xv[i] = a; }
  let err = 0;
  for (let it = 0; it <= passes; it++) {
    eq.g0 = xv[0];
    for (let k = 0; k < n - 1; k++) eq.shelfDb[k] = Math.max(-STEP_MAX, Math.min(STEP_MAX, xv[k + 1]));
    err = 0;
    responseAll(eq, oct, fs, rsp);
    for (let b = 0; b < n; b++) { ev[b] = tgt[b] - rsp[b]; err = Math.max(err, wgt[b] * Math.abs(ev[b])); }
    if (err < 0.02 || it === passes) break;
    for (let i = 0; i < n; i++) { let a = 0; for (let j = 0; j < n; j++) a += Inv[i * n + j] * wgt[j] * ev[j]; xv[i] += DAMP * a; }
  }
  return err;
}

/** local fall (dB per octave) beyond which the low-pass takes over from the shelves */
const LP_SLOPE_DB = 18;

/**
 * Low-pass corner from the (conditioned) target, a CONTINUOUS function of it: the frequency (log-interpolated
 * between octave mid-points) where the target's local slope first becomes steeper than -LP_SLOPE_DB per
 * octave. Without such a cliff the corner is extrapolated above 11 kHz and parks at Nyquist beyond 0.45 fs.
 * Continuity matters: switching the filter structure between control ticks makes clicks.
 */
function lpCorner(oct: number[], fs: number): number {
  let prevS = 0, prevF = oct[0];
  for (let b = 1; b < NB; b++) {
    const s = tgt[b] - tgt[b - 1];
    const fm = Math.sqrt(oct[b] * oct[b - 1]);
    if (s <= -LP_SLOPE_DB) {
      if (b === 1) return fm;
      const f = (-LP_SLOPE_DB - prevS) / (s - prevS);
      return prevF * Math.pow(fm / prevF, Math.min(Math.max(f, 0), 1));
    }
    prevS = s; prevF = fm;
  }
  const top = prevF * Math.pow(2, (LP_SLOPE_DB + prevS) / 6);
  return top >= fs * 0.45 ? 0 : top;
}

/**
 * Design the chain so it hits `targetDb` at the octave centres. Targets are clamped to [EQ_MIN_DB, EQ_MAX_DB];
 * with `assetDb` (the group's asset band level) inaudible bands are conditioned and down-weighted.
 * Returns the weighted max residual (dB): audible bands count fully.
 */
export function designShelfEq(targetDb: ArrayLike<number>, oct: number[], fs: number, eq: ShelfEq, passes = 5,
  assetDb?: ArrayLike<number>): number {
  for (let b = 0; b < NB; b++) { tgt[b] = Math.min(EQ_MAX_DB, Math.max(EQ_MIN_DB, targetDb[b])); wgt[b] = 1; }
  if (assetDb) conditionTargets(assetDb);
  return fit(oct, fs, eq, lpCorner(oct, fs), passes);
}
