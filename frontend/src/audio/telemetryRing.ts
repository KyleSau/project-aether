// Telemetry history for emission-time evaluation. A fixed-capacity ring of SourceFrames (column storage,
// no allocation per push) with time interpolation, plus the emission-time solver
//   t_r = t_e + |x_src(t_e) - x_L| / c_path   ->   t_e
// Doppler is NOT computed separately: it is the read rate d t_e / d t_r of this delay line.
import type { SourceFrame, V3 } from './acoustics';

const NF = 18; // t, phase, thrust, mdot, throttle, P, T, rho, a, h, mach, qbar, noz(3), ax(3)

export class TelemetryRing {
  readonly cap: number;
  private d: Float64Array;
  private head = 0; // index of the oldest frame
  size = 0;
  constructor(capacity = 4096) {
    this.cap = capacity;
    this.d = new Float64Array(capacity * NF);
  }
  clear() { this.head = 0; this.size = 0; }
  get t0() { return this.size ? this.d[this.head * NF] : NaN; }
  get tLast() { return this.size ? this.d[((this.head + this.size - 1) % this.cap) * NF] : NaN; }
  /** seconds of history held */
  get span() { return this.size ? this.tLast - this.t0 : 0; }

  push(f: SourceFrame) {
    if (this.size && f.t <= this.tLast + 1e-9) {
      // same time stamp (event boundary): the later sample carries the new phase -> replace
      if (Math.abs(f.t - this.tLast) <= 1e-9) { this.write((this.head + this.size - 1) % this.cap, f); }
      return;
    }
    let i: number;
    if (this.size < this.cap) { i = (this.head + this.size) % this.cap; this.size++; }
    else { i = this.head; this.head = (this.head + 1) % this.cap; }
    this.write(i, f);
  }

  private write(i: number, f: SourceFrame) {
    const o = i * NF, d = this.d;
    d[o] = f.t; d[o + 1] = f.phase; d[o + 2] = f.thrust; d[o + 3] = f.mdot; d[o + 4] = f.throttle; d[o + 5] = f.P;
    d[o + 6] = f.T; d[o + 7] = f.rho; d[o + 8] = f.a; d[o + 9] = f.h; d[o + 10] = f.mach; d[o + 11] = f.qbar;
    d[o + 12] = f.noz[0]; d[o + 13] = f.noz[1]; d[o + 14] = f.noz[2]; d[o + 15] = f.ax[0]; d[o + 16] = f.ax[1]; d[o + 17] = f.ax[2];
  }

  private tAt(k: number) { return this.d[((this.head + k) % this.cap) * NF]; }

  /** Interpolated frame at time t (clamped to the held span). Returns false when empty or t < t0. */
  sample(t: number, out: SourceFrame): boolean {
    if (!this.size || t < this.t0 - 1e-9) return false;
    let lo = 0, hi = this.size - 1;
    if (t >= this.tAt(hi)) { lo = hi; }
    else {
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.tAt(m) <= t) lo = m; else hi = m; }
    }
    const a = ((this.head + lo) % this.cap) * NF;
    const b = ((this.head + Math.min(lo + 1, this.size - 1)) % this.cap) * NF;
    const d = this.d;
    const ta = d[a], tb = d[b];
    const f = tb > ta ? Math.min(Math.max((t - ta) / (tb - ta), 0), 1) : 0;
    const L = (k: number) => d[a + k] + f * (d[b + k] - d[a + k]);
    out.t = t;
    out.phase = f < 1 ? d[a + 1] : d[b + 1];
    out.thrust = L(2); out.mdot = L(3); out.throttle = L(4); out.P = L(5); out.T = L(6); out.rho = L(7); out.a = L(8);
    out.h = L(9); out.mach = L(10); out.qbar = L(11);
    out.noz[0] = L(12); out.noz[1] = L(13); out.noz[2] = L(14);
    const x = L(15), y = L(16), z = L(17), n = Math.hypot(x, y, z) || 1;
    out.ax[0] = x / n; out.ax[1] = y / n; out.ax[2] = z / n;
    // never interpolate thrust across an engine cut/start inside one interval: keep the earlier sample's
    // burning state when phases differ (MECO/SES boundaries are recorded as their own samples)
    if (d[a + 1] !== d[b + 1] && f < 1) { out.thrust = d[a + 2]; out.mdot = d[a + 3]; out.throttle = d[a + 4]; }
    return true;
  }
}

/**
 * Emission time for a listener at `lis` and reception (sim) time t:  t_e + r(t_e)/c = t.
 * Solved from the previous t_e with Newton steps on g(te) = te + r(te)/c - t, i.e. the fixed-point
 * iteration with its slope 1 + (dr/dte)/c divided out. (The plain fixed point te <- t - r(te)/c diverges
 * once the radial speed exceeds c, which this vehicle reaches at about T+60 s for a 5 km listener.)
 * Falls back to bisection if the steps do not converge. Returns NaN when t_e precedes the history.
 */
export function solveEmissionTime(t: number, guess: number, c: number, rAt: (te: number) => number, tMin: number, iters = 3): number {
  const g = (te: number) => te + rAt(te) / c - t;
  let te = Math.min(isFinite(guess) ? guess : t, t);
  const h = 0.05;
  for (let k = 0; k < iters; k++) {
    const g0 = g(te);
    if (Math.abs(g0) < 1e-5) return te < tMin ? NaN : te;
    const slope = 1 + (rAt(te + h) - rAt(te - h)) / (2 * h * c);
    te = te - g0 / Math.max(slope, 0.05);
    if (te > t) te = t;
  }
  if (Math.abs(g(te)) < 2e-3) return te < tMin ? NaN : te;
  // bracketed bisection on [tMin, t] (g increasing for receding/slow sources)
  let a = tMin, b = t;
  if (g(a) > 0) return NaN; // sound emitted at tMin has not arrived yet
  for (let k = 0; k < 60; k++) {
    const m = 0.5 * (a + b);
    if (g(m) > 0) b = m; else a = m;
    if (b - a < 1e-5) break;
  }
  return 0.5 * (a + b);
}

export const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
