// Mission clock, time control, replay, and secondary bodies (spent stage, fairing halves).
// The vehicle pose shown in the scene always comes from the 6-DOF simulation state
// (live) or from the recorded simulation history (replay) - never from an animation curve.
import type { VehicleConfig } from '../physics/config';
import { COAST_SEP, MU, NSTATE, ORBIT, PAD, RE, S2_DEPLETED, evaluate, newTelemetry, type Telemetry, type Ctx } from '../physics/dynamics';
import { MissionSim, type Stepper, type HistoryFrame } from '../physics/sim';
import { elements, type Elements } from '../physics/orbit';
import { atmosphere } from '../physics/atmosphere';

export type MissionState = 'idle' | 'countdown' | 'flight' | 'done';
export const COUNTDOWN_S = 10; // T-10 s: sound-suppression water (artistic) then engine start at T-3

export interface Debris {
  y: Float64Array; // r(3) v(3) q(4) w(3)
  t: number;
  active: boolean;
}

export class MissionController {
  sim!: MissionSim;
  state: MissionState = 'idle';
  /** mission elapsed time (T+), negative during countdown */
  met = -COUNTDOWN_S;
  warp = 1;
  paused = false;
  replay = false;
  replayT = 0;
  tel: Telemetry = newTelemetry();
  /** pose shown this frame */
  view: { t: number; y: Float64Array; phase: number; fairing: boolean } = { t: 0, y: new Float64Array(NSTATE), phase: PAD, fairing: true };
  s1: Debris | null = null;
  fairingL: Debris | null = null;
  fairingR: Debris | null = null;
  orbitEl: Elements | null = null;
  maxQ = { qbar: 0, t: 0, h: 0, mach: 0 };
  downrange = 0;
  private ctxTmp: Ctx = { phase: 0, fairing: true, t_meco: 0, t_sep: 0, t_ses: 0, t_seco: 0 };

  constructor(public cfg: VehicleConfig, public makeStepper: (cfg: VehicleConfig) => Stepper | undefined) {
    this.reset();
  }

  reset() {
    this.sim = new MissionSim(this.cfg, 0.02, this.makeStepper(this.cfg));
    this.state = 'idle';
    this.met = -COUNTDOWN_S;
    this.replay = false;
    this.paused = false;
    this.s1 = this.fairingL = this.fairingR = null;
    this.orbitEl = null;
    this.view.y.set(this.sim.y);
    this.view.t = this.sim.t;
    this.view.phase = PAD;
    this.view.fairing = true;
    this.updateTelemetry();
  }

  launch() {
    if (this.state === 'idle') this.state = 'countdown';
  }

  get t0() { return this.sim.history[0].t; }

  update(dtReal: number) {
    const dt = this.paused ? 0 : dtReal * this.warp;
    if (this.replay) {
      if (!this.paused) this.replayT = Math.min(this.replayT + dt, this.sim.t);
      this.sampleHistory(this.replayT);
      this.met = this.replayT;
    } else if (this.state === 'countdown') {
      this.met += dt;
      if (this.met >= this.sim.t) { this.state = 'flight'; }
    }
    if (!this.replay && this.state === 'flight') {
      this.met += dt;
      // cap work per frame (very high warp)
      this.sim.advanceTo(this.met, 50000);
      if (this.sim.t < this.met - 1) this.met = this.sim.t;
      this.view.t = this.sim.t;
      this.view.y.set(this.sim.y);
      this.view.phase = this.sim.ctx.phase;
      this.view.fairing = this.sim.ctx.fairing;
      if (this.sim.done()) this.state = this.sim.ctx.phase === ORBIT ? 'flight' : 'done';
    }
    this.updateDebris();
    this.updateTelemetry();
  }

  private sampleHistory(t: number) {
    const H = this.sim.history;
    if (!H.length) return;
    let lo = 0, hi = H.length - 1;
    if (t <= H[0].t) { this.setView(H[0], H[0], 0); return; }
    if (t >= H[hi].t) { this.setView(H[hi], H[hi], 0); return; }
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (H[m].t <= t) lo = m; else hi = m; }
    const a = H[lo], b = H[hi];
    this.setView(a, b, b.t > a.t ? (t - a.t) / (b.t - a.t) : 0);
    this.view.t = t;
  }

  private setView(a: HistoryFrame, b: HistoryFrame, f: number) {
    const y = this.view.y;
    for (let i = 0; i < NSTATE; i++) y[i] = a.y[i] + f * (b.y[i] - a.y[i]);
    const n = Math.hypot(y[6], y[7], y[8], y[9]);
    for (let i = 6; i < 10; i++) y[i] /= n;
    this.view.phase = f < 1 ? a.phase : b.phase;
    this.view.fairing = f < 1 ? a.fairing : b.fairing;
    this.view.t = a.t + f * (b.t - a.t);
  }

  private ctxFor(phase: number, fairing: boolean): Ctx {
    const c = this.ctxTmp;
    const sc = this.sim.ctx;
    c.phase = phase; c.fairing = fairing; c.t_meco = sc.t_meco; c.t_sep = sc.t_sep; c.t_ses = sc.t_ses; c.t_seco = sc.t_seco;
    return c;
  }

  updateTelemetry() {
    const d = new Float64Array(NSTATE);
    const v = this.view;
    evaluate(v.t, v.y, this.ctxFor(v.phase, v.fairing), this.cfg, d, this.tel);
    this.orbitEl = elements(v.y.subarray(0, 3), v.y.subarray(3, 6));
    // max-Q so far (from history up to displayed time)
    this.maxQ = this.sim.maxQ;
    // downrange: great-circle distance from pad in Earth-fixed frame
    const t = v.t, th = (this.cfg.site.rotating_earth ? 7.292115e-5 : 0) * (t - this.t0);
    const c = Math.cos(th), s = Math.sin(th);
    const x = c * v.y[0] + s * v.y[1], y = -s * v.y[0] + c * v.y[1], z = v.y[2];
    const p = this.sim.history[0].y;
    const dot = (x * p[0] + y * p[1] + z * p[2]) / (Math.hypot(x, y, z) * Math.hypot(p[0], p[1], p[2]));
    this.downrange = RE * Math.acos(Math.min(1, Math.max(-1, dot)));
  }

  // ---------------- secondary bodies (reduced fidelity: 3-DOF ballistic with drag) -------------
  private updateDebris() {
    const sim = this.sim;
    const tNow = this.view.t;
    if (this.s1 && tNow < this.s1.t - 0.5) this.s1 = null;
    if (this.fairingL && tNow < this.fairingL.t - 0.5) this.fairingL = this.fairingR = null;
    if (sim.sepState && !this.s1) {
      this.s1 = { y: sim.sepState.slice(0, 13), t: sim.ctx.t_sep, active: true };
      // separation push (assumed 1.2 m/s relative, along -body x) + small tip-off rate (artistic assumption)
      const y = this.s1.y;
      const [w, x, yy, z] = [y[6], y[7], y[8], y[9]];
      const bx = [1 - 2 * (yy * yy + z * z), 2 * (x * yy + w * z), 2 * (x * z - w * yy)];
      for (let i = 0; i < 3; i++) y[3 + i] -= 1.2 * bx[i];
      y[12] += -0.015;
    }
    if (sim.fairingState && !this.fairingL) {
      const ev = sim.events.find((e) => e.name === 'FAIRING_SEP');
      const mk = (sign: number): Debris => {
        const y = sim.fairingState!.slice(0, 13);
        // separation push: 3 m/s outward along body y (halves split in the body x-y plane; assumed)
        const q = [y[6], y[7], y[8], y[9]];
        const [w, x, yy, z] = q;
        const by = [2 * (x * yy - w * z), 1 - 2 * (x * x + z * z), 2 * (yy * z + w * x)];
        for (let i = 0; i < 3; i++) y[3 + i] += sign * 3.0 * by[i];
        y[10] = 0; y[11] = 0; y[12] = sign * 0.35;
        return { y, t: ev ? ev.t : tNow, active: true };
      };
      this.fairingL = mk(1);
      this.fairingR = mk(-1);
    }
    for (const [d, cdA, m] of [[this.s1, 0.9 * 3.7 * 45 * 0.6, 25600], [this.fairingL, 1.4 * 2.6 * 12, 950], [this.fairingR, 1.4 * 2.6 * 12, 950]] as [Debris | null, number, number][]) {
      if (!d || !d.active) continue;
      if (tNow < d.t) continue; // replay before separation
      let steps = 0;
      while (d.t < tNow - 1e-6 && steps++ < 5000) {
        const h = Math.min(0.1, tNow - d.t);
        debrisStep(d.y, h, cdA / m, this.cfg.site.rotating_earth);
        d.t += h;
      }
      if (Math.hypot(d.y[0], d.y[1], d.y[2]) - RE < 0) d.active = false;
    }
  }

  isSeparated() { return this.view.phase > COAST_SEP; }
}

/** semi-implicit point-mass step + free rotation for debris (visual-fidelity) */
function debrisStep(y: Float64Array, h: number, cdA_over_m: number, rot: boolean) {
  const wE = rot ? 7.292115e-5 : 0;
  const r = Math.hypot(y[0], y[1], y[2]);
  const alt = r - RE;
  const atm = atmosphere(alt);
  const vrx = y[3] + wE * y[1], vry = y[4] - wE * y[0], vrz = y[5];
  const V = Math.hypot(vrx, vry, vrz);
  const k = 0.5 * atm.rho * V * cdA_over_m;
  const g = -MU / (r * r * r);
  y[3] += h * (g * y[0] - k * vrx);
  y[4] += h * (g * y[1] - k * vry);
  y[5] += h * (g * y[2] - k * vrz);
  y[0] += h * y[3]; y[1] += h * y[4]; y[2] += h * y[5];
  // quaternion integration with body rate
  const [qw, qx, qy, qz] = [y[6], y[7], y[8], y[9]];
  const ox = y[10], oy = y[11], oz = y[12];
  y[6] += 0.5 * h * (-qx * ox - qy * oy - qz * oz);
  y[7] += 0.5 * h * (qw * ox + qy * oz - qz * oy);
  y[8] += 0.5 * h * (qw * oy - qx * oz + qz * ox);
  y[9] += 0.5 * h * (qw * oz + qx * oy - qy * ox);
  const n = Math.hypot(y[6], y[7], y[8], y[9]);
  y[6] /= n; y[7] /= n; y[8] /= n; y[9] /= n;
}
