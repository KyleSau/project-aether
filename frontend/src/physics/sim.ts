// Real-time mission simulation: fixed-step RK4 with event location (secant on state events,
// exact landing on scheduled events). Same event sequence as physics/aether_physics/sim.py.
import type { VehicleConfig } from './config';
import {
  COAST_S2, COAST_SEP, Ctx, evaluate, initialState, MU, NSTATE, newCtx, newTelemetry, ORBIT, PAD,
  RE, S1_BURN, S2_BURN, S2_DEPLETED, Telemetry,
} from './dynamics';

export interface SimEvent { name: string; t: number }

export interface Stepper {
  /** advance y in place by one RK4 step of size h at time t with ctx */
  rk4(t: number, y: Float64Array, ctx: Ctx, h: number): void;
}

export class TsStepper implements Stepper {
  private k1 = new Float64Array(NSTATE);
  private k2 = new Float64Array(NSTATE);
  private k3 = new Float64Array(NSTATE);
  private k4 = new Float64Array(NSTATE);
  private tmp = new Float64Array(NSTATE);
  constructor(private cfg: VehicleConfig) {}
  rk4(t: number, y: Float64Array, ctx: Ctx, h: number) {
    const { k1, k2, k3, k4, tmp, cfg } = this;
    evaluate(t, y, ctx, cfg, k1);
    for (let i = 0; i < NSTATE; i++) tmp[i] = y[i] + 0.5 * h * k1[i];
    evaluate(t + 0.5 * h, tmp, ctx, cfg, k2);
    for (let i = 0; i < NSTATE; i++) tmp[i] = y[i] + 0.5 * h * k2[i];
    evaluate(t + 0.5 * h, tmp, ctx, cfg, k3);
    for (let i = 0; i < NSTATE; i++) tmp[i] = y[i] + h * k3[i];
    evaluate(t + h, tmp, ctx, cfg, k4);
    for (let i = 0; i < NSTATE; i++) y[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    // renormalise quaternion
    const n = Math.hypot(y[6], y[7], y[8], y[9]);
    y[6] /= n; y[7] /= n; y[8] /= n; y[9] /= n;
  }
}

export interface HistoryFrame {
  t: number;
  phase: number;
  fairing: boolean;
  y: Float64Array;
}

const energy = (y: ArrayLike<number>) =>
  0.5 * (y[3] * y[3] + y[4] * y[4] + y[5] * y[5]) - MU / Math.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
const alt = (y: ArrayLike<number>) => Math.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) - RE;

export class MissionSim {
  t: number;
  y: Float64Array;
  ctx: Ctx;
  events: SimEvent[] = [];
  history: HistoryFrame[] = [];
  historyDt = 0.1;
  lost = false;
  maxQ = { qbar: 0, t: 0, h: 0, mach: 0 };
  stepCount = 0;
  /** stage-1 state at separation (for the debris propagator) */
  sepState: Float64Array | null = null;
  fairingState: Float64Array | null = null;
  private tel = newTelemetry();
  private yPrev = new Float64Array(NSTATE);
  private nextHist: number;
  private epsTarget: number;

  constructor(public cfg: VehicleConfig, public h = 0.02, public stepper: Stepper = new TsStepper(cfg), tStart = -3) {
    this.t = Math.min(tStart, -cfg.engines.stage1.startup_s);
    this.y = initialState(cfg);
    this.ctx = newCtx();
    this.nextHist = this.t;
    this.epsTarget = -MU / (2 * (RE + cfg.guidance.target_alt_m));
    this.emit('IGNITION');
    this.record();
  }

  private record() {
    if (this.ctx.phase <= S2_BURN && this.ctx.phase !== PAD) {
      this.telemetry();
      if (this.tel.qbar > this.maxQ.qbar) this.maxQ = { qbar: this.tel.qbar, t: this.t, h: this.tel.h, mach: this.tel.mach };
    }
    this.history.push({ t: this.t, phase: this.ctx.phase, fairing: this.ctx.fairing, y: this.y.slice() });
    this.nextHist = this.t + this.historyDt;
  }

  private emit(name: string) {
    this.events.push({ name, t: this.t });
    if (this.history.length && this.history[this.history.length - 1].t < this.t) this.record();
  }

  /** time of the next scheduled (time-triggered) event, or Infinity */
  private nextScheduled(): number {
    const g = this.cfg.guidance;
    switch (this.ctx.phase) {
      case PAD: return 0;
      case COAST_SEP: return this.ctx.t_meco + g.sep_delay_s;
      case COAST_S2: return this.ctx.t_sep + g.ses_delay_s;
      default: return Infinity;
    }
  }

  private onScheduled() {
    const c = this.ctx;
    if (c.phase === PAD) {
      c.phase = S1_BURN;
      this.emit('LIFTOFF');
    } else if (c.phase === COAST_SEP) {
      this.sepState = this.y.slice();
      c.phase = COAST_S2;
      c.t_sep = this.t;
      this.emit('STAGE_SEP');
    } else if (c.phase === COAST_S2) {
      c.phase = S2_BURN;
      c.t_ses = this.t;
      this.emit('SES');
    }
  }

  /** state-event functions active in the current phase: [value fn, direction, handler] */
  private stateEvents(): [(y: Float64Array) => number, number, () => void][] {
    const c = this.ctx;
    const out: [(y: Float64Array) => number, number, () => void][] = [];
    if (c.phase === S1_BURN) {
      out.push([(y) => y[13], -1, () => { this.y[13] = 0; c.t_meco = this.t; c.phase = COAST_SEP; this.emit('MECO'); }]);
    }
    if (c.fairing && (c.phase === COAST_S2 || c.phase === S2_BURN)) {
      const fa = this.cfg.guidance.fairing_alt_m;
      out.push([(y) => alt(y) - fa, 1, () => { c.fairing = false; this.fairingState = this.y.slice(); this.emit('FAIRING_SEP'); }]);
    }
    if (c.phase === S2_BURN) {
      out.push([(y) => y[14], -1, () => { this.y[14] = 0; c.phase = S2_DEPLETED; c.t_seco = this.t; this.emit('S2_DEPLETION'); }]);
      out.push([(y) => energy(y) - this.epsTarget, 1, () => { c.phase = ORBIT; c.t_seco = this.t; this.emit('SECO'); }]);
    }
    if (c.phase !== PAD && c.phase !== ORBIT) {
      out.push([(y) => alt(y) + 5, -1, () => { this.lost = true; this.emit('LOSS_OF_VEHICLE'); }]);
    }
    return out;
  }

  /** advance one step (<= h), handling events. */
  private stepOnce(tLimit: number) {
    let h = Math.min(this.h, tLimit - this.t);
    const tSched = this.nextScheduled();
    let hitSched = false;
    if (this.t + h >= tSched - 1e-12) { h = tSched - this.t; hitSched = true; }
    if (h <= 1e-12) {
      if (hitSched) this.onScheduled();
      return;
    }
    const evs = this.stateEvents();
    const g0 = evs.map((e) => e[0](this.y));
    this.yPrev.set(this.y);
    this.stepper.rk4(this.t, this.y, this.ctx, h);
    // check state events
    let first = -1, firstFrac = 2;
    for (let k = 0; k < evs.length; k++) {
      const [fn, dir] = evs[k];
      const g1 = fn(this.y);
      const crossed = dir < 0 ? g0[k] > 0 && g1 <= 0 : dir > 0 ? g0[k] < 0 && g1 >= 0 : g0[k] * g1 < 0;
      if (crossed) {
        const fr = g0[k] / (g0[k] - g1);
        if (fr < firstFrac) { firstFrac = fr; first = k; }
      }
    }
    if (first >= 0) {
      // secant refinement of the crossing time
      const fn = evs[first][0];
      let a = 0, ga = g0[first], b = h, gb = fn(this.y);
      let hc = h * firstFrac;
      const ytmp = new Float64Array(NSTATE);
      for (let it = 0; it < 6; it++) {
        ytmp.set(this.yPrev);
        this.stepper.rk4(this.t, ytmp, this.ctx, hc);
        const gc = fn(ytmp);
        if (Math.abs(gc) < 1e-9 * (1 + Math.abs(ga))) break;
        if ((gc > 0) === (ga > 0)) { a = hc; ga = gc; } else { b = hc; gb = gc; }
        hc = a + (b - a) * (ga / (ga - gb));
      }
      this.y.set(this.yPrev);
      this.stepper.rk4(this.t, this.y, this.ctx, hc);
      this.t += hc;
      this.stepCount++;
      evs[first][2]();
    } else {
      this.t = hitSched ? tSched : this.t + h;
      this.stepCount++;
      if (hitSched) this.onScheduled();
    }
    if (this.t >= this.nextHist - 1e-9) this.record();
  }

  advanceTo(tTarget: number, maxSteps = 200000) {
    let n = 0;
    while (this.t < tTarget - 1e-9 && !this.lost && n++ < maxSteps) this.stepOnce(tTarget);
  }

  private dTmp = new Float64Array(NSTATE);
  telemetry(): Telemetry {
    evaluate(this.t, this.y, this.ctx, this.cfg, this.dTmp, this.tel);
    return this.tel;
  }

  done() {
    return this.ctx.phase === ORBIT || this.ctx.phase === S2_DEPLETED || this.lost;
  }
}

export { COAST_SEP };
