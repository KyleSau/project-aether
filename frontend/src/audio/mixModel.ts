// Per-tick acoustic model: telemetry history + listener -> per-group octave EQ targets (dB), voice read
// rates/positions, onboard levels, warp masks, and one-shot plans. Pure (no Web Audio): the same code drives
// the live AudioEngine, the OfflineAudioContext validation render and the unit tests.
//
// Mirrors tools/audio/render_launch.py (model_grid + render), evaluated at the emission time of the
// current reception time instead of on a precomputed grid.
import { atmosphere } from '../physics/atmosphere';
import type { VehicleConfig } from '../physics/config';
import { engineThrust } from '../physics/dynamics';
import {
  apparentSource, bandSpl, cabinTlDb, cPath, crackleWeight, listenerImpedanceDb, newFrame, oaDb, onboardAirWeight,
  aeroWeight, padWeight, Ray, sourceState, type SourceFrame, type SourceState, type V3,
} from './acoustics';
import type { Listener } from './listeners';
import { NB, type PreparedProfile } from './profile';
import { TelemetryRing, solveEmissionTime } from './telemetryRing';

export type LayerName = 'rumble_lf' | 'jet_mix' | 'crackle' | 'pad_interaction' | 'onboard_structure' | 'aero_buffet';
export const LAYERS: LayerName[] = ['rumble_lf', 'jet_mix', 'crackle', 'pad_interaction', 'onboard_structure', 'aero_buffet'];
export const SILENT_DB = -200;
/** a group whose loudest OUTPUT band (asset band level + EQ) is below this many dBFS is inaudible (below the
 *  16-bit floor at any volume setting): it is reported off so its voices stop */
export const INAUDIBLE_DBFS = -110;
const audible = (eq: ArrayLike<number>, A: ArrayLike<number>) => {
  for (let i = 0; i < eq.length; i++) if (eq[i] + A[i] > INAUDIBLE_DBFS) return true;
  return false;
};

// ------------------------------------------------------------------ time-warp policy (profile.time_warp)
export interface WarpPolicy {
  airborneDb: number; // gain on the external (airborne) groups
  fadeTau: number; // time constant for reaching airborneDb [s]
  mask: Record<LayerName, number>;
  oneShots: 'all' | 'no-overlap' | 'none';
  doppler: boolean;
}

export function warpPolicy(warp: number): WarpPolicy {
  const all: Record<LayerName, number> = { rumble_lf: 1, jet_mix: 1, crackle: 1, pad_interaction: 1, onboard_structure: 1, aero_buffet: 1 };
  if (warp <= 1 + 1e-9) return { airborneDb: 0, fadeTau: 0.1, mask: all, oneShots: 'all', doppler: true };
  if (warp <= 4 + 1e-9) return { airborneDb: (-6 * Math.log2(warp)) / 2, fadeTau: 0.1, mask: all, oneShots: 'no-overlap', doppler: true };
  // time-lapse ambience: rumble + structure only, airborne at -24 dB reached over ~1 s
  return {
    airborneDb: -24, fadeTau: 1 / 3, oneShots: 'none', doppler: false,
    mask: { rumble_lf: 1, jet_mix: 0, crackle: 0, pad_interaction: 0, onboard_structure: 1, aero_buffet: 0 },
  };
}

// ------------------------------------------------------------------ outputs
export interface GroupOut { on: boolean; eqDb: Float64Array }
export interface MixOut {
  t: number;
  listenerKey: string;
  jet: GroupOut & { tau: number; rate: number };
  crackle: GroupOut;
  pad: GroupOut & { tau: number; rate: number };
  /** flat gains (dB) for the structure-borne / aero loops; SILENT_DB = off */
  structDb: number;
  aeroDb: number;
  onboardTau: number;
  warp: WarpPolicy;
  dbg: { te: number; delay: number; r: number; theta: number; wc: number; wpad: number; wair: number; LpOA: number; LpMax: number; c: number };
}

export interface OneShotPlan {
  key: string;
  asset: 'ignition' | 'meco_tail' | 'stage_sep_structural';
  tArrive: number; // sim time at the listener
  eqDb: Float64Array; // static octave EQ (dB)
}

export interface SimEventLike { name: string; t: number }

const PAD_SRC: V3 = [0, 60, 1]; // flame-trench exit: trench_dir (0, 1) x 60 m, as render_launch.model_grid
const ONBOARD_FWD = 55; // listener 55 m forward of the active nozzle (render_launch.py)

export function liftoffFrame(veh: VehicleConfig): SourceFrame {
  const g = atmosphere(0);
  const [F, mdot] = engineThrust(veh.engines.stage1, 1, g.P);
  const f = newFrame();
  Object.assign(f, { t: 0, phase: 1, thrust: F, mdot, throttle: 1, P: g.P, T: g.T, rho: g.rho, a: g.a, h: 0, mach: 0, qbar: 0 });
  return f;
}

export class AcousticMixer {
  ring = new TelemetryRing(8192); // 819 s at the sim history rate (0.1 s): MECO reaches the pad listener ~440 s late
  listener: Listener;
  k: number;
  lpMaxOverride: number | null = null;
  /** playback-rate clamp of the delay-line read (profile doppler_policy: [0.5, 1.5]) */
  rateClamp: [number, number] = [0.5, 1.5];
  private fr = newFrame();
  private frB = newFrame();
  private ray = new Ray(24);
  private L = new Float64Array(NB);
  private Lp = new Float64Array(NB);
  private A = new Float64Array(NB);
  private Aeff = new Float64Array(NB);
  private fEff: Float64Array;
  private crkMask: boolean[];
  private teJet = NaN;
  private cJet = 340;
  private padCache: { key: string; r: number; c: number; Aabs: Float64Array; imp: number } | null = null;
  private refCache = new Map<string, { LpMax: number; ignEq: Float64Array }>();
  private out: MixOut;

  constructor(public prep: PreparedProfile, public veh: VehicleConfig, listener: Listener, k?: number) {
    this.listener = listener;
    this.k = k ?? prep.raw.level_mapping.k_default;
    const cb = prep.raw.propagation.atmospheric_absorption.crackle_band_policy;
    this.fEff = Float64Array.from(prep.oct, (f) => (f > cb.evaluate_at_hz ? cb.evaluate_at_hz * Math.pow(f / cb.evaluate_at_hz, cb.exponent) : f));
    this.crkMask = prep.oct.map((f) => f > cb.evaluate_at_hz);
    const g = (): GroupOut => ({ on: false, eqDb: new Float64Array(NB).fill(SILENT_DB) });
    this.out = {
      t: 0, listenerKey: listener.key, jet: { ...g(), tau: 0, rate: 1 }, crackle: g(), pad: { ...g(), tau: 0, rate: 1 },
      structDb: SILENT_DB, aeroDb: SILENT_DB, onboardTau: 0, warp: warpPolicy(1),
      dbg: { te: NaN, delay: 0, r: 0, theta: 0, wc: 0, wpad: 0, wair: 0, LpOA: -Infinity, LpMax: 0, c: 340 },
    };
  }

  setListener(l: Listener) {
    if (l.key !== this.listener.key) { this.teJet = NaN; this.padCache = null; }
    this.listener = l;
  }

  reset() {
    this.ring.clear();
    this.teJet = NaN;
    this.padCache = null;
  }

  /** digital target band levels: D = Lp - Lp_OA + target + k (Lp_OA - Lp_max) */
  private toDigital(L: ArrayLike<number>, LpMax: number, out: Float64Array) {
    const lm = this.prep.raw.level_mapping;
    const oa = oaDb(L);
    for (let b = 0; b < NB; b++) out[b] = L[b] - oa + lm.target_peak_dbfs + this.k * (oa - LpMax);
    return out;
  }

  private pad(lis: V3) {
    const key = `${this.listener.key}:${lis[0].toFixed(1)},${lis[1].toFixed(1)},${lis[2].toFixed(1)}`;
    if (!this.padCache || this.padCache.key !== key) {
      const r = Math.hypot(PAD_SRC[0] - lis[0], PAD_SRC[1] - lis[1], PAD_SRC[2] - lis[2]);
      const ray = new Ray(24).set(PAD_SRC, lis, this.prep);
      const Aabs = ray.absorption(this.prep.oct, new Float64Array(NB));
      this.padCache = { key, r, c: cPath(PAD_SRC, lis, this.prep), Aabs, imp: listenerImpedanceDb(lis[2], this.prep) };
    }
    return this.padCache;
  }

  /** pad-interaction band SPL (dB) for the source state `st` at nozzle height hz; returns w_pad (0 = none) */
  private padSpl(st: SourceState, phase: number, hz: number, lis: V3, out: Float64Array): number {
    const w_pad = phase <= 2 ? padWeight(hz, st.De) : 0;
    if (w_pad <= 1e-4) { out.fill(SILENT_DB); return 0; }
    const pc = this.pad(lis);
    const rEff = Math.sqrt(pc.r * pc.r + st.xt * st.xt);
    const base = st.Lw + this.prep.raw.pad_interaction.level_db_re_jet - 10 * Math.log10(2 * Math.PI * rEff * rEff) + pc.imp + 10 * Math.log10(w_pad);
    for (let b = 0; b < NB; b++) out[b] = base + this.prep.padShape[b] - pc.Aabs[b];
    return w_pad;
  }

  /** jet band SPL at `lis` incl. the pad's (1 - 0.5 w_pad) reduction of the free jet */
  private jetSpl(st: SourceState, f: SourceFrame, lis: V3, out: Float64Array, ground: boolean, rO?: number, thO?: number) {
    const g = bandSpl(this.prep, st, f.noz, f.ax, lis, out, { ground, absorb: true, rOverride: rO, thetaOverride: thO }, this.ray);
    const w_pad = f.phase <= 2 ? padWeight(Math.max(f.noz[2], 0), st.De) : 0;
    const js = 20 * Math.log10(Math.max(1 - 0.5 * w_pad, 1e-3));
    for (let b = 0; b < NB; b++) out[b] += js;
    return g;
  }

  /** mic-gain reference and ignition EQ for the current listener, from a synthetic full-thrust liftoff state */
  private reference(): { LpMax: number; ignEq: Float64Array } {
    const l = this.listener;
    const key = `${l.key}|${l.mode}|${this.k}|${this.lpMaxOverride}`;
    const hit = this.refCache.get(key);
    if (hit) return hit;
    const f = liftoffFrame(this.veh);
    const st = sourceState(this.prep, this.veh, f)!;
    const L = new Float64Array(NB), Lp = new Float64Array(NB), D = new Float64Array(NB);
    let LpMax: number;
    const ign = new Float64Array(NB);
    if (l.mode === 'ground') {
      this.jetSpl(st, f, l.pos, L, true);
      LpMax = this.lpMaxOverride ?? l.LpMax ?? oaDb(L);
      this.toDigital(L, LpMax, D);
      for (let b = 0; b < NB; b++) ign[b] = D[b] - this.prep.A_jet[b] - 2;
    } else {
      const onboard = l.mode === 'onboard' || l.mode === 'cabin';
      const lis: V3 = onboard ? [0, 0, ONBOARD_FWD] : [l.chaseR ?? 30, 0, 1.7];
      this.jetSpl(st, f, lis, L, false, l.chaseR ?? (onboard ? undefined : 30), l.chaseTheta ?? (onboard ? undefined : 100));
      const wAirDb = 10 * Math.log10(Math.max(onboardAirWeight(0, f.rho, f.a), 1e-12));
      const tl = (b: number) => (l.mode === 'cabin' ? cabinTlDb(this.prep.oct[b], this.prep.raw.onboard.cabin_transmission_loss.m_s_kg_m2) : 0);
      for (let b = 0; b < NB; b++) L[b] += wAirDb - tl(b);
      if (onboard) {
        this.padSpl(st, 1, 0, lis, Lp);
        for (let b = 0; b < NB; b++) Lp[b] += wAirDb - tl(b);
        const tot = new Float64Array(NB);
        for (let b = 0; b < NB; b++) tot[b] = 10 * Math.log10(Math.pow(10, L[b] / 10) + Math.pow(10, Lp[b] / 10));
        LpMax = this.lpMaxOverride ?? oaDb(tot);
      } else {
        LpMax = this.lpMaxOverride ?? oaDb(L); // chase: Engines reference r = 30 m, theta = 100 deg
      }
      this.toDigital(L, LpMax, D);
      for (let b = 0; b < NB; b++) ign[b] = D[b] - this.prep.A_jet[b] - 2;
    }
    const r = { LpMax, ignEq: ign };
    this.refCache.set(key, r);
    return r;
  }

  get LpMax() { return this.reference().LpMax; }

  /** evaluate the model at reception (sim) time t */
  compute(t: number, warp: number): MixOut {
    const o = this.out, p = this.prep, l = this.listener, ring = this.ring;
    o.t = t;
    o.listenerKey = l.key;
    o.warp = warpPolicy(warp);
    o.jet.on = o.crackle.on = o.pad.on = false;
    o.structDb = o.aeroDb = SILENT_DB;
    o.onboardTau = t;
    o.jet.rate = o.pad.rate = 1;
    const ref = this.reference();
    o.dbg.LpMax = ref.LpMax;
    o.dbg.wc = o.dbg.wpad = o.dbg.wair = 0;
    o.dbg.LpOA = -Infinity;
    if (!ring.size || l.mode === 'vacuum') { o.jet.tau = o.pad.tau = t; return o; }
    if (l.mode === 'ground') this.computeGround(t, ref.LpMax);
    else this.computeAttached(t, ref.LpMax);
    return o;
  }

  private rAt = (te: number) => {
    const f = this.frB;
    this.ring.sample(Math.max(te, this.ring.t0), f);
    const st = sourceState(this.prep, this.veh, f);
    const s = apparentSource(st, f.noz, f.ax);
    const lis = this.listener.pos;
    return Math.hypot(s[0] - lis[0], s[1] - lis[1], s[2] - lis[2]);
  };

  private computeGround(t: number, LpMax: number) {
    const o = this.out, p = this.prep, lis = this.listener.pos, ring = this.ring, f = this.fr;
    // ---- free jet: emission time (variable delay line)
    let c = this.cJet;
    let te = solveEmissionTime(t, this.teJet, c, this.rAt, ring.t0);
    // c_path depends (weakly) on the ray, i.e. on t_e: iterate to self-consistency so the result does not
    // depend on the previous tick (replay / seek determinism)
    for (let it = 0; it < 4 && !isNaN(te); it++) {
      ring.sample(te, f);
      const st0 = sourceState(p, this.veh, f);
      const c2 = cPath(apparentSource(st0, f.noz, f.ax), lis, p);
      if (Math.abs(c2 - c) <= 1e-6 * c) break;
      c = c2;
      te = solveEmissionTime(t, te, c, this.rAt, ring.t0);
    }
    this.cJet = c;
    this.teJet = te;
    o.jet.tau = isNaN(te) ? t - this.rAt(t) / c : te;
    o.dbg.te = te;
    o.dbg.c = c;
    if (!isNaN(te)) {
      ring.sample(te, f);
      const st = sourceState(p, this.veh, f);
      const h = 0.05;
      const drdt = (this.rAt(te + h) - this.rAt(te - h)) / (2 * h);
      o.jet.rate = o.warp.doppler ? Math.min(this.rateClamp[1], Math.max(this.rateClamp[0], 1 / (1 + drdt / c))) : 1;
      if (st) {
        const L = this.L;
        const g = this.jetSpl(st, f, lis, L, true);
        o.dbg.delay = g.delay; o.dbg.r = g.r; o.dbg.theta = g.theta;
        o.dbg.LpOA = oaDb(L);
        if (L[0] >= -150) {
          this.toDigital(L, LpMax, o.jet.eqDb);
          for (let b = 0; b < NB; b++) o.jet.eqDb[b] -= p.A_jet[b];
          o.jet.on = true;
          // crackle: weight x jet EQ, plus the weak-shock absorption policy (absorbed as if at 250 (f/250)^0.2)
          const wc = crackleWeight(p, st.w_crackle, g.theta, g.r, st.xt);
          o.dbg.wc = wc;
          this.ray.absorption(p.oct, this.A);
          this.ray.absorption(this.fEff, this.Aeff);
          const fr = Math.min(1, p.raw.propagation.atmospheric_absorption.crackle_band_policy.nonlinear_range_m / Math.max(g.r, 1));
          const wdb = 20 * Math.log10(Math.max(wc, 1e-6));
          for (let b = 0; b < NB; b++) o.crackle.eqDb[b] = o.jet.eqDb[b] + wdb + (this.crkMask[b] ? (this.A[b] - this.Aeff[b]) * fr : 0);
          o.crackle.on = wc > 1e-5 && audible(o.crackle.eqDb, p.assets.crackle.octave_db);
          o.jet.on = audible(o.jet.eqDb, p.A_jet);
        }
      }
    }
    // ---- pad interaction: static source at the trench exit, its own (constant) delay
    const pc = this.pad(lis);
    const tp = t - pc.r / pc.c;
    o.pad.tau = tp;
    if (tp >= ring.t0) {
      ring.sample(tp, f);
      const st = sourceState(p, this.veh, f);
      if (st) {
        const w = this.padSpl(st, f.phase, Math.max(f.noz[2], 0), lis, this.Lp);
        o.dbg.wpad = w;
        if (w > 0) {
          this.toDigital(this.Lp, LpMax, o.pad.eqDb);
          for (let b = 0; b < NB; b++) o.pad.eqDb[b] -= p.A_pad[b];
          o.pad.on = audible(o.pad.eqDb, p.A_pad);
        }
      }
    }
  }

  /** onboard / cabin / chase: the listener rides with the vehicle -> no delay, no Doppler */
  private computeAttached(t: number, LpMax: number) {
    const o = this.out, p = this.prep, l = this.listener, f = this.fr;
    const tt = Math.min(t, this.ring.tLast);
    if (t < this.ring.t0) return;
    this.ring.sample(tt, f);
    o.jet.tau = o.pad.tau = t;
    const onboard = l.mode === 'onboard' || l.mode === 'cabin';
    const lis: V3 = onboard ? [f.noz[0] - f.ax[0] * ONBOARD_FWD, f.noz[1] - f.ax[1] * ONBOARD_FWD, f.noz[2] - f.ax[2] * ONBOARD_FWD] : l.pos;
    const st = sourceState(p, this.veh, f);
    const wAir = onboardAirWeight(f.mach, f.rho, f.a);
    o.dbg.wair = wAir;
    const wAirDb = 10 * Math.log10(Math.max(wAir, 1e-12));
    const cabin = l.mode === 'cabin';
    const ms = p.raw.onboard.cabin_transmission_loss.m_s_kg_m2;
    if (st) {
      const L = this.L;
      const g = this.jetSpl(st, f, lis, L, false, l.chaseR, l.chaseTheta);
      o.dbg.r = g.r; o.dbg.theta = g.theta;
      for (let b = 0; b < NB; b++) L[b] += wAirDb - (cabin ? cabinTlDb(p.oct[b], ms) : 0);
      const w = this.padSpl(st, f.phase, Math.max(f.noz[2], 0), lis, this.Lp);
      o.dbg.wpad = w;
      if (w > 0) for (let b = 0; b < NB; b++) this.Lp[b] += wAirDb - (cabin ? cabinTlDb(p.oct[b], ms) : 0);
      // chase: cap Lp at the preset maximum
      let cap = 0;
      if (l.mode === 'chase') {
        const tot = 10 * Math.log10(Math.pow(10, oaDb(L) / 10) + (w > 0 ? Math.pow(10, oaDb(this.Lp) / 10) : 0));
        cap = Math.min(0, LpMax - tot);
      }
      o.dbg.LpOA = oaDb(L) + cap;
      if (L[0] + cap >= -150) {
        for (let b = 0; b < NB; b++) L[b] += cap;
        this.toDigital(L, LpMax, o.jet.eqDb);
        for (let b = 0; b < NB; b++) o.jet.eqDb[b] -= p.A_jet[b];
        o.jet.on = true;
        const wc = crackleWeight(p, st.w_crackle, g.theta, g.r, st.xt);
        o.dbg.wc = wc;
        const wdb = 20 * Math.log10(Math.max(wc, 1e-6));
        for (let b = 0; b < NB; b++) o.crackle.eqDb[b] = o.jet.eqDb[b] + wdb;
        o.crackle.on = wc > 1e-5 && audible(o.crackle.eqDb, p.assets.crackle.octave_db);
        o.jet.on = audible(o.jet.eqDb, p.A_jet);
      }
      if (w > 0 && this.Lp[0] + cap >= -150) {
        for (let b = 0; b < NB; b++) this.Lp[b] += cap;
        this.toDigital(this.Lp, LpMax, o.pad.eqDb);
        for (let b = 0; b < NB; b++) o.pad.eqDb[b] -= p.A_pad[b];
        o.pad.on = audible(o.pad.eqDb, p.A_pad);
      }
    }
    if (onboard) {
      // structure-borne and aero buffet: ARTISTIC reference levels, flat EQ (shape = the asset's own)
      const dl = p.raw.onboard.digital_reference_levels_dbfs[cabin ? 'cabin' : 'onboard'];
      const sdb = 20 * Math.log10(Math.max(f.thrust, 1) / p.raw.onboard.structure_borne.thrust_ref_stage1_N);
      o.structDb = dl.structure_at_stage1_full_thrust + sdb - p.oa.onboard_structure;
      o.aeroDb = dl.aero_at_w_aero_1 + 20 * Math.log10(Math.max(aeroWeight(f.qbar, f.mach), 1e-6)) - p.oa.aero_buffet;
      // below -100 dB re the reference the loop is inaudible: report it as off so its voices can stop
      if (f.thrust <= 1 || o.structDb < -100) o.structDb = SILENT_DB;
      if (o.aeroDb < -100) o.aeroDb = SILENT_DB;
    }
  }

  /** one-shot plans for an event, for the current listener (empty when the event is silent here) */
  planOneShots(ev: SimEventLike): OneShotPlan[] {
    const l = this.listener, p = this.prep, out: OneShotPlan[] = [];
    if (l.mode === 'vacuum') return out;
    const ref = this.reference();
    const key = (a: string, v = '') => `${ev.name}@${ev.t.toFixed(3)}:${a}${v}:${l.key}`;
    const onboard = l.mode === 'onboard' || l.mode === 'cabin';
    const tl = (b: number) => cabinTlDb(p.oct[b], p.raw.onboard.cabin_transmission_loss.m_s_kg_m2);
    if (ev.name === 'IGNITION') {
      // emitted at the pad (trench exit for the IOP), at the full-thrust jet level - 2 dB
      const delay = l.mode === 'ground' ? this.pad(l.pos).r / this.pad(l.pos).c : 0;
      const eq = Float64Array.from(ref.ignEq);
      if (l.mode === 'cabin') for (let b = 0; b < NB; b++) eq[b] -= tl(b);
      out.push({ key: key('ignition'), asset: 'ignition', tArrive: ev.t + delay, eqDb: eq });
    } else if (ev.name === 'MECO') {
      const f = newFrame();
      if (!this.ring.size || !this.ring.sample(Math.max(ev.t - 0.1, this.ring.t0), f)) return out;
      const st = sourceState(p, this.veh, f);
      if (!st) return out;
      const L = new Float64Array(NB), D = new Float64Array(NB);
      if (l.mode === 'ground') {
        const g = this.jetSpl(st, f, l.pos, L, true);
        this.toDigital(L, ref.LpMax, D);
        for (let b = 0; b < NB; b++) D[b] -= p.A_jet[b];
        out.push({ key: key('meco_tail'), asset: 'meco_tail', tArrive: ev.t + g.delay, eqDb: D });
      } else {
        const lis: V3 = onboard ? [f.noz[0] - f.ax[0] * ONBOARD_FWD, f.noz[1] - f.ax[1] * ONBOARD_FWD, f.noz[2] - f.ax[2] * ONBOARD_FWD] : l.pos;
        this.jetSpl(st, f, lis, L, false, l.chaseR, l.chaseTheta);
        const wAirDb = 10 * Math.log10(Math.max(onboardAirWeight(f.mach, f.rho, f.a), 1e-12));
        for (let b = 0; b < NB; b++) L[b] += wAirDb;
        if (L[0] > -150) {
          this.toDigital(L, ref.LpMax, D);
          for (let b = 0; b < NB; b++) D[b] -= p.A_jet[b];
          if (audible(D, p.A_jet)) out.push({ key: key('meco_tail', ':air'), asset: 'meco_tail', tArrive: ev.t, eqDb: D });
        }
        if (onboard) {
          const dl = p.raw.onboard.digital_reference_levels_dbfs[l.mode === 'cabin' ? 'cabin' : 'onboard'];
          const sdb = 20 * Math.log10(Math.max(f.thrust, 1) / p.raw.onboard.structure_borne.thrust_ref_stage1_N);
          const eq = new Float64Array(NB);
          for (let b = 0; b < NB; b++) eq[b] = dl.structure_at_stage1_full_thrust + sdb - 6 - p.oa.meco_tail - tl(b) * 0.3;
          out.push({ key: key('meco_tail', ':struct'), asset: 'meco_tail', tArrive: ev.t, eqDb: eq });
        }
      }
    } else if (ev.name === 'STAGE_SEP' && onboard) {
      const dl = p.raw.onboard.digital_reference_levels_dbfs[l.mode === 'cabin' ? 'cabin' : 'onboard'];
      const eq = new Float64Array(NB).fill(dl.stage_sep - p.oa.stage_sep_structural);
      out.push({ key: key('stage_sep_structural'), asset: 'stage_sep_structural', tArrive: ev.t, eqDb: eq });
    }
    return out;
  }
}

// ------------------------------------------------------------------ one-shot scheduling
const LATE_TOLERANCE_S = 0.15;
export interface OneShotFire { plan: OneShotPlan; delay: number /* sim s from now; <0 = start late */; offset: number /* s into the asset */ }

/**
 * Decides which one-shots start this tick. Each plan fires once per listener; a one-shot whose arrival
 * already passed (seek in replay, camera switch) starts part-way through if it is still sounding.
 */
export class OneShotScheduler {
  private done = new Set<string>();
  reset() { this.done.clear(); }
  update(t: number, plans: OneShotPlan[], policy: WarpPolicy, lookahead: number, durationOf: (a: string) => number,
    isPlaying: (a: string) => boolean): OneShotFire[] {
    const fires: OneShotFire[] = [];
    for (const pl of plans) {
      if (this.done.has(pl.key)) continue;
      const dt = pl.tArrive - t;
      if (dt > lookahead) continue;
      this.done.add(pl.key);
      let late = Math.max(0, -dt);
      if (late >= durationOf(pl.asset)) continue;
      // events are only seen at the first control tick after they happen: a one-tick lateness is absorbed
      // (start from the beginning, <= 0.15 s late) rather than cutting into the transient
      if (late <= LATE_TOLERANCE_S) late = 0;
      if (policy.oneShots === 'none') continue;
      if (policy.oneShots === 'no-overlap' && isPlaying(pl.asset)) continue;
      fires.push({ plan: pl, delay: Math.max(dt, 0), offset: late });
    }
    return fires;
  }
}
