// Web Audio graph for the launch acoustics. Works on any BaseAudioContext (live AudioContext or the
// OfflineAudioContext used for validation). It only realises what AcousticMixer computes:
//
//   rumble_lf A/B ─ xfade ─ layer ┐
//   jet_mix   A/B ─ xfade ─ layer ┴─ jet EQ ──────┐
//   crackle   A/B ─ xfade ─ layer ── crackle EQ ──┼─ airborne (warp) ─┐
//   pad_int.  A/B ─ xfade ─ layer ── pad EQ ──────┘                   │
//   onboard_structure A/B ─ xfade ─ level ────────────────────────────┼─ mix ─ soft limiter ─ master ─ out
//   aero_buffet       A/B ─ xfade ─ level ────────────────────────────┤
//   one-shots ─ own static EQ ─ (airborne | mix) ─────────────────────┘
//
// EQ = gain + 10 high-shelf biquads + a 4th-order low-pass (eqDesign.ts). Loops are AudioBufferSourceNodes; the two voices of a
// layer are offset by 0.618 of the loop and cross-faded on a 37.3 s cycle (profile.anti_repetition). All
// layers of a sync group share read position (from the emission time) and playback rate (Doppler).
import { designShelfEq, LP_Q_DB, newShelfEq, type ShelfEq } from './eqDesign';
import { LAYERS, SILENT_DB, type LayerName, type MixOut, type OneShotFire } from './mixModel';
import { NB, type PreparedProfile } from './profile';

const db2lin = (db: number) => (db <= SILENT_DB + 1 ? 0 : Math.pow(10, db / 20));
const TAU_PARAM = 0.05; // smoothing of control-rate gains [s]
const VOICE_IDLE_S = 1.2; // a layer silent this long releases its voices

/** Python render_launch.soft_limit as a static curve (input range +-4 = +12 dBFS through a 1/4 pre-gain) */
export function softLimitCurve(n = 8193, thrDb = -1, range = 4): Float32Array {
  const t = Math.pow(10, thrDb / 20), k = 0.8 * t;
  const c = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 * range - range;
    const a = Math.abs(x);
    c[i] = a > k ? Math.sign(x) * (k + (t - k) * Math.tanh((a - k) / (t - k))) : x;
  }
  return c;
}

class Tracker {
  alive = 0;
  created = 0;
  add<T extends AudioNode>(n: T): T { this.alive++; this.created++; return n; }
  drop(...ns: (AudioNode | null | undefined)[]) {
    for (const n of ns) { if (!n) continue; try { n.disconnect(); } catch { /* already */ } this.alive--; }
  }
}

class EqChain {
  input: GainNode;
  shelves: BiquadFilterNode[] = [];
  lps: BiquadFilterNode[] = [];
  output: AudioNode;
  eq: ShelfEq;
  resid = 0;
  private lpOn = false;
  /** assetDb: the band level of what feeds this chain (for the don't-care conditioning) */
  constructor(private ctx: BaseAudioContext, private oct: number[], tr: Tracker, private assetDb?: ArrayLike<number>) {
    this.eq = newShelfEq(oct);
    this.input = tr.add(ctx.createGain());
    this.input.gain.value = 0;
    let prev: AudioNode = this.input;
    for (let k = 0; k < NB - 1; k++) {
      const f = tr.add(ctx.createBiquadFilter());
      f.type = 'highshelf';
      f.frequency.value = this.eq.shelfHz[k];
      f.gain.value = 0;
      prev.connect(f);
      prev = f;
      this.shelves.push(f);
    }
    for (const q of LP_Q_DB) {
      const f = tr.add(ctx.createBiquadFilter());
      f.type = 'lowpass';
      f.Q.value = q;
      f.frequency.value = ctx.sampleRate / 2; // parked: flat
      prev.connect(f);
      prev = f;
      this.lps.push(f);
    }
    this.output = prev;
  }
  /** apply a target response (dB per octave band); `lin` multiplies the overall gain; off -> gain 0 */
  set(targetDb: ArrayLike<number>, on: boolean, when: number, tau: number, lin = 1, step = false) {
    if (on) {
      this.resid = designShelfEq(targetDb, this.oct, this.ctx.sampleRate, this.eq, 5, this.assetDb);
      for (let k = 0; k < this.shelves.length; k++) this.shelves[k].gain.setValueAtTime(this.eq.shelfDb[k], when);
      const lp = this.eq.lpHz > 0 ? this.eq.lpHz : this.ctx.sampleRate / 2;
      if (this.eq.lpHz > 0 || this.lpOn) for (const f of this.lps) f.frequency.setValueAtTime(lp, when);
      this.lpOn = this.eq.lpHz > 0;
    }
    const g = on ? Math.pow(10, this.eq.g0 / 20) * lin : 0;
    if (step) this.input.gain.setValueAtTime(g, when);
    else this.input.gain.setTargetAtTime(g, when, tau);
  }
  nodes(): AudioNode[] { return [this.input, ...this.shelves, ...this.lps]; }
}

interface VoicePair { a: AudioBufferSourceNode; b: AudioBufferSourceNode; ga: GainNode; gb: GainNode }

class LoopLayer {
  out: GainNode;
  v: VoicePair | null = null;
  offSince = -1;
  constructor(private ctx: BaseAudioContext, public name: LayerName, public buf: AudioBuffer, private tr: Tracker,
    private arFrac: number, private arCycle: number, public extraOffset = 0) {
    this.out = tr.add(ctx.createGain());
    this.out.gain.value = 0;
  }
  private phi(tau: number) { return Math.PI / 4 + (Math.PI / 4) * Math.sin((2 * Math.PI * tau) / this.arCycle); }

  start(when: number, tauRel: number, rate: number) {
    const L = this.buf.duration;
    const pos = (((tauRel + this.extraOffset) % L) + L) % L;
    const mk = (off: number) => {
      const s = this.tr.add(this.ctx.createBufferSource());
      s.buffer = this.buf;
      s.loop = true;
      s.playbackRate.value = rate;
      const g = this.tr.add(this.ctx.createGain());
      g.gain.value = 0;
      s.connect(g).connect(this.out);
      s.start(when, off % L);
      return [s, g] as const;
    };
    const [a, ga] = mk(pos);
    const [b, gb] = mk(pos + this.arFrac * L);
    const ph = this.phi(tauRel + this.extraOffset);
    ga.gain.setTargetAtTime(Math.cos(ph), when, 0.02);
    gb.gain.setTargetAtTime(Math.sin(ph), when, 0.02);
    this.v = { a, b, ga, gb };
  }

  stop(when: number) {
    const v = this.v;
    if (!v) return;
    this.v = null;
    v.ga.gain.setTargetAtTime(0, when, 0.015);
    v.gb.gain.setTargetAtTime(0, when, 0.015);
    const tEnd = when + 0.12;
    v.a.onended = () => this.tr.drop(v.a, v.ga);
    v.b.onended = () => this.tr.drop(v.b, v.gb);
    try { v.a.stop(tEnd); v.b.stop(tEnd); } catch { this.tr.drop(v.a, v.ga, v.b, v.gb); }
  }

  /** stop immediately (reset / dispose); no scheduling into the future */
  kill() {
    const v = this.v;
    if (!v) return;
    this.v = null;
    v.a.onended = v.b.onended = null;
    try { v.a.stop(); } catch { /* not started */ }
    try { v.b.stop(); } catch { /* not started */ }
    this.tr.drop(v.a, v.ga, v.b, v.gb);
  }

  update(on: boolean, gainLin: number, when: number, tauRel: number, rate: number, gTau: number) {
    this.out.gain.setTargetAtTime(on ? gainLin : 0, when, gTau);
    if (on) {
      this.offSince = -1;
      if (!this.v) this.start(when, tauRel, rate);
      else {
        const ph = this.phi(tauRel + this.extraOffset);
        this.v.ga.gain.setTargetAtTime(Math.cos(ph), when, 0.1);
        this.v.gb.gain.setTargetAtTime(Math.sin(ph), when, 0.1);
        this.v.a.playbackRate.setTargetAtTime(rate, when, 0.1);
        this.v.b.playbackRate.setTargetAtTime(rate, when, 0.1);
      }
    } else if (this.v) {
      if (this.offSince < 0) this.offSince = when;
      else if (when - this.offSince > VOICE_IDLE_S) this.stop(when);
    }
  }
}

interface ActiveShot { asset: string; src: AudioBufferSourceNode; eq: EqChain; g: GainNode; key: string }

export interface EngineStats { loopVoices: number; oneShots: number; nodesAlive: number; nodesCreated: number; eqResidDb: number }

export class AudioEngine {
  readonly ctx: BaseAudioContext;
  master: GainNode;
  private tr = new Tracker();
  private pre: GainNode;
  private shaper: WaveShaperNode;
  private mix: GainNode;
  private airborne: GainNode;
  private jetEq: EqChain;
  private crkEq: EqChain;
  private padEq: EqChain;
  private layers: Record<LayerName, LoopLayer>;
  private shots = new Set<ActiveShot>();
  private listenerKey = '';
  /** emission time that maps to loop read position 0 (render_launch: tau[0] of the telemetry) */
  tau0 = 0;

  constructor(ctx: BaseAudioContext, private prep: PreparedProfile, private buffers: Record<string, AudioBuffer>, destination?: AudioNode) {
    this.ctx = ctx;
    const tr = this.tr;
    this.master = tr.add(ctx.createGain());
    this.master.gain.value = 1;
    this.shaper = tr.add(ctx.createWaveShaper());
    this.shaper.curve = softLimitCurve() as Float32Array<ArrayBuffer>;
    this.shaper.oversample = 'none';
    this.pre = tr.add(ctx.createGain());
    this.pre.gain.value = 0.25;
    this.mix = tr.add(ctx.createGain());
    this.airborne = tr.add(ctx.createGain());
    this.airborne.gain.value = 1;
    this.airborne.connect(this.mix);
    this.mix.connect(this.pre).connect(this.shaper).connect(this.master).connect(destination ?? ctx.destination);
    this.jetEq = new EqChain(ctx, prep.oct, tr, prep.A_jet);
    this.crkEq = new EqChain(ctx, prep.oct, tr, prep.assets.crackle.octave_db);
    this.padEq = new EqChain(ctx, prep.oct, tr, prep.A_pad);
    this.jetEq.output.connect(this.airborne);
    this.crkEq.output.connect(this.airborne);
    this.padEq.output.connect(this.airborne);
    const ar = prep.raw.anti_repetition;
    const mk = (n: LayerName, extra = 0) => new LoopLayer(ctx, n, buffers[n], tr, ar.voice_offset_fraction, ar.crossfade_cycle_s, extra);
    this.layers = {
      rumble_lf: mk('rumble_lf'), jet_mix: mk('jet_mix'), crackle: mk('crackle'), pad_interaction: mk('pad_interaction'),
      onboard_structure: mk('onboard_structure'), aero_buffet: mk('aero_buffet', 7.3), // render_launch reads aero 7.3 s ahead
    };
    this.layers.rumble_lf.out.connect(this.jetEq.input);
    this.layers.jet_mix.out.connect(this.jetEq.input);
    this.layers.crackle.out.connect(this.crkEq.input);
    this.layers.pad_interaction.out.connect(this.padEq.input);
    this.layers.onboard_structure.out.connect(this.mix);
    this.layers.aero_buffet.out.connect(this.mix);
  }

  /** apply one control tick at audio time `when` */
  apply(m: MixOut, when: number) {
    let tSet = when;
    if (m.listenerKey !== this.listenerKey) {
      // camera cut: dip the mix, swap the acoustic situation underneath, come back (no clicks, no stale one-shots)
      const first = this.listenerKey === '';
      this.listenerKey = m.listenerKey;
      if (!first) {
        this.mix.gain.setTargetAtTime(0, when, 0.012);
        tSet = when + 0.07;
        this.mix.gain.setTargetAtTime(1, tSet, 0.03);
        for (const L of LAYERS) this.layers[L].stop(when);
        this.stopOneShots(when);
      }
    }
    const w = m.warp;
    this.airborne.gain.setTargetAtTime(db2lin(w.airborneDb), tSet, w.fadeTau);
    const step = tSet !== when;
    this.jetEq.set(m.jet.eqDb, m.jet.on, tSet, TAU_PARAM, 1, step);
    this.crkEq.set(m.crackle.eqDb, m.crackle.on, tSet, TAU_PARAM, 1, step);
    this.padEq.set(m.pad.eqDb, m.pad.on, tSet, TAU_PARAM, 1, step);
    const t0 = this.tau0;
    const lay = this.layers, mk = w.mask;
    lay.rumble_lf.update(m.jet.on && mk.rumble_lf > 0, mk.rumble_lf, tSet, m.jet.tau - t0, m.jet.rate, w.fadeTau);
    lay.jet_mix.update(m.jet.on && mk.jet_mix > 0, mk.jet_mix, tSet, m.jet.tau - t0, m.jet.rate, w.fadeTau);
    lay.crackle.update(m.crackle.on && mk.crackle > 0, mk.crackle, tSet, m.jet.tau - t0, m.jet.rate, w.fadeTau);
    lay.pad_interaction.update(m.pad.on && mk.pad_interaction > 0, mk.pad_interaction, tSet, m.pad.tau - t0, m.pad.rate, w.fadeTau);
    lay.onboard_structure.update(m.structDb > SILENT_DB && mk.onboard_structure > 0, mk.onboard_structure * db2lin(m.structDb), tSet,
      m.onboardTau - t0, 1, TAU_PARAM);
    lay.aero_buffet.update(m.aeroDb > SILENT_DB && mk.aero_buffet > 0, mk.aero_buffet * db2lin(m.aeroDb), tSet, m.onboardTau - t0, 1, TAU_PARAM);
  }

  /** silence everything (idle on the pad, vacuum, before launch) but keep the graph */
  silence(when: number) {
    for (const e of [this.jetEq, this.crkEq, this.padEq]) e.input.gain.setTargetAtTime(0, when, TAU_PARAM);
    for (const L of LAYERS) {
      const l = this.layers[L];
      l.out.gain.setTargetAtTime(0, when, TAU_PARAM);
      if (l.v) { if (l.offSince < 0) l.offSince = when; else if (when - l.offSince > VOICE_IDLE_S) l.stop(when); }
    }
  }

  /** time discontinuity (replay seek, restart): restart every running loop at its new read position */
  resync(when: number, m: MixOut) {
    for (const L of LAYERS) this.layers[L].stop(when);
    this.stopOneShots(when);
    this.apply(m, when);
  }

  durationOf = (asset: string) => this.buffers[asset]?.duration ?? 0;
  isPlaying = (asset: string) => { for (const s of this.shots) if (s.asset === asset) return true; return false; };

  /** start a one-shot. `delayWall` is in audio seconds from `now` (>= 0), offset in seconds into the asset */
  fireOneShot(f: OneShotFire, now: number, delayWall: number) {
    const buf = this.buffers[f.plan.asset];
    if (!buf) return;
    const tr = this.tr, ctx = this.ctx;
    const src = tr.add(ctx.createBufferSource());
    src.buffer = buf;
    const eq = new EqChain(ctx, this.prep.oct, tr, this.prep.assets[f.plan.asset]?.octave_db);
    const g = tr.add(ctx.createGain());
    const when = now + Math.max(0, delayWall);
    if (f.offset > 0) { g.gain.value = 0; g.gain.setTargetAtTime(1, when, 0.006); } // joined mid-way: no click
    else g.gain.value = 1;
    eq.set(f.plan.eqDb, true, now, TAU_PARAM, 1, true);
    src.connect(eq.input);
    eq.output.connect(g);
    const structural = f.plan.asset === 'stage_sep_structural' || f.plan.key.includes(':struct');
    g.connect(structural ? this.mix : this.airborne);
    const shot: ActiveShot = { asset: f.plan.asset, src, eq, g, key: f.plan.key };
    this.shots.add(shot);
    src.onended = () => this.releaseShot(shot);
    src.start(when, Math.min(f.offset, buf.duration - 1e-3));
  }

  private releaseShot(s: ActiveShot) {
    if (!this.shots.delete(s)) return;
    s.src.onended = null;
    this.tr.drop(s.src, ...s.eq.nodes(), s.g);
  }

  stopOneShots(when: number) {
    for (const s of this.shots) {
      s.g.gain.setTargetAtTime(0, when, 0.015);
      try { s.src.stop(when + 0.1); } catch { this.releaseShot(s); }
    }
  }

  /** hard stop of every voice and one-shot (reset / dispose) */
  stopAll() {
    for (const L of LAYERS) this.layers[L].kill();
    for (const s of [...this.shots]) { try { s.src.stop(); } catch { /* */ } this.releaseShot(s); }
    const now = this.ctx.currentTime;
    for (const e of [this.jetEq, this.crkEq, this.padEq]) { e.input.gain.cancelScheduledValues(now); e.input.gain.setValueAtTime(0, now); }
    for (const L of LAYERS) { const g = this.layers[L].out.gain; g.cancelScheduledValues(now); g.setValueAtTime(0, now); }
    this.mix.gain.cancelScheduledValues(now);
    this.mix.gain.setValueAtTime(1, now);
    this.listenerKey = '';
  }

  dispose() {
    this.stopAll();
    const all: AudioNode[] = [this.master, this.shaper, this.pre, this.mix, this.airborne, ...this.jetEq.nodes(), ...this.crkEq.nodes(), ...this.padEq.nodes()];
    for (const L of LAYERS) all.push(this.layers[L].out);
    this.tr.drop(...all);
  }

  stats(): EngineStats {
    let v = 0;
    for (const L of LAYERS) if (this.layers[L].v) v += 2;
    return { loopVoices: v, oneShots: this.shots.size, nodesAlive: this.tr.alive, nodesCreated: this.tr.created, eqResidDb: Math.max(this.jetEq.resid, this.padEq.resid) };
  }
}
