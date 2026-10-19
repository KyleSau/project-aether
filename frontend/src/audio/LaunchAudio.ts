// Browser driver: unlocks audio on the first user gesture, loads the profile and assets, and runs the
// acoustic model at 30 Hz from a timer (outside the three.js render loop). Sim clock, pause, warp,
// replay, seek and restart are read from the MissionController; the audio clock never drives the sim.
import type { Engine } from '../engine/Engine';
import { GROUND_Y } from '../engine/scene/Launchpad';
import type { V3 } from './acoustics';
import { AudioEngine, type EngineStats } from './AudioEngine';
import { loadAssets, opusLikelySupported } from './assets';
import { listenerForCamera, type AetherCamera } from './listeners';
import { AcousticMixer, OneShotScheduler, type OneShotPlan } from './mixModel';
import { prepareProfile, type AcousticProfile, type PreparedProfile } from './profile';
import { SimFeed } from './simFeed';

export type AudioStatus = 'locked' | 'loading' | 'ready' | 'error' | 'unsupported';
const TICK_MS = 1000 / 30;
const BASE = './data/audio/';
const LS_KEY = 'aether.audio';

export class LaunchAudio {
  status: AudioStatus = 'locked';
  muted = false;
  volume = 0.8;
  ctx: AudioContext | null = null;
  engine: AudioEngine | null = null;
  mixer: AcousticMixer | null = null;
  info: { format: string; bytes: number; decodedMB: number; sampleRate: number; tickUs: number; error?: string } =
    { format: '', bytes: 0, decodedMB: 0, sampleRate: 0, tickUs: 0 };
  private app: Engine | null = null;
  private prep: PreparedProfile | null = null;
  private feed: SimFeed | null = null;
  private sched = new OneShotScheduler();
  private plans = new Map<string, OneShotPlan[]>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastT = NaN;
  private lastWall = 0;
  private running = true; // desired context state (false while paused / hidden)
  private ctxBusy: Promise<void> | null = null;
  private profileP: Promise<AcousticProfile> | null = null;
  private loadP: Promise<void> | null = null;
  private tickUsAcc = 0;
  private tickN = 0;

  constructor() {
    try {
      const s = JSON.parse(localStorage.getItem(LS_KEY) ?? 'null');
      if (s && typeof s.volume === 'number') { this.volume = Math.min(1, Math.max(0, s.volume)); this.muted = !!s.muted; }
    } catch { /* storage unavailable */ }
  }

  /** wire to the app; audio stays locked until the first user gesture */
  attach(app: Engine) {
    this.app = app;
    this.profileP = fetch(BASE + 'acoustic_profile.json').then((r) => {
      if (!r.ok) throw new Error(`acoustic profile: HTTP ${r.status}`);
      return r.json();
    });
    this.profileP.catch(() => { /* reported on unlock */ });
    const gesture = () => { this.unlock(); };
    for (const ev of ['pointerdown', 'keydown', 'touchend'] as const) window.addEventListener(ev, gesture, { passive: true });
    document.addEventListener('visibilitychange', () => this.reconcileRunning());
  }

  /** must be called from a user gesture (first one creates and resumes the AudioContext) */
  unlock() {
    if (this.ctx) {
      if (this.running && this.ctx.state !== 'running' && this.ctx.state !== 'closed') void this.ctx.resume().catch(() => {});
      return;
    }
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) { this.status = 'unsupported'; return; }
    try { this.ctx = new Ctx({ sampleRate: 48000, latencyHint: 'playback' }); } catch {
      try { this.ctx = new Ctx(); } catch { this.status = 'unsupported'; return; }
    }
    void this.ctx.resume().catch(() => {});
    this.status = 'loading';
    this.loadP = this.load().catch((e) => { this.status = 'error'; this.info.error = String(e?.message ?? e); console.warn('launch audio disabled:', e); });
  }

  private async load() {
    const ctx = this.ctx!;
    const raw = await this.profileP!;
    const prep = prepareProfile(raw);
    this.prep = prep;
    const fetchBytes = async (p: string) => {
      const r = await fetch(p);
      if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
      return r.arrayBuffer();
    };
    const assets = await loadAssets(ctx, prep, fetchBytes, BASE, { preferOpus: opusLikelySupported() });
    const fmts = new Set(Object.values(assets.format));
    this.info = { ...this.info, format: [...fmts].join('+'), bytes: assets.bytes, decodedMB: assets.decodedBytes / 1048576, sampleRate: ctx.sampleRate };
    this.engine = new AudioEngine(ctx, prep, assets.buffers);
    this.applyVolume(0);
    const app = this.app!;
    this.mixer = new AcousticMixer(prep, app.cfg, listenerForCamera(prep, 'pad', [310, -260, 1.7]));
    this.feed = new SimFeed(app.cfg);
    this.status = 'ready';
    this.lastWall = performance.now();
    this.timer = setInterval(() => this.tick(), TICK_MS);
  }

  setVolume(v: number) { this.volume = Math.min(1, Math.max(0, v)); this.applyVolume(); this.persist(); }
  setMuted(m: boolean) { this.muted = m; this.applyVolume(); this.persist(); }
  private persist() { try { localStorage.setItem(LS_KEY, JSON.stringify({ volume: this.volume, muted: this.muted })); } catch { /* */ } }
  private applyVolume(tau = 0.03) {
    if (!this.engine || !this.ctx) return;
    const g = this.muted || !this.running ? 0 : this.volume * this.volume;
    const p = this.engine.master.gain;
    if (tau <= 0) { p.cancelScheduledValues(this.ctx.currentTime); p.setValueAtTime(g, this.ctx.currentTime); }
    else p.setTargetAtTime(g, this.ctx.currentTime, tau);
  }

  /** restart / reset: drop every voice and all history; nothing keeps sounding */
  reset() {
    this.engine?.stopAll();
    this.mixer?.reset();
    this.sched.reset();
    this.plans.clear();
    this.lastT = NaN;
  }

  /** pause / tab hidden -> fade out then suspend the context; resume -> resume then fade in */
  private reconcileRunning() {
    const M = this.app?.mission;
    const want = !!M && !M.paused && document.visibilityState !== 'hidden';
    if (want === this.running) return;
    this.running = want;
    const ctx = this.ctx;
    if (!ctx || !this.engine) return;
    const chain = (this.ctxBusy ?? Promise.resolve()).then(async () => {
      if (this.running) {
        if (ctx.state !== 'running') await ctx.resume().catch(() => {});
        this.applyVolume();
      } else {
        this.applyVolume(0.012);
        await new Promise((r) => setTimeout(r, 80));
        if (!this.running && ctx.state === 'running') await ctx.suspend().catch(() => {});
      }
    });
    this.ctxBusy = chain.finally(() => { if (this.ctxBusy === chain) this.ctxBusy = null; });
  }

  private tick() {
    const app = this.app, eng = this.engine, mixer = this.mixer, feed = this.feed, ctx = this.ctx;
    if (!app || !eng || !mixer || !feed || !ctx) return;
    const t0w = performance.now();
    const wallDt = (t0w - this.lastWall) / 1000;
    this.lastWall = t0w;
    this.reconcileRunning();
    if (!this.running) return;
    const M = app.mission;
    const now = ctx.currentTime;
    if (feed.bind(M.sim)) { this.reset(); }
    const cam = app.rig.mode as AetherCamera;
    const c = app.camWorld;
    const camEnu: V3 = [c[0], -c[2], Math.max(c[1] - GROUND_Y, 1.7)];
    mixer.setListener(listenerForCamera(this.prep!, cam, camEnu));
    if ((M.state === 'idle' || M.sim.lost) && !M.replay) { eng.silence(now); this.lastT = NaN; return; }
    const t = M.replay || M.state !== 'countdown' ? M.view.t : M.met;
    const warp = M.warp;
    // discontinuity: first tick, backwards, or a jump far beyond what the clock allows (seek)
    const jump = isNaN(this.lastT) || t < this.lastT - 1e-6 || t - this.lastT > wallDt * warp + 1.0;
    if (jump) { feed.refill(mixer.ring, t); this.sched.reset(); }
    else feed.feed(mixer.ring, t);
    eng.tau0 = M.sim.history[0].t;
    const m = mixer.compute(t, warp);
    if (jump) eng.resync(now, m); else eng.apply(m, now);
    // one-shots: events already emitted (sim time <= t), planned for this listener, delayed by propagation
    const plans: OneShotPlan[] = [];
    for (const ev of M.sim.events) {
      if (ev.t > t + 1e-9) continue;
      const k = `${ev.name}@${ev.t}|${m.listenerKey}`;
      let p = this.plans.get(k);
      if (!p) { p = mixer.planOneShots(ev); this.plans.set(k, p); }
      plans.push(...p);
    }
    const look = 0.12 * warp;
    for (const f of this.sched.update(t, plans, m.warp, look, eng.durationOf, eng.isPlaying)) {
      eng.fireOneShot(f, now, Math.max(0, f.delay) / Math.max(warp, 1e-3));
    }
    this.lastT = t;
    this.tickUsAcc += (performance.now() - t0w) * 1000;
    if (++this.tickN >= 30) { this.info.tickUs = this.tickUsAcc / this.tickN; this.tickUsAcc = 0; this.tickN = 0; }
  }

  stats(): (EngineStats & { status: AudioStatus; ctxState: string }) | { status: AudioStatus; ctxState: string } {
    const base = { status: this.status, ctxState: this.ctx?.state ?? 'none' };
    return this.engine ? { ...base, ...this.engine.stats() } : base;
  }

  /** wait until assets are decoded (tests / benchmarks) */
  ready() { return this.loadP ?? Promise.resolve(); }
}
