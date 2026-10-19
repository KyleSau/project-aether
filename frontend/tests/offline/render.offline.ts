// Offline validation render: the REAL browser graph (AudioEngine + AcousticMixer + scheduler) on an
// OfflineAudioContext (node-web-audio-api, a spec implementation of Web Audio), driven at 30 Hz from the
// reference telemetry (data/audio/validation/telemetry_reference.csv), exactly as the live timer does.
// Writes tools/audio/_cache/renders/browser_<name>.wav (+ .json) for tools/audio/compare_browser.py.
//
//   npm run audio:offline
import { describe, it } from 'vitest';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { AudioWorkletNode as NodeAudioWorkletNode, OfflineAudioContext } from 'node-web-audio-api';
// AudioEngine uses the global AudioWorkletNode constructor (as in browsers)
(globalThis as unknown as { AudioWorkletNode: unknown }).AudioWorkletNode ??= NodeAudioWorkletNode;
import { AudioEngine } from '../../src/audio/AudioEngine';
import { loadAssets } from '../../src/audio/assets';
import { loadLimiterWorklet } from '../../src/audio/limiter';
import { listenerForCamera, type Listener } from '../../src/audio/listeners';
import { AcousticMixer, OneShotScheduler, type OneShotPlan } from '../../src/audio/mixModel';
import { cfg, DATA, loadReferenceTelemetry, prep, ROOT } from '../audioHelpers';

const OUT = join(ROOT, 'tools', 'audio', '_cache', 'renders');
const FS = 48000;
const TICK = 1 / 30;

function wav16(path: string, x: Float32Array) {
  const n = x.length, buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVE', 8); buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(FS, 24);
  buf.writeUInt32LE(FS * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), 44 + i * 2);
  writeFileSync(path, buf);
}

async function render(name: string, listener: Listener, t0: number, t1: number, lpMaxPython: number | null, rateClamp?: [number, number]) {
  const tel = loadReferenceTelemetry();
  const n = Math.round((t1 - t0) * FS);
  const ctx = new OfflineAudioContext(1, n, FS) as unknown as OfflineAudioContext & BaseAudioContext;
  const fetchBytes = async (p: string) => { const b = readFileSync(join(DATA, 'audio', p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; };
  const assets = await loadAssets(ctx, prep, fetchBytes, '', { preferOpus: false });
  const workletOk = await loadLimiterWorklet(ctx); // Blob URL (node-web-audio-api rejects Windows file paths)
  if (!workletOk) throw new Error('AudioWorklet limiter failed to load in node-web-audio-api');
  const eng = new AudioEngine(ctx, prep, assets.buffers);
  const mixer = new AcousticMixer(prep, cfg, listener);
  const lpMaxRuntime = mixer.LpMax;
  if (rateClamp) mixer.rateClamp = rateClamp;
  if (lpMaxPython !== null) mixer.lpMaxOverride = lpMaxPython; // same mic gain as the Python render: compare shape
  eng.tau0 = tel.frames[0].t;
  const sched = new OneShotScheduler();
  const plansCache = new Map<string, OneShotPlan[]>();
  let fi = 0, ticks = 0, tickUs = 0, maxTickUs = 0;
  const fired: [string, number][] = [];
  const nTicks = Math.ceil((t1 - t0) / TICK) - 1; // suspend times must lie inside the render
  for (let k = 0; k <= nTicks; k++) {
    const ta = k * TICK;
    void ctx.suspend(ta).then(() => {
      const c0 = performance.now();
      const t = t0 + ta;
      while (fi < tel.frames.length && tel.frames[fi].t <= t + 1e-9) mixer.ring.push(tel.frames[fi++]);
      const m = mixer.compute(t, 1);
      if (k === 0) eng.resync(ctx.currentTime, m); else eng.apply(m, ctx.currentTime);
      const plans: OneShotPlan[] = [];
      for (const ev of tel.events) {
        if (ev.t > t) continue;
        const key = `${ev.name}@${ev.t}`;
        let p = plansCache.get(key);
        if (!p) { p = mixer.planOneShots(ev); plansCache.set(key, p); }
        plans.push(...p);
      }
      for (const f of sched.update(t, plans, m.warp, 0.12, eng.durationOf, eng.isPlaying)) {
        eng.fireOneShot(f, ctx.currentTime, Math.max(0, f.delay));
        fired.push([f.plan.asset, f.plan.tArrive]);
      }
      const us = (performance.now() - c0) * 1000;
      tickUs += us; maxTickUs = Math.max(maxTickUs, us); ticks++;
      void ctx.resume();
    });
  }
  const w0 = performance.now();
  const out = await ctx.startRendering();
  const wallS = (performance.now() - w0) / 1000;
  const y = out.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
  mkdirSync(OUT, { recursive: true });
  wav16(join(OUT, `browser_${name}.wav`), y);
  const info = {
    name, mode: listener.mode, pos: listener.pos, t0, t1, Lp_max_used: lpMaxPython ?? lpMaxRuntime, Lp_max_runtime: lpMaxRuntime,
    oneshots: fired, peak_dbfs: 20 * Math.log10(peak + 1e-12), ticks, mean_tick_us: tickUs / ticks, max_tick_us: maxTickUs,
    render_wall_s: wallS, realtime_factor: (t1 - t0) / wallS, decoded_mb: assets.decodedBytes / 1048576, stats: eng.stats(),
  };
  writeFileSync(join(OUT, `browser_${name}.json`), JSON.stringify(info, null, 1));
  console.log(JSON.stringify(info));
}

const py = (n: string) => JSON.parse(readFileSync(join(OUT, `launch_${n}.json`), 'utf8')).Lp_max_db as number;
const pad = listenerForCamera(prep, 'pad', [0, 0, 0]);

describe('offline browser renders', () => {
  it('pad (Aether Pad camera, 405 m)', async () => { await render('pad', pad, -6, 200, py('pad')); });
  it('obs5km (5 km NW)', async () => { await render('obs5km', { key: 'obs5km', mode: 'ground', pos: [-3536, 3536, 1.7], LpMax: null }, -6, 260, py('obs5km')); });
  it('onboard (Aether Onboard camera)', async () => { await render('onboard', listenerForCamera(prep, 'onboard', [0, 0, 0]), -6, 200, null); });
  // diagnostic: render_launch.py's delay line is not clamped; the profile clamps the read rate to [0.5, 1.5]
  it('obs5km with the read-rate clamp relaxed to [0.05, 1.5] (as the Python delay line)', async () => {
    await render('obs5km_noclamp', { key: 'obs5km', mode: 'ground', pos: [-3536, 3536, 1.7], LpMax: null }, -6, 260, py('obs5km'), [0.05, 1.5]);
  });
  it('pad with the profile preset mic gain (Lp_max 150 dB)', async () => { await render('pad_preset', pad, -6, 200, null); });
});
