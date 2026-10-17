import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { MissionSim } from '../src/physics/sim';
import { atmosphere } from '../src/physics/atmosphere';
import { elements } from '../src/physics/orbit';
import { engineThrust, RE } from '../src/physics/dynamics';
import type { VehicleConfig } from '../src/physics/config';

const DATA = join(__dirname, '..', '..', 'data');
const cfg: VehicleConfig = JSON.parse(readFileSync(join(DATA, 'vehicle.json'), 'utf8'));
const ref = JSON.parse(readFileSync(join(DATA, 'reference_trajectory.json'), 'utf8'));

describe('atmosphere (TS) vs USSA-1976', () => {
  const table: [number, number, number, number][] = [
    [0, 288.15, 101325.0, 1.225], [5000, 255.676, 54048.3, 0.736429], [11000, 216.774, 22699.9, 0.364801],
    [20000, 216.65, 5529.31, 0.0889097], [50000, 270.65, 79.7787, 1.02688e-3], [80000, 198.639, 1.05247, 1.8458e-5],
  ];
  for (const [z, T, P, rho] of table) {
    it(`z=${z}`, () => {
      const a = atmosphere(z);
      expect(Math.abs(a.T - T)).toBeLessThan(0.01);
      expect(Math.abs(a.P / P - 1)).toBeLessThan(2e-4);
      expect(Math.abs(a.rho / rho - 1)).toBeLessThan(2e-4);
    });
  }
});

describe('engine model', () => {
  it('thrust varies with ambient pressure by n*Ae*pa', () => {
    const e = cfg.engines.stage1;
    const [fv] = engineThrust(e, 1, 0);
    const [fs] = engineThrust(e, 1, 101325);
    expect(fv - fs).toBeCloseTo(e.count * e.ae_m2 * 101325, 3);
  });
});

function sampleAt(sim: MissionSim, t: number): { y: Float64Array } | null {
  const H = sim.history;
  let lo = 0, hi = H.length - 1;
  if (t < H[0].t || t > H[hi].t) return null;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (H[m].t <= t) lo = m; else hi = m; }
  const a = H[lo], b = H[hi];
  const f = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
  const y = new Float64Array(15);
  for (let i = 0; i < 15; i++) y[i] = a.y[i] + f * (b.y[i] - a.y[i]);
  return { y };
}

describe('TS RK4 real-time sim vs Python solve_ivp reference', () => {
  const sim = new MissionSim(cfg, 0.02);
  const t0 = performance.now();
  sim.advanceTo(ref.samples[ref.samples.length - 1].t);
  const wall = performance.now() - t0;
  it('event sequence and times agree', () => {
    const names = sim.events.map((e) => e.name);
    const rnames = ref.events.map((e: [string, number]) => e[0]);
    expect(names).toEqual(rnames);
    for (const [n, t] of ref.events as [string, number][]) {
      const ev = sim.events.find((e) => e.name === n)!;
      expect(Math.abs(ev.t - t)).toBeLessThan(0.05);
    }
  });
  it('trajectory agrees within tolerance at sampled times', () => {
    let maxPos = 0, maxVel = 0, n = 0, worst = 0;
    for (const s of ref.samples as { t: number; y: number[] }[]) {
      const f = sampleAt(sim, s.t);
      if (!f) continue;
      n++;
      const dp = Math.hypot(f.y[0] - s.y[0], f.y[1] - s.y[1], f.y[2] - s.y[2]);
      const dv = Math.hypot(f.y[3] - s.y[3], f.y[4] - s.y[4], f.y[5] - s.y[5]);
      maxPos = Math.max(maxPos, dp);
      if (dv > maxVel) worst = s.t;
      maxVel = Math.max(maxVel, dv);
    }
    console.log(`TS vs Python: max |dr| = ${maxPos.toFixed(2)} m, max |dv| = ${maxVel.toFixed(4)} m/s, worst t=${worst} steps=${sim.stepCount}, wall=${wall.toFixed(0)} ms`);
    expect(n).toBeGreaterThan(ref.samples.length - 5);
    expect(maxPos).toBeLessThan(10);
    expect(maxVel).toBeLessThan(0.1);
  });
  it('reaches the same orbit', () => {
    const seco = sim.history.find((hf) => hf.phase === 5)!;
    const el = elements(seco.y.subarray(0, 3), seco.y.subarray(3, 6));
    const re = ref.elements_seco;
    expect(Math.abs(el.periapsis_alt - re.periapsis_alt)).toBeLessThan(500);
    expect(Math.abs(el.apoapsis_alt - re.apoapsis_alt)).toBeLessThan(500);
    expect(Math.abs(el.i_deg - re.i_deg)).toBeLessThan(0.01);
    expect(el.periapsis_alt).toBeGreaterThan(150e3);
  });
  it('max-Q agrees', () => {
    expect(Math.abs(sim.maxQ.qbar - ref.max_q.qbar) / ref.max_q.qbar).toBeLessThan(0.01);
  });
  it('altitude non-negative and vehicle never lost', () => {
    expect(sim.lost).toBe(false);
    for (const f of sim.history) expect(Math.hypot(f.y[0], f.y[1], f.y[2]) - RE).toBeGreaterThan(-1);
  });
});
