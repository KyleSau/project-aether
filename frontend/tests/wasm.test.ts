import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { initSync, WasmSim } from '../src/wasm/aether_wasm.js';
import { MissionSim } from '../src/physics/sim';
import type { VehicleConfig } from '../src/physics/config';

const DATA = join(__dirname, '..', '..', 'data');
const cfgText = readFileSync(join(DATA, 'vehicle.json'), 'utf8');
const cfg: VehicleConfig = JSON.parse(cfgText);
initSync({ module: readFileSync(join(__dirname, '..', 'src', 'wasm', 'aether_wasm_bg.wasm')) });

describe('WASM (Rust) integrator vs TS integrator', () => {
  it('identical event sequence and trajectory to round-off', () => {
    const tEnd = 760;
    const ts = new MissionSim(cfg, 0.02);
    ts.advanceTo(tEnd);
    const ws = new WasmSim(cfgText, 0.02);
    ws.advance_to(tEnd);
    const wy = ws.state();
    const evs: [string, number][] = JSON.parse(ws.events_json());
    expect(evs.map((e) => e[0])).toEqual(ts.events.map((e) => e.name));
    for (let i = 0; i < evs.length; i++) expect(Math.abs(evs[i][1] - ts.events[i].t)).toBeLessThan(1e-6);
    const dr = Math.hypot(wy[0] - ts.y[0], wy[1] - ts.y[1], wy[2] - ts.y[2]);
    const dv = Math.hypot(wy[3] - ts.y[3], wy[4] - ts.y[4], wy[5] - ts.y[5]);
    console.log(`WASM vs TS at t=${tEnd}s: |dr|=${dr.toExponential(3)} m |dv|=${dv.toExponential(3)} m/s, steps ts=${ts.stepCount} wasm=${ws.steps()}`);
    expect(dr).toBeLessThan(1e-3);
    expect(dv).toBeLessThan(1e-5);
    expect(ws.phase()).toBe(5);
    ws.free();
  });
});
