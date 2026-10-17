// Physics benchmark: TS vs Rust/WASM (Node 22 / V8, same JS engine as Chrome).
// Run: npm run bench   -> writes ../data/benchmarks_physics.json
import { it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import os from 'node:os';
import { initSync, WasmSim, WasmStepper } from '../src/wasm/aether_wasm.js';
import { MissionSim, type Stepper } from '../src/physics/sim';
import { evaluate, NSTATE, newCtx } from '../src/physics/dynamics';
import type { VehicleConfig } from '../src/physics/config';

const DATA = join(__dirname, '..', '..', 'data');
const cfgText = readFileSync(join(DATA, 'vehicle.json'), 'utf8');
const cfg: VehicleConfig = JSON.parse(cfgText);
const ref = JSON.parse(readFileSync(join(DATA, 'reference_trajectory.json'), 'utf8'));
initSync({ module: readFileSync(join(__dirname, '..', 'src', 'wasm', 'aether_wasm_bg.wasm')) });
const T_END = 760;

function median(a: number[]) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }
function timeIt(fn: () => number, reps: number) {
  const ms: number[] = []; let steps = 0;
  fn(); // warm-up / JIT
  for (let i = 0; i < reps; i++) { const t0 = performance.now(); steps = fn(); ms.push(performance.now() - t0); }
  return { medianMs: median(ms), steps, stepsPerSec: steps / (median(ms) / 1000) };
}
function finalErr(y: ArrayLike<number>) {
  const last = ref.samples.find((s: { t: number }) => Math.abs(s.t - T_END) < 1e-6) ?? ref.samples[ref.samples.length - 1];
  return Math.hypot(y[0] - last.y[0], y[1] - last.y[1], y[2] - last.y[2]);
}

it('physics benchmark', () => {
  const reps = 7;
  const ts = timeIt(() => { const s = new MissionSim(cfg, 0.02); s.advanceTo(T_END); return s.stepCount; }, reps);
  const tsSim = new MissionSim(cfg, 0.02); tsSim.advanceTo(T_END);
  const wasmFull = timeIt(() => { const s = new WasmSim(cfgText, 0.02); s.advance_to(T_END); const n = s.steps(); s.free(); return n; }, reps);
  const wsim = new WasmSim(cfgText, 0.02); wsim.advance_to(T_END); const wy = wsim.state(); wsim.free();
  const stepper = new WasmStepper(cfgText);
  const wasmStep: Stepper = { rk4: (t, y, ctx, h) => stepper.rk4(t, y, ctx.phase, ctx.fairing, ctx.t_ses, h) };
  const hybrid = timeIt(() => { const s = new MissionSim(cfg, 0.02, wasmStep); s.advanceTo(T_END); return s.stepCount; }, reps);
  // raw derivative evaluations
  const y = tsSim.history[500].y, d = new Float64Array(NSTATE), ctx = newCtx(); ctx.phase = 1;
  const N = 200000;
  const ev = timeIt(() => { for (let i = 0; i < N; i++) evaluate(10, y, ctx, cfg, d); return N; }, 5);
  const out = {
    schema: 'aether-bench-physics/1', date: new Date().toISOString(), node: process.version,
    cpu: os.cpus()[0]?.model, method: `full pad-to-orbit mission (t=-3..${T_END}s, RK4 h=0.02 s, ~${ts.steps} steps incl. event refinement), median of ${reps} runs after warm-up`,
    ts_full: ts, wasm_full: wasmFull, ts_loop_wasm_rk4: hybrid,
    ts_evaluate_per_sec: ev.stepsPerSec,
    speedup_wasm_full_vs_ts: wasmFull.stepsPerSec / ts.stepsPerSec,
    speedup_hybrid_vs_ts: hybrid.stepsPerSec / ts.stepsPerSec,
    accuracy: {
      ts_final_pos_err_vs_python_m: finalErr(tsSim.y),
      wasm_final_pos_err_vs_python_m: finalErr(wy),
      wasm_vs_ts_final_pos_diff_m: Math.hypot(wy[0] - tsSim.y[0], wy[1] - tsSim.y[1], wy[2] - tsSim.y[2]),
    },
    realtime_budget: `At 50x time warp the sim needs 50/0.02 = 2500 RK4 steps per second; TS delivers ${(ts.stepsPerSec / 2500).toFixed(0)}x that.`,
  };
  writeFileSync(join(DATA, 'benchmarks_physics.json'), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out, null, 1));
}, 300000);
