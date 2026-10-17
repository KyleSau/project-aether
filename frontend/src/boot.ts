import type { VehicleConfig } from './physics/config';
import type { Stepper } from './physics/sim';
import { Engine } from './engine/Engine';

export interface ExtraData { thermo: unknown | null; plumeTable: unknown | null; reference: unknown | null; benchmarks: unknown | null; thermal: unknown | null }
export const extra: ExtraData = { thermo: null, plumeTable: null, reference: null, benchmarks: null, thermal: null };

async function getJson(path: string) {
  try {
    const r = await fetch(path);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

/** Physics backend: TS reference port by default; ?physics=wasm delegates the RK4 inner loop to Rust/WASM. */
async function makeStepperFactory(): Promise<{ make: (c: VehicleConfig) => Stepper | undefined; label: string }> {
  const want = new URLSearchParams(location.search).get('physics') ?? 'ts';
  if (want === 'wasm') {
    try {
      const mod = await import('./wasm/aether_wasm.js');
      const wasmUrl = new URL('./wasm/aether_wasm_bg.wasm', import.meta.url);
      await mod.default({ module_or_path: wasmUrl });
      return {
        label: 'WASM RK4',
        make: (c) => {
          const ws = new mod.WasmStepper(JSON.stringify(c));
          return { rk4: (t, y, ctx, h) => ws.rk4(t, y, ctx.phase, ctx.fairing, ctx.t_ses, h) };
        },
      };
    } catch (e) {
      console.warn('WASM physics unavailable, falling back to TS', e);
    }
  }
  return { label: 'TS RK4', make: () => undefined };
}

export async function boot(canvas: HTMLCanvasElement): Promise<Engine> {
  const cfg: VehicleConfig = await getJson('./data/vehicle.json');
  if (!cfg) throw new Error('could not load data/vehicle.json');
  const [thermo, plumeTable, reference, benchmarks, thermal] = await Promise.all([
    getJson('./data/thermo.json'), getJson('./data/plume_table.json'), getJson('./data/reference_trajectory.json'),
    getJson('./data/benchmarks_physics.json'), getJson('./data/thermal.json'),
  ]);
  Object.assign(extra, { thermo, plumeTable, reference, benchmarks, thermal });
  const st = await makeStepperFactory();
  const e = new Engine(canvas, cfg, st.make, st.label);
  await e.init();
  return e;
}
