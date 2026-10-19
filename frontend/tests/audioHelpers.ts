// Shared helpers for the audio tests and the offline validation render (node only).
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { newFrame, type SourceFrame } from '../src/audio/acoustics';
import { prepareProfile, type AcousticProfile } from '../src/audio/profile';
import type { VehicleConfig } from '../src/physics/config';

export const ROOT = join(__dirname, '..', '..');
export const DATA = join(ROOT, 'data');
export const cfg: VehicleConfig = JSON.parse(readFileSync(join(DATA, 'vehicle.json'), 'utf8'));
export const rawProfile: AcousticProfile = JSON.parse(readFileSync(join(DATA, 'audio', 'acoustic_profile.json'), 'utf8'));
export const prep = prepareProfile(rawProfile);

export interface RefTelemetry { frames: SourceFrame[]; events: { name: string; t: number }[] }

/** data/audio/validation/telemetry_reference.csv (render_launch.py format) */
export function loadReferenceTelemetry(): RefTelemetry {
  const txt = readFileSync(join(DATA, 'audio', 'validation', 'telemetry_reference.csv'), 'utf8').split(/\r?\n/);
  const events = txt[0].split('events:')[1].trim().split(';').map((kv) => { const [n, t] = kv.split('='); return { name: n, t: parseFloat(t) }; });
  const cols = txt[1].split(',');
  const ix = (k: string) => cols.indexOf(k);
  const frames: SourceFrame[] = [];
  for (const line of txt.slice(2)) {
    if (!line.trim()) continue;
    const v = line.split(',').map(Number);
    const f = newFrame();
    f.t = v[ix('t')]; f.phase = v[ix('phase')]; f.thrust = v[ix('thrust')]; f.mdot = v[ix('mdot')]; f.throttle = v[ix('throttle')];
    f.P = v[ix('P')]; f.T = v[ix('T')]; f.rho = v[ix('rho')]; f.a = v[ix('a')]; f.h = v[ix('h')]; f.mach = v[ix('mach')]; f.qbar = v[ix('qbar')];
    f.noz = [v[ix('noz_e')], v[ix('noz_n')], v[ix('noz_u')]];
    f.ax = [v[ix('ax_e')], v[ix('ax_n')], v[ix('ax_u')]];
    frames.push(f);
  }
  return { frames, events };
}
