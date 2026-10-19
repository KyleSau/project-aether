// Camera -> listener mapping, exactly as acoustic_profile.json listeners.aether recommends.
import type { V3 } from './acoustics';
import type { PreparedProfile } from './profile';

export type ListenerMode = 'ground' | 'chase' | 'onboard' | 'cabin' | 'vacuum';
export type AetherCamera = 'pad' | 'engine' | 'tracking' | 'onboard' | 'aerial' | 'free' | 'orbital';

export interface Listener {
  key: string; // changes whenever the acoustic situation changes (camera switch)
  mode: ListenerMode;
  /** ground: fixed ENU position; chase (free): camera ENU position, updated per tick */
  pos: V3;
  /** fixed microphone gain reference (dB SPL that maps to target_peak_dbfs); null = derive from the source model */
  LpMax: number | null;
  /** chase: fixed geometry (Engines camera: r = 30 m, theta = 100 deg) */
  chaseR?: number;
  chaseTheta?: number;
}

const PRESET_OF: Record<AetherCamera, string> = {
  pad: 'Pad', engine: 'Engines', tracking: 'Tracker', onboard: 'Onboard', aerial: 'Aerial', free: 'Free', orbital: 'Orbital',
};

/** Listener for an Aether camera. `camEnu` is the camera position in pad-local ENU (used by Free). */
export function listenerForCamera(p: PreparedProfile, cam: AetherCamera, camEnu: V3): Listener {
  const name = PRESET_OF[cam];
  const pr = p.raw.listeners.aether[name];
  if (!pr) throw new Error(`profile has no listener preset ${name}`);
  switch (pr.mode) {
    case 'ground':
      return { key: name, mode: 'ground', pos: [...pr.pos_enu!] as V3, LpMax: pr.Lp_max_db ?? null };
    case 'chase':
      // Engines: rides 26 m off the engines -> r = 30 m, theta = 100 deg (profile note); Free: actual geometry
      return cam === 'engine'
        ? { key: name, mode: 'chase', pos: [...camEnu] as V3, LpMax: null, chaseR: 30, chaseTheta: 100 }
        : { key: name, mode: 'chase', pos: [...camEnu] as V3, LpMax: null };
    case 'onboard':
    case 'cabin':
      return { key: name, mode: pr.mode, pos: [0, 0, 0], LpMax: null };
    case 'vacuum':
    default:
      return { key: name, mode: 'vacuum', pos: [...camEnu] as V3, LpMax: null };
  }
}
