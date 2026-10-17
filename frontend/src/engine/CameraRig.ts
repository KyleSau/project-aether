// Camera modes. All positions are float64 render-world coordinates (floating origin applied later).
export type CamMode = 'pad' | 'engine' | 'tracking' | 'onboard' | 'aerial' | 'free' | 'orbital';
export const CAM_MODES: { id: CamMode; label: string }[] = [
  { id: 'pad', label: 'Pad' },
  { id: 'engine', label: 'Engines' },
  { id: 'tracking', label: 'Tracker' },
  { id: 'onboard', label: 'Onboard' },
  { id: 'aerial', label: 'Aerial' },
  { id: 'free', label: 'Free' },
  { id: 'orbital', label: 'Orbital' },
];

export type V3 = [number, number, number];
const add = (a: V3, b: V3, s = 1): V3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export interface RigInput {
  base: V3; // active vehicle base (body x = 0) position
  axis: V3; // active vehicle nose direction
  side: V3; // vehicle body +y in render frame
  s1Base: V3 | null; // spent stage
  separated: boolean;
  earthCenter: V3;
  velDir: V3;
  alt: number;
}

export interface RigOutput { pos: V3; target: V3; up: V3; fov: number }

export class CameraRig {
  mode: CamMode = 'pad';
  // free/orbital orbit-control state
  yaw = 0.8;
  pitch = 0.15;
  dist = 120;
  orbYaw = 0.6;
  orbPitch = 0.35;
  orbDist = 2.6e6;
  private smoothFov = 40;

  setMode(m: CamMode) {
    this.mode = m;
  }

  rotate(dx: number, dy: number) {
    if (this.mode === 'orbital') {
      this.orbYaw -= dx * 0.005;
      this.orbPitch = Math.max(-1.4, Math.min(1.4, this.orbPitch + dy * 0.005));
    } else {
      this.yaw -= dx * 0.005;
      this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy * 0.005));
    }
  }
  zoom(f: number) {
    if (this.mode === 'orbital') this.orbDist = Math.max(2e4, Math.min(4e7, this.orbDist * f));
    else this.dist = Math.max(8, Math.min(5e5, this.dist * f));
  }

  update(inp: RigInput, dt: number): RigOutput {
    const center = add(inp.base, inp.axis, inp.separated ? 8 : 32);
    let pos: V3, target: V3 = center, up: V3 = [0, 1, 0];
    let fov = 40;
    const worldUp = norm(sub(center, inp.earthCenter));
    switch (this.mode) {
      case 'pad': {
        pos = [310, -13.5, 260];
        const d = len(sub(center, pos));
        fov = Math.max(1.2, Math.min(38, (2 * Math.atan(55 / d) * 180) / Math.PI));
        break;
      }
      case 'tracking': {
        pos = [-1800, 2, 2600];
        const d = len(sub(center, pos));
        fov = Math.max(0.25, Math.min(30, (2 * Math.atan(70 / d) * 180) / Math.PI));
        break;
      }
      case 'aerial': {
        pos = [1400, 650, -1700];
        const d = len(sub(center, pos));
        fov = Math.max(1.0, Math.min(55, (2 * Math.atan(400 / d) * 180) / Math.PI));
        target = inp.alt < 3000 ? add(center, [0, -30, 0]) : center;
        break;
      }
      case 'engine': {
        const off = cross(inp.axis, inp.side);
        pos = add(add(inp.base, inp.axis, 6), add(inp.side, off, 0.6), 16);
        target = add(inp.base, inp.axis, -10);
        up = inp.axis;
        fov = 50;
        break;
      }
      case 'onboard': {
        const off = cross(inp.axis, inp.side);
        const mountX = inp.separated ? 50 : 40;
        pos = add(add(inp.base, inp.axis, mountX), off, 3.1);
        target = add(add(inp.base, inp.axis, mountX - 60), off, 7.5);
        up = norm(off);
        fov = 62;
        break;
      }
      case 'free': {
        const cy = Math.cos(this.pitch), sy = Math.sin(this.pitch);
        // orbit around the vehicle in a local frame (worldUp, east-ish)
        const e = norm(cross([0, 0, -1], worldUp));
        const n = cross(worldUp, e);
        const dir = add(add([0, 0, 0], e, Math.cos(this.yaw) * cy), add(n, worldUp, 0), Math.sin(this.yaw) * cy);
        const dd = add(dir, worldUp, sy);
        pos = add(center, dd, this.dist);
        up = worldUp;
        fov = 45;
        break;
      }
      case 'orbital':
      default: {
        const cy = Math.cos(this.orbPitch), sy = Math.sin(this.orbPitch);
        const e = norm(cross([0, 0, -1], worldUp));
        const n = cross(worldUp, e);
        const dir: V3 = [
          e[0] * Math.cos(this.orbYaw) * cy + n[0] * Math.sin(this.orbYaw) * cy + worldUp[0] * sy,
          e[1] * Math.cos(this.orbYaw) * cy + n[1] * Math.sin(this.orbYaw) * cy + worldUp[1] * sy,
          e[2] * Math.cos(this.orbYaw) * cy + n[2] * Math.sin(this.orbYaw) * cy + worldUp[2] * sy,
        ];
        pos = add(center, dir, this.orbDist);
        target = add(center, sub(inp.earthCenter, center), 0.25);
        up = worldUp;
        fov = 50;
        break;
      }
    }
    // smooth only the zoom of long-lens cameras
    const k = 1 - Math.exp(-dt * 3);
    this.smoothFov += (fov - this.smoothFov) * k;
    if (this.mode !== 'pad' && this.mode !== 'tracking' && this.mode !== 'aerial') this.smoothFov = fov;
    return { pos, target, up, fov: this.smoothFov };
  }
}
