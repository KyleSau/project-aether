// Telemetry adapter: MissionSim history (0.1 s frames, deterministic, also what replay plays back) ->
// SourceFrames in pad-local ENU, the frame of the acoustic profile and of render_launch.py.
import type { VehicleConfig } from '../physics/config';
import { evaluate, newTelemetry, NSTATE, OMEGA_E, RE, type Ctx } from '../physics/dynamics';
import type { MissionSim } from '../physics/sim';
import { newFrame, type SourceFrame } from './acoustics';
import type { TelemetryRing } from './telemetryRing';

/** ECI -> pad-local ENU (same construction as engine/frames.ts, without three.js) */
export class EnuFrame {
  pad: [number, number, number];
  E: [number, number, number];
  N: [number, number, number];
  U: [number, number, number];
  wE: number;
  constructor(cfg: VehicleConfig, public t0: number) {
    const lat = (cfg.site.latitude_deg * Math.PI) / 180, lon = (cfg.site.longitude_deg * Math.PI) / 180;
    const r = RE + cfg.site.pad_altitude_m;
    this.pad = [r * Math.cos(lat) * Math.cos(lon), r * Math.cos(lat) * Math.sin(lon), r * Math.sin(lat)];
    this.U = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
    this.E = [-Math.sin(lon), Math.cos(lon), 0];
    this.N = [-Math.sin(lat) * Math.cos(lon), -Math.sin(lat) * Math.sin(lon), Math.cos(lat)];
    this.wE = cfg.site.rotating_earth ? OMEGA_E : 0;
  }
  vec(v: ArrayLike<number>, t: number, out: number[]) {
    const th = this.wE * (t - this.t0), c = Math.cos(th), s = Math.sin(th);
    const x = c * v[0] + s * v[1], y = -s * v[0] + c * v[1], z = v[2];
    const { E, N, U } = this;
    out[0] = E[0] * x + E[1] * y + E[2] * z;
    out[1] = N[0] * x + N[1] * y + N[2] * z;
    out[2] = U[0] * x + U[1] * y + U[2] * z;
    return out;
  }
  pos(r: ArrayLike<number>, t: number, out: number[]) {
    const th = this.wE * (t - this.t0), c = Math.cos(th), s = Math.sin(th);
    const x = c * r[0] + s * r[1] - this.pad[0], y = -s * r[0] + c * r[1] - this.pad[1], z = r[2] - this.pad[2];
    const { E, N, U } = this;
    out[0] = E[0] * x + E[1] * y + E[2] * z;
    out[1] = N[0] * x + N[1] * y + N[2] * z;
    out[2] = U[0] * x + U[1] * y + U[2] * z;
    return out;
  }
}

export class SimFeed {
  private sim: MissionSim | null = null;
  private idx = 0;
  private enu: EnuFrame | null = null;
  private tel = newTelemetry();
  private dy = new Float64Array(NSTATE);
  private ctx: Ctx = { phase: 0, fairing: true, t_meco: 0, t_sep: 0, t_ses: 0, t_seco: 0 };
  private f = newFrame();
  private noz = new Float64Array(3);
  private bx = [0, 0, 0];
  private tmp = [0, 0, 0];

  constructor(private cfg: VehicleConfig) {}

  /** true when the simulation object changed (restart) */
  bind(sim: MissionSim): boolean {
    if (sim === this.sim) return false;
    this.sim = sim;
    this.idx = 0;
    this.enu = new EnuFrame(this.cfg, sim.history[0].t);
    return true;
  }

  frameAt(i: number): SourceFrame {
    const sim = this.sim!, h = sim.history[i], f = this.f, c = this.ctx, sc = sim.ctx;
    c.phase = h.phase; c.fairing = h.fairing; c.t_meco = sc.t_meco; c.t_sep = sc.t_sep; c.t_ses = sc.t_ses; c.t_seco = sc.t_seco;
    evaluate(h.t, h.y, c, this.cfg, this.dy, this.tel);
    const y = h.y, tel = this.tel;
    const [w, x, yy, z] = [y[6], y[7], y[8], y[9]];
    const bx = this.bx;
    bx[0] = 1 - 2 * (yy * yy + z * z); bx[1] = 2 * (x * yy + w * z); bx[2] = 2 * (x * z - w * yy);
    const xN = h.phase > 2 ? this.cfg.geometry.stage2.x0 : 0; // active nozzle exit (body x, from the stage-1 exit plane)
    for (let k = 0; k < 3; k++) this.noz[k] = y[k] + bx[k] * xN;
    const enu = this.enu!;
    enu.pos(this.noz, h.t, this.tmp);
    f.noz[0] = this.tmp[0]; f.noz[1] = this.tmp[1]; f.noz[2] = this.tmp[2];
    enu.vec(bx, h.t, this.tmp);
    f.ax[0] = -this.tmp[0]; f.ax[1] = -this.tmp[1]; f.ax[2] = -this.tmp[2];
    f.t = h.t; f.phase = h.phase; f.thrust = tel.thrust; f.mdot = tel.mdot; f.throttle = tel.throttle;
    f.P = tel.P; f.T = tel.T; f.rho = tel.rho; f.a = tel.a; f.h = tel.h; f.mach = tel.mach; f.qbar = tel.qbar;
    return f;
  }

  /** push every history frame with t <= tUpTo not yet pushed */
  feed(ring: TelemetryRing, tUpTo: number) {
    const H = this.sim!.history;
    while (this.idx < H.length && H[this.idx].t <= tUpTo + 1e-9) ring.push(this.frameAt(this.idx++));
  }

  /** rebuild the ring for a jump to time t (replay seek / restart) */
  refill(ring: TelemetryRing, t: number, span = 800) {
    ring.clear();
    const H = this.sim!.history;
    let lo = 0, hi = H.length;
    const t0 = t - span;
    while (lo < hi) { const m = (lo + hi) >> 1; if (H[m].t < t0) lo = m + 1; else hi = m; }
    this.idx = Math.max(0, lo - 1);
    this.feed(ring, t);
  }
}
