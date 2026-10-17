import { MU, RE } from './dynamics';

export interface Elements {
  a: number; e: number; i_deg: number; rp: number; ra: number;
  periapsis_alt: number; apoapsis_alt: number; period: number; energy: number; h: number;
  raan_deg: number; argp_deg: number; nu_deg: number;
}

export function elements(r: ArrayLike<number>, v: ArrayLike<number>, mu = MU): Elements {
  const [rx, ry, rz] = [r[0], r[1], r[2]];
  const [vx, vy, vz] = [v[0], v[1], v[2]];
  const rn = Math.hypot(rx, ry, rz);
  const v2 = vx * vx + vy * vy + vz * vz;
  const hx = ry * vz - rz * vy, hy = rz * vx - rx * vz, hz = rx * vy - ry * vx;
  const hn = Math.hypot(hx, hy, hz);
  const energy = 0.5 * v2 - mu / rn;
  const a = Math.abs(energy) > 1e-12 ? -mu / (2 * energy) : Infinity;
  const rv = rx * vx + ry * vy + rz * vz;
  const ex = ((v2 - mu / rn) * rx - rv * vx) / mu;
  const ey = ((v2 - mu / rn) * ry - rv * vy) / mu;
  const ez = ((v2 - mu / rn) * rz - rv * vz) / mu;
  const e = Math.hypot(ex, ey, ez);
  const inc = (Math.acos(Math.max(-1, Math.min(1, hz / hn))) * 180) / Math.PI;
  let rp: number, ra: number, period: number;
  if (energy < 0) { rp = a * (1 - e); ra = a * (1 + e); period = 2 * Math.PI * Math.sqrt((a * a * a) / mu); }
  else { rp = (hn * hn) / mu / (1 + e); ra = Infinity; period = Infinity; }
  const nx = -hy, ny = hx; // node vector z x h
  const nn = Math.hypot(nx, ny);
  let raan = nn > 1e-9 ? Math.atan2(ny, nx) : 0;
  if (raan < 0) raan += 2 * Math.PI;
  let argp = 0;
  if (nn > 1e-9 && e > 1e-9) {
    argp = Math.acos(Math.max(-1, Math.min(1, (nx * ex + ny * ey) / (nn * e))));
    if (ez < 0) argp = 2 * Math.PI - argp;
  }
  let nu = 0;
  if (e > 1e-9) {
    nu = Math.acos(Math.max(-1, Math.min(1, (ex * rx + ey * ry + ez * rz) / (e * rn))));
    if (rv < 0) nu = 2 * Math.PI - nu;
  }
  const D = 180 / Math.PI;
  return {
    a, e, i_deg: inc, rp, ra, periapsis_alt: rp - RE, apoapsis_alt: ra - RE, period, energy, h: hn,
    raan_deg: raan * D, argp_deg: argp * D, nu_deg: nu * D,
  };
}

/** Sample a Kepler orbit (two-body) as ECI points, starting from the current state. */
export function keplerPath(r: ArrayLike<number>, v: ArrayLike<number>, n: number, duration: number, mu = MU): Float64Array {
  // universal-variable propagation sampled n times (vacuum two-body prediction)
  const out = new Float64Array(n * 3);
  const r0 = [r[0], r[1], r[2]], v0 = [v[0], v[1], v[2]];
  const rn = Math.hypot(r0[0], r0[1], r0[2]);
  const vr = (r0[0] * v0[0] + r0[1] * v0[1] + r0[2] * v0[2]) / rn;
  const alpha = 2 / rn - (v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]) / mu;
  const smu = Math.sqrt(mu);
  const C = (z: number) => (z > 1e-8 ? (1 - Math.cos(Math.sqrt(z))) / z : z < -1e-8 ? (Math.cosh(Math.sqrt(-z)) - 1) / -z : 0.5 - z / 24);
  const S = (z: number) => {
    if (z > 1e-8) { const s = Math.sqrt(z); return (s - Math.sin(s)) / (s * s * s); }
    if (z < -1e-8) { const s = Math.sqrt(-z); return (Math.sinh(s) - s) / (s * s * s); }
    return 1 / 6 - z / 120;
  };
  let x = 0;
  for (let k = 0; k < n; k++) {
    const dt = (duration * k) / (n - 1);
    if (k === 0) x = 0; // warm start from previous x otherwise
    for (let it = 0; it < 50; it++) {
      const z = alpha * x * x;
      const f = ((rn * vr) / smu) * x * x * C(z) + (1 - alpha * rn) * x * x * x * S(z) + rn * x - smu * dt;
      const df = ((rn * vr) / smu) * x * (1 - alpha * x * x * S(z)) + (1 - alpha * rn) * x * x * C(z) + rn;
      const dx = f / df;
      x -= dx;
      if (Math.abs(dx) < 1e-8) break;
    }
    const z = alpha * x * x;
    const fq = 1 - ((x * x) / rn) * C(z);
    const gq = dt - (x * x * x / smu) * S(z);
    out[k * 3] = fq * r0[0] + gq * v0[0];
    out[k * 3 + 1] = fq * r0[1] + gq * v0[1];
    out[k * 3 + 2] = fq * r0[2] + gq * v0[2];
  }
  return out;
}
