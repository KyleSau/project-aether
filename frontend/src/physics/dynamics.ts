// 6-DOF dynamics, guidance and attitude control — real-time port of
// physics/aether_physics/dynamics.py (the authoritative reference). Keep in sync.
import { atmosphere, type AtmoOut } from './atmosphere';
import type { EngineCfg, VehicleConfig } from './config';

export const MU = 3.986004418e14;
export const RE = 6378137.0;
export const J2 = 1.08262668e-3;
export const OMEGA_E = 7.292115e-5;
export const G0 = 9.80665;
const DEG = Math.PI / 180;

export const PAD = 0, S1_BURN = 1, COAST_SEP = 2, COAST_S2 = 3, S2_BURN = 4, ORBIT = 5, S2_DEPLETED = 6;
export const PHASE_NAMES = ['PAD', 'S1_BURN', 'COAST_SEP', 'COAST_S2', 'S2_BURN', 'ORBIT', 'S2_DEPLETED'];
export const NSTATE = 15;

export interface Ctx {
  phase: number;
  fairing: boolean;
  t_meco: number;
  t_sep: number;
  t_ses: number;
  t_seco: number;
}

export function newCtx(): Ctx {
  return { phase: PAD, fairing: true, t_meco: 0, t_sep: 0, t_ses: 0, t_seco: 0 };
}

export interface MassProps {
  m: number;
  x_cg: number;
  ixx: number;
  iyy: number;
}

export function interp(xs: number[], ys: number[], x: number): number {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const f = (x - xs[lo]) / (xs[lo + 1] - xs[lo]);
  return ys[lo] + f * (ys[lo + 1] - ys[lo]);
}

const CB = new Float64Array(24); // up to 6 components x [m, xc, ixx, iyy_own]
function cyl(i: number, m: number, x0: number, x1: number, r: number, shell: boolean): number {
  const l = Math.max(x1 - x0, 1e-6);
  const o = i * 4;
  CB[o] = m;
  CB[o + 1] = 0.5 * (x0 + x1);
  CB[o + 2] = shell ? m * r * r : 0.5 * m * r * r;
  CB[o + 3] = shell ? m * ((r * r) / 2 + (l * l) / 12) : (m * (3 * r * r + l * l)) / 12;
  return i + 1;
}

export function massProps(cfg: VehicleConfig, mp1: number, mp2: number, stage: number, fairing: boolean, out?: MassProps): MassProps {
  const g = cfg.geometry, ms = cfg.mass, r = g.radius_m;
  let n = 0;
  if (stage === 1) {
    const s1 = g.stage1;
    n = cyl(n, ms.stage1_dry_kg, s1.x0, s1.x1, r, true);
    if (mp1 > 0) {
      const frac = mp1 / ms.stage1_prop_kg;
      n = cyl(n, mp1, s1.tank_x0, s1.tank_x0 + frac * (s1.tank_x1 - s1.tank_x0), r * 0.98, false);
    }
  }
  const s2 = g.stage2;
  n = cyl(n, ms.stage2_dry_kg, s2.x0, s2.x1, r, true);
  if (mp2 > 0) {
    const frac = mp2 / ms.stage2_prop_kg;
    n = cyl(n, mp2, s2.tank_x0, s2.tank_x0 + frac * (s2.tank_x1 - s2.tank_x0), r * 0.98, false);
  }
  const pl = g.payload;
  n = cyl(n, ms.payload_kg, pl.x0, pl.x1, pl.radius_m, false);
  if (fairing) {
    const fa = g.fairing;
    n = cyl(n, ms.fairing_kg, fa.x0, fa.x1, fa.radius_m, true);
  }
  let m = 0, mx = 0, ixx = 0;
  for (let i = 0; i < n; i++) { m += CB[i * 4]; mx += CB[i * 4] * CB[i * 4 + 1]; ixx += CB[i * 4 + 2]; }
  const xcg = mx / m;
  let iyy = 0;
  for (let i = 0; i < n; i++) { const d = CB[i * 4 + 1] - xcg; iyy += CB[i * 4 + 3] + CB[i * 4] * d * d; }
  const o = out ?? { m: 0, x_cg: 0, ixx: 0, iyy: 0 };
  o.m = m; o.x_cg = xcg; o.ixx = ixx; o.iyy = iyy;
  return o;
}

/** F = n*(thr*mdot0*ve + (thr*pe0 - pa)*Ae), clamped >= 0. Returns [F, mdot_total]. */
export function engineThrust(e: EngineCfg, thr: number, pa: number): [number, number] {
  if (thr <= 0) return [0, 0];
  const mdot = thr * e.mdot0_kg_s;
  const f = e.count * (mdot * e.ve_m_s + (thr * e.pe_pa - pa) * e.ae_m2);
  return [Math.max(f, 0), e.count * mdot];
}

export const stageOf = (phase: number) => (phase <= COAST_SEP ? 1 : 2);

const smoothstep = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

export interface Telemetry {
  t: number; phase: number; h: number; rn: number; T: number; P: number; rho: number; a: number;
  V_air: number; mach: number; qbar: number; thrust: number; mdot: number; throttle: number;
  mass: number; x_cg: number; ixx: number; iyy: number; alpha: number; g_load: number;
  v_inertial: number; v_rel: number; gimbal_y: number; gimbal_z: number;
  fa: [number, number, number]; thrust_b: [number, number, number];
  acc: [number, number, number]; grav: [number, number, number]; v_air_eci: [number, number, number];
  d_cmd: [number, number, number];
}

export function newTelemetry(): Telemetry {
  return {
    t: 0, phase: 0, h: 0, rn: 0, T: 0, P: 0, rho: 0, a: 0, V_air: 0, mach: 0, qbar: 0, thrust: 0, mdot: 0,
    throttle: 0, mass: 0, x_cg: 0, ixx: 0, iyy: 0, alpha: 0, g_load: 0, v_inertial: 0, v_rel: 0,
    gimbal_y: 0, gimbal_z: 0, fa: [0, 0, 0], thrust_b: [0, 0, 0], acc: [0, 0, 0], grav: [0, 0, 0],
    v_air_eci: [0, 0, 0], d_cmd: [1, 0, 0],
  };
}

const atm: AtmoOut = { T: 0, P: 0, rho: 0, a: 0 };
const mpTmp: MassProps = { m: 0, x_cg: 0, ixx: 0, iyy: 0 };

/** Derivative of the 15-element state. Writes dy; fills tel when given. */
export function evaluate(t: number, y: ArrayLike<number>, ctx: Ctx, cfg: VehicleConfig, dy: Float64Array, tel?: Telemetry): void {
  const site = cfg.site, gd = cfg.guidance;
  const rx = y[0], ry = y[1], rz = y[2];
  const vx = y[3], vy = y[4], vz = y[5];
  let qw = y[6], qx = y[7], qy = y[8], qz = y[9];
  const wx = y[10], wy = y[11], wz = y[12];
  const mp1 = Math.max(y[13], 0), mp2 = Math.max(y[14], 0);
  const phase = ctx.phase;
  const stage = stageOf(phase);
  const mp = massProps(cfg, mp1, mp2, stage, ctx.fairing, mpTmp);
  const rn = Math.sqrt(rx * rx + ry * ry + rz * rz);
  const h = rn - RE;
  atmosphere(h, atm);
  const P = atm.P, rho = atm.rho;
  // local frame
  const ux = rx / rn, uy = ry / rn, uz = rz / rn;
  let ex = -uy, ey = ux; // z x up = (-uy, ux, 0)
  const en = Math.hypot(ex, ey);
  if (en > 1e-12) { ex /= en; ey /= en; } else { ex = 0; ey = 0; }
  const ez = 0;
  // north = up x east
  const nx = uy * ez - uz * ey, ny = uz * ex - ux * ez, nz = ux * ey - uy * ex;
  const wE = site.rotating_earth ? OMEGA_E : 0;
  // gravity
  const rn2 = rn * rn;
  const k = -MU / (rn2 * rn);
  let gx: number, gy: number, gz: number;
  if (site.j2) {
    const f = 1.5 * J2 * ((RE * RE) / rn2);
    const zz = (5 * rz * rz) / rn2;
    gx = k * rx * (1 - f * (zz - 1));
    gy = k * ry * (1 - f * (zz - 1));
    gz = k * rz * (1 - f * (zz - 3));
  } else { gx = k * rx; gy = k * ry; gz = k * rz; }
  // earth-relative and air-relative velocity
  const vrx = vx + wE * ry, vry = vy - wE * rx, vrz = vz; // v - (wE z) x r
  const ws = interp(cfg.wind.alt_m, cfg.wind.speed_m_s, h);
  const vax = vrx - ws * ex, vay = vry - ws * ey, vaz = vrz - ws * ez;
  // rotation matrix body->ECI
  const qn = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  qw /= qn; qx /= qn; qy /= qn; qz /= qn;
  const m00 = 1 - 2 * (qy * qy + qz * qz), m01 = 2 * (qx * qy - qw * qz), m02 = 2 * (qx * qz + qw * qy);
  const m10 = 2 * (qx * qy + qw * qz), m11 = 1 - 2 * (qx * qx + qz * qz), m12 = 2 * (qy * qz - qw * qx);
  const m20 = 2 * (qx * qz - qw * qy), m21 = 2 * (qy * qz + qw * qx), m22 = 1 - 2 * (qx * qx + qy * qy);
  // air velocity in body
  const vbx = m00 * vax + m10 * vay + m20 * vaz;
  const vby = m01 * vax + m11 * vay + m21 * vaz;
  const vbz = m02 * vax + m12 * vay + m22 * vaz;
  const V = Math.sqrt(vax * vax + vay * vay + vaz * vaz);
  const mach = V / atm.a;
  const qbar = 0.5 * rho * V * V;
  const S = cfg.geometry.reference_area_m2;
  let fax = 0, fay = 0, faz = 0, alpha = 0;
  if (V > 1e-3 && site.aero !== false) {
    const ca = interp(cfg.aero.ca_mach, cfg.aero.ca_value, mach);
    const cna = cfg.aero.cn_alpha_per_rad;
    fax = (-ca * qbar * S * vbx) / V;
    fay = (-cna * qbar * S * vby) / V;
    faz = (-cna * qbar * S * vbz) / V;
    alpha = Math.atan2(Math.hypot(vby, vbz), vbx);
  }
  const arm = cfg.aero.x_cp_m - mp.x_cg;
  const tauAy = -arm * faz, tauAz = arm * fay;

  const eng = stage === 1 ? cfg.engines.stage1 : cfg.engines.stage2;
  let thr = 0;
  let dcx = m00, dcy = m10, dcz = m20; // body x in ECI
  const az = site.launch_azimuth_deg * DEG;
  const saz = Math.sin(az), caz = Math.cos(az);
  const dhx = saz * ex + caz * nx, dhy = saz * ey + caz * ny, dhz = saz * ez + caz * nz;
  if (phase === PAD) {
    const su = eng.startup_s;
    thr = Math.min(Math.max((t + su) / (0.8 * su), 0), 1);
  } else if (phase === S1_BURN || phase === S2_BURN) {
    if (phase === S1_BURN) {
      thr = Math.min(1, 1 - (qbar - gd.q_limit_pa) / gd.q_band_pa);
    } else {
      thr = Math.min(Math.max((t - ctx.t_ses) / eng.startup_s, 0), 1);
    }
    const n = eng.count;
    const fUnit = n * (eng.mdot0_kg_s * eng.ve_m_s + eng.pe_pa * eng.ae_m2);
    const thrG = (gd.g_limit * G0 * mp.m + n * P * eng.ae_m2) / fUnit;
    thr = Math.min(thr, thrG);
    if (phase === S1_BURN || t - ctx.t_ses > eng.startup_s) thr = Math.max(thr, eng.throttle_min);
    if (phase === S1_BURN) {
      const tv = gd.t_vertical_s, tk = gd.kick_duration_s, kick = gd.kick_angle_deg * DEG;
      if (t < tv) { dcx = ux; dcy = uy; dcz = uz; }
      else if (t < tv + tk) {
        const th = kick * smoothstep((t - tv) / tk);
        const c = Math.cos(th), s = Math.sin(th);
        dcx = c * ux + s * dhx; dcy = c * uy + s * dhy; dcz = c * uz + s * dhz;
      } else {
        const c = Math.cos(kick), s = Math.sin(kick);
        const dkx = c * ux + s * dhx, dky = c * uy + s * dhy, dkz = c * uz + s * dhz;
        const vrn = Math.sqrt(vrx * vrx + vry * vry + vrz * vrz);
        const dvx = vrn > 1e-12 ? vrx / vrn : 0, dvy = vrn > 1e-12 ? vry / vrn : 0, dvz = vrn > 1e-12 ? vrz / vrn : 0;
        const b = smoothstep((t - tv - tk) / gd.blend_s);
        let x = (1 - b) * dkx + b * dvx, yy = (1 - b) * dky + b * dvy, zz = (1 - b) * dkz + b * dvz;
        const nn = Math.sqrt(x * x + yy * yy + zz * zz);
        if (nn > 1e-12) { x /= nn; yy /= nn; zz /= nn; } else { x = 0; yy = 0; zz = 0; }
        dcx = x; dcy = yy; dcz = zz;
      }
    }
  }
  let F = 0, mdot = 0;
  if (thr > 0) {
    mdot = eng.count * thr * eng.mdot0_kg_s;
    F = Math.max(eng.count * (thr * eng.mdot0_kg_s * eng.ve_m_s + (thr * eng.pe_pa - P) * eng.ae_m2), 0);
  }
  if (phase === COAST_S2 || phase === S2_BURN) {
    const vzu = vx * ux + vy * uy + vz * uz;
    let hx = vx - vzu * ux, hy = vy - vzu * uy, hz = vz - vzu * uz;
    const vh = Math.sqrt(hx * hx + hy * hy + hz * hz);
    if (vh > 1e-12) { hx /= vh; hy /= vh; hz /= vh; } else { hx = 0; hy = 0; hz = 0; }
    const rt = RE + gd.target_alt_m;
    const wn = gd.s2_wn, ze = gd.s2_zeta;
    const azc = wn * wn * (rt - rn) - 2 * ze * wn * vzu;
    const acc = phase === S2_BURN && F > 0 ? Math.max(F / mp.m, 0.1) : 30.0;
    let s = (azc + MU / (rn * rn) - (vh * vh) / rn) / acc;
    s = Math.min(Math.max(s, gd.s2_sin_min), gd.s2_sin_max);
    const c = Math.sqrt(1 - s * s);
    dcx = c * hx + s * ux; dcy = c * hy + s * uy; dcz = c * hz + s * uz;
  } else if (phase === ORBIT || phase === S2_DEPLETED) {
    const vn = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (vn > 1e-12) { dcx = vx / vn; dcy = vy / vn; dcz = vz / vn; } else { dcx = 0; dcy = 0; dcz = 0; }
  }
  // attitude control
  const dbx = m00 * dcx + m10 * dcy + m20 * dcz;
  const dby = m01 * dcx + m11 * dcy + m21 * dcz;
  const dbz = m02 * dcx + m12 * dcy + m22 * dcz;
  const axy = -dbz, axz = dby;
  const sn = Math.hypot(axy, axz);
  const ang = Math.atan2(sn, dbx);
  let ey_ = 0, ez_ = 0;
  if (sn > 1e-12) { ey_ = (axy / sn) * ang; ez_ = (axz / sn) * ang; }
  const wn = gd.attitude_wn, ze = gd.attitude_zeta;
  let aly = wn * wn * ey_ - 2 * ze * wn * wy;
  let alz = wn * wn * ez_ - 2 * ze * wn * wz;
  let tx = 0, ty = 0, tz = 0;
  let tbx = 0, tby = 0, tbz = 0, gimY = 0, gimZ = 0;
  if (F > 1.0) {
    const L = mp.x_cg - eng.x_gimbal;
    const tyr = mp.iyy * aly - tauAy;
    const tzr = mp.iyy * alz - tauAz;
    let dyy = -tzr / (L * F);
    let dzz = tyr / (L * F);
    const dm = Math.hypot(dyy, dzz);
    const lim = Math.tan(eng.gimbal_max_deg * DEG);
    if (dm > lim) { dyy = (dyy * lim) / dm; dzz = (dzz * lim) / dm; }
    const nrm = Math.sqrt(1 + dyy * dyy + dzz * dzz);
    tbx = F / nrm; tby = (F * dyy) / nrm; tbz = (F * dzz) / nrm;
    ty = L * tbz; tz = -L * tby;
    gimY = dyy; gimZ = dzz;
  } else if (stage === 2 && phase !== PAD) {
    const amax = gd.rcs_alpha_max;
    const am = Math.hypot(aly, alz);
    if (am > amax) { aly = (aly * amax) / am; alz = (alz * amax) / am; }
    ty = mp.iyy * aly; tz = mp.iyy * alz;
  }
  ty += tauAy; tz += tauAz;

  let ox: number, oy: number, oz: number;
  let accx: number, accy: number, accz: number;
  if (phase === PAD) {
    const vrrx = -wE * ry, vrry = wE * rx; // wE x r
    dy[0] = vrrx; dy[1] = vrry; dy[2] = 0;
    accx = -wE * vrry; accy = wE * vrrx; accz = 0;
    dy[3] = accx; dy[4] = accy; dy[5] = accz;
    // Earth rate in body frame: R^T (0,0,wE)
    ox = m20 * wE; oy = m21 * wE; oz = m22 * wE;
    dy[10] = 0; dy[11] = 0; dy[12] = 0;
  } else {
    const fbx = tbx + fax, fby = tby + fay, fbz = tbz + faz;
    const fix = m00 * fbx + m01 * fby + m02 * fbz;
    const fiy = m10 * fbx + m11 * fby + m12 * fbz;
    const fiz = m20 * fbx + m21 * fby + m22 * fbz;
    accx = fix / mp.m + gx; accy = fiy / mp.m + gy; accz = fiz / mp.m + gz;
    dy[0] = vx; dy[1] = vy; dy[2] = vz;
    dy[3] = accx; dy[4] = accy; dy[5] = accz;
    ox = wx; oy = wy; oz = wz;
    const I0 = mp.ixx, I1 = mp.iyy, I2 = mp.iyy;
    const Iw0 = I0 * wx, Iw1 = I1 * wy, Iw2 = I2 * wz;
    const g0 = wy * Iw2 - wz * Iw1, g1 = wz * Iw0 - wx * Iw2, g2 = wx * Iw1 - wy * Iw0;
    dy[10] = (tx - g0) / I0; dy[11] = (ty - g1) / I1; dy[12] = (tz - g2) / I2;
  }
  // quaternion kinematics using the raw (unnormalised) q as in the reference
  const Qw = y[6], Qx = y[7], Qy = y[8], Qz = y[9];
  dy[6] = 0.5 * (-Qx * ox - Qy * oy - Qz * oz);
  dy[7] = 0.5 * (Qw * ox + Qy * oz - Qz * oy);
  dy[8] = 0.5 * (Qw * oy - Qx * oz + Qz * ox);
  dy[9] = 0.5 * (Qw * oz + Qx * oy - Qy * ox);
  dy[13] = stage === 1 ? -mdot : 0;
  dy[14] = stage === 2 ? -mdot : 0;
  if (tel) {
    tel.t = t; tel.phase = phase; tel.h = h; tel.rn = rn; tel.T = atm.T; tel.P = P; tel.rho = rho; tel.a = atm.a;
    tel.V_air = V; tel.mach = mach; tel.qbar = qbar; tel.thrust = F; tel.mdot = mdot; tel.throttle = thr;
    tel.mass = mp.m; tel.x_cg = mp.x_cg; tel.ixx = mp.ixx; tel.iyy = mp.iyy; tel.alpha = alpha;
    const sx = accx - gx, sy = accy - gy, sz = accz - gz;
    tel.g_load = Math.sqrt(sx * sx + sy * sy + sz * sz) / G0;
    tel.v_inertial = Math.sqrt(vx * vx + vy * vy + vz * vz);
    tel.v_rel = Math.sqrt(vrx * vrx + vry * vry + vrz * vrz);
    tel.gimbal_y = gimY; tel.gimbal_z = gimZ;
    tel.fa[0] = fax; tel.fa[1] = fay; tel.fa[2] = faz;
    tel.thrust_b[0] = tbx; tel.thrust_b[1] = tby; tel.thrust_b[2] = tbz;
    tel.acc[0] = accx; tel.acc[1] = accy; tel.acc[2] = accz;
    tel.grav[0] = gx; tel.grav[1] = gy; tel.grav[2] = gz;
    tel.v_air_eci[0] = vax; tel.v_air_eci[1] = vay; tel.v_air_eci[2] = vaz;
    tel.d_cmd[0] = dcx; tel.d_cmd[1] = dcy; tel.d_cmd[2] = dcz;
  }
}

export function initialState(cfg: VehicleConfig): Float64Array {
  const s = cfg.site;
  const lat = s.latitude_deg * DEG, lon = s.longitude_deg * DEG;
  const rr = RE + s.pad_altitude_m;
  const r = [rr * Math.cos(lat) * Math.cos(lon), rr * Math.cos(lat) * Math.sin(lon), rr * Math.sin(lat)];
  const wE = s.rotating_earth ? OMEGA_E : 0;
  const v = [-wE * r[1], wE * r[0], 0];
  const rn = Math.hypot(r[0], r[1], r[2]);
  const up = [r[0] / rn, r[1] / rn, r[2] / rn];
  let east = [-up[1], up[0], 0];
  const en = Math.hypot(east[0], east[1]);
  east = [east[0] / en, east[1] / en, 0];
  const north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
  const az = s.launch_azimuth_deg * DEG;
  const bx = up;
  const by = [0, 1, 2].map((i) => Math.sin(az) * east[i] + Math.cos(az) * north[i]);
  const bz = [bx[1] * by[2] - bx[2] * by[1], bx[2] * by[0] - bx[0] * by[2], bx[0] * by[1] - bx[1] * by[0]];
  const m = [[bx[0], by[0], bz[0]], [bx[1], by[1], bz[1]], [bx[2], by[2], bz[2]]];
  const q = matToQuat(m);
  const wb = [m[2][0] * wE, m[2][1] * wE, m[2][2] * wE];
  return new Float64Array([...r, ...v, ...q, ...wb, cfg.mass.stage1_prop_kg, cfg.mass.stage2_prop_kg]);
}

export function matToQuat(m: number[][]): number[] {
  const tr = m[0][0] + m[1][1] + m[2][2];
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s];
  }
  if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s];
  }
  if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s];
  }
  const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
  return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s];
}
