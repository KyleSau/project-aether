//! Project Aether — real-time 6-DOF ascent integrator compiled to WebAssembly.
//! Line-by-line port of frontend/src/physics/{atmosphere,dynamics,sim}.ts, which mirror the
//! authoritative Python reference (physics/aether_physics). Keep all three in sync.

use serde::Deserialize;
use wasm_bindgen::prelude::*;

pub const MU: f64 = 3.986004418e14;
pub const RE: f64 = 6378137.0;
pub const J2: f64 = 1.08262668e-3;
pub const OMEGA_E: f64 = 7.292115e-5;
pub const G0: f64 = 9.80665;
const DEG: f64 = std::f64::consts::PI / 180.0;

pub const PAD: u32 = 0;
pub const S1_BURN: u32 = 1;
pub const COAST_SEP: u32 = 2;
pub const COAST_S2: u32 = 3;
pub const S2_BURN: u32 = 4;
pub const ORBIT: u32 = 5;
pub const S2_DEPLETED: u32 = 6;
pub const NSTATE: usize = 15;

// ----------------------------------------------------------------- config
#[derive(Deserialize, Clone)]
pub struct Site { pub latitude_deg: f64, pub longitude_deg: f64, pub pad_altitude_m: f64, pub launch_azimuth_deg: f64, pub rotating_earth: bool, pub j2: bool, #[serde(default = "yes")] pub aero: bool }
fn yes() -> bool { true }
#[derive(Deserialize, Clone)]
pub struct Seg { pub x0: f64, pub x1: f64, #[serde(default)] pub tank_x0: f64, #[serde(default)] pub tank_x1: f64, #[serde(default)] pub radius_m: f64 }
#[derive(Deserialize, Clone)]
pub struct Geometry { pub radius_m: f64, pub reference_area_m2: f64, pub stage1: Seg, pub stage2: Seg, pub payload: Seg, pub fairing: Seg }
#[derive(Deserialize, Clone)]
pub struct Mass { pub stage1_dry_kg: f64, pub stage1_prop_kg: f64, pub stage2_dry_kg: f64, pub stage2_prop_kg: f64, pub payload_kg: f64, pub fairing_kg: f64 }
#[derive(Deserialize, Clone)]
pub struct Engine { pub count: f64, pub mdot0_kg_s: f64, pub ve_m_s: f64, pub pe_pa: f64, pub ae_m2: f64, pub throttle_min: f64, pub gimbal_max_deg: f64, pub x_gimbal: f64, pub startup_s: f64 }
#[derive(Deserialize, Clone)]
pub struct Engines { pub stage1: Engine, pub stage2: Engine }
#[derive(Deserialize, Clone)]
pub struct Aero { pub ca_mach: Vec<f64>, pub ca_value: Vec<f64>, pub cn_alpha_per_rad: f64, pub x_cp_m: f64 }
#[derive(Deserialize, Clone)]
pub struct Wind { pub alt_m: Vec<f64>, pub speed_m_s: Vec<f64> }
#[derive(Deserialize, Clone)]
pub struct Guidance {
    pub t_vertical_s: f64, pub kick_duration_s: f64, pub kick_angle_deg: f64, pub blend_s: f64,
    pub q_limit_pa: f64, pub q_band_pa: f64, pub g_limit: f64, pub attitude_wn: f64, pub attitude_zeta: f64,
    pub rcs_alpha_max: f64, pub sep_delay_s: f64, pub ses_delay_s: f64, pub fairing_alt_m: f64,
    pub target_alt_m: f64, pub s2_wn: f64, pub s2_zeta: f64, pub s2_sin_min: f64, pub s2_sin_max: f64,
}
#[derive(Deserialize, Clone)]
pub struct Config { pub site: Site, pub geometry: Geometry, pub mass: Mass, pub engines: Engines, pub aero: Aero, pub wind: Wind, pub guidance: Guidance }

#[derive(Clone, Copy, Default)]
pub struct Ctx { pub phase: u32, pub fairing: bool, pub t_meco: f64, pub t_sep: f64, pub t_ses: f64, pub t_seco: f64 }

// ----------------------------------------------------------------- atmosphere (USSA-1976)
const R_AIR: f64 = 8314.32 / 28.9644;
const R0: f64 = 6356766.0;
const LAYERS: [(f64, f64, f64); 8] = [
    (0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001), (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028), (71000.0, 214.65, -0.002), (84852.0, 186.946, 0.0),
];
const UPPER: [(f64, f64, f64, f64); 13] = [
    (86000.0, 186.87, 3.7338e-1, 6.958e-6), (90000.0, 186.87, 1.8359e-1, 3.416e-6), (100000.0, 195.08, 3.2011e-2, 5.604e-7),
    (110000.0, 240.0, 7.1042e-3, 9.708e-8), (120000.0, 360.0, 2.5382e-3, 2.222e-8), (150000.0, 634.39, 4.5422e-4, 2.076e-9),
    (200000.0, 854.56, 8.4736e-5, 2.541e-10), (300000.0, 976.01, 8.7704e-6, 1.916e-11), (400000.0, 995.83, 1.4518e-6, 2.803e-12),
    (500000.0, 999.24, 3.0236e-7, 5.215e-13), (600000.0, 999.85, 8.213e-8, 1.137e-13), (800000.0, 999.99, 1.7036e-8, 1.136e-14),
    (1000000.0, 1000.0, 7.5138e-9, 3.561e-15),
];

pub struct Atmo { pub t: f64, pub p: f64, pub rho: f64, pub a: f64, pb: [f64; 8] }

impl Atmo {
    pub fn new() -> Self {
        let mut pb = [0.0; 8];
        pb[0] = 101325.0;
        for i in 0..7 {
            let (hb, tb, lb) = LAYERS[i];
            let h1 = LAYERS[i + 1].0;
            pb[i + 1] = if lb == 0.0 { pb[i] * (-G0 * (h1 - hb) / (R_AIR * tb)).exp() } else {
                let t1 = tb + lb * (h1 - hb);
                pb[i] * (t1 / tb).powf(-G0 / (R_AIR * lb))
            };
        }
        Atmo { t: 0.0, p: 0.0, rho: 0.0, a: 0.0, pb }
    }
    pub fn eval(&mut self, z: f64) {
        let z = if z < 0.0 { 0.0 } else { z };
        if z <= 86000.0 {
            let h = R0 * z / (R0 + z);
            let mut i = 7;
            while i > 0 && h < LAYERS[i].0 { i -= 1; }
            let (hb, tb, lb) = LAYERS[i];
            let (t, p) = if lb == 0.0 { (tb, self.pb[i] * (-G0 * (h - hb) / (R_AIR * tb)).exp()) } else {
                let t = tb + lb * (h - hb);
                (t, self.pb[i] * (t / tb).powf(-G0 / (R_AIR * lb)))
            };
            self.t = t; self.p = p; self.rho = p / (R_AIR * t); self.a = (1.4 * R_AIR * t).sqrt();
            return;
        }
        let n = UPPER.len();
        let k = if z >= UPPER[n - 1].0 { n - 2 } else { let mut k = 0; while UPPER[k + 1].0 < z { k += 1; } k };
        let (z0, t0, p0, r0) = UPPER[k];
        let (z1, t1, p1, r1) = UPPER[k + 1];
        let f = (z - z0) / (z1 - z0);
        self.t = t0 + f * (t1 - t0);
        self.p = (p0.ln() + f * (p1.ln() - p0.ln())).exp();
        self.rho = (r0.ln() + f * (r1.ln() - r0.ln())).exp();
        self.a = (1.4 * R_AIR * self.t).sqrt();
    }
}

fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] { return ys[0]; }
    if x >= xs[n - 1] { return ys[n - 1]; }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 { let mid = (lo + hi) >> 1; if xs[mid] <= x { lo = mid } else { hi = mid } }
    let f = (x - xs[lo]) / (xs[lo + 1] - xs[lo]);
    ys[lo] + f * (ys[lo + 1] - ys[lo])
}

pub struct MassProps { pub m: f64, pub x_cg: f64, pub ixx: f64, pub iyy: f64 }

fn mass_props(cfg: &Config, mp1: f64, mp2: f64, stage: u32, fairing: bool) -> MassProps {
    let g = &cfg.geometry; let ms = &cfg.mass; let r = g.radius_m;
    let mut c = [[0.0f64; 4]; 6];
    let mut n = 0usize;
    let mut push = |m: f64, x0: f64, x1: f64, rr: f64, shell: bool| {
        let l = (x1 - x0).max(1e-6);
        c[n] = [m, 0.5 * (x0 + x1), if shell { m * rr * rr } else { 0.5 * m * rr * rr },
            if shell { m * (rr * rr / 2.0 + l * l / 12.0) } else { m * (3.0 * rr * rr + l * l) / 12.0 }];
        n += 1;
    };
    if stage == 1 {
        let s1 = &g.stage1;
        push(ms.stage1_dry_kg, s1.x0, s1.x1, r, true);
        if mp1 > 0.0 { let f = mp1 / ms.stage1_prop_kg; push(mp1, s1.tank_x0, s1.tank_x0 + f * (s1.tank_x1 - s1.tank_x0), r * 0.98, false); }
    }
    let s2 = &g.stage2;
    push(ms.stage2_dry_kg, s2.x0, s2.x1, r, true);
    if mp2 > 0.0 { let f = mp2 / ms.stage2_prop_kg; push(mp2, s2.tank_x0, s2.tank_x0 + f * (s2.tank_x1 - s2.tank_x0), r * 0.98, false); }
    push(ms.payload_kg, g.payload.x0, g.payload.x1, g.payload.radius_m, false);
    if fairing { push(ms.fairing_kg, g.fairing.x0, g.fairing.x1, g.fairing.radius_m, true); }
    let (mut m, mut mx, mut ixx) = (0.0, 0.0, 0.0);
    for k in 0..n { m += c[k][0]; mx += c[k][0] * c[k][1]; ixx += c[k][2]; }
    let xcg = mx / m;
    let mut iyy = 0.0;
    for k in 0..n { let d = c[k][1] - xcg; iyy += c[k][3] + c[k][0] * d * d; }
    MassProps { m, x_cg: xcg, ixx, iyy }
}

#[inline] fn smoothstep(x: f64) -> f64 { if x <= 0.0 { 0.0 } else if x >= 1.0 { 1.0 } else { x * x * (3.0 - 2.0 * x) } }
#[inline] fn stage_of(phase: u32) -> u32 { if phase <= COAST_SEP { 1 } else { 2 } }

/// Derivative of the 15-element state (see dynamics.py for the documented model).
pub fn evaluate(t: f64, y: &[f64], ctx: &Ctx, cfg: &Config, atm: &mut Atmo, dy: &mut [f64]) {
    let site = &cfg.site; let gd = &cfg.guidance;
    let (rx, ry, rz) = (y[0], y[1], y[2]);
    let (vx, vy, vz) = (y[3], y[4], y[5]);
    let (mut qw, mut qx, mut qy, mut qz) = (y[6], y[7], y[8], y[9]);
    let (wx, wy, wz) = (y[10], y[11], y[12]);
    let mp1 = y[13].max(0.0); let mp2 = y[14].max(0.0);
    let phase = ctx.phase; let stage = stage_of(phase);
    let mp = mass_props(cfg, mp1, mp2, stage, ctx.fairing);
    let rn = (rx * rx + ry * ry + rz * rz).sqrt();
    let h = rn - RE;
    atm.eval(h);
    let p_amb = atm.p; let rho = atm.rho;
    let (ux, uy, uz) = (rx / rn, ry / rn, rz / rn);
    let (mut ex, mut ey) = (-uy, ux);
    let en = ex.hypot(ey);
    if en > 1e-12 { ex /= en; ey /= en; } else { ex = 0.0; ey = 0.0; }
    let ez = 0.0;
    let (nx, ny, nz) = (uy * ez - uz * ey, uz * ex - ux * ez, ux * ey - uy * ex);
    let we = if site.rotating_earth { OMEGA_E } else { 0.0 };
    let rn2 = rn * rn;
    let k = -MU / (rn2 * rn);
    let (gx, gy, gz) = if site.j2 {
        let f = 1.5 * J2 * (RE * RE / rn2);
        let zz = 5.0 * rz * rz / rn2;
        (k * rx * (1.0 - f * (zz - 1.0)), k * ry * (1.0 - f * (zz - 1.0)), k * rz * (1.0 - f * (zz - 3.0)))
    } else { (k * rx, k * ry, k * rz) };
    let (vrx, vry, vrz) = (vx + we * ry, vy - we * rx, vz);
    let ws = interp(&cfg.wind.alt_m, &cfg.wind.speed_m_s, h);
    let (vax, vay, vaz) = (vrx - ws * ex, vry - ws * ey, vrz - ws * ez);
    let qn = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    qw /= qn; qx /= qn; qy /= qn; qz /= qn;
    let m00 = 1.0 - 2.0 * (qy * qy + qz * qz); let m01 = 2.0 * (qx * qy - qw * qz); let m02 = 2.0 * (qx * qz + qw * qy);
    let m10 = 2.0 * (qx * qy + qw * qz); let m11 = 1.0 - 2.0 * (qx * qx + qz * qz); let m12 = 2.0 * (qy * qz - qw * qx);
    let m20 = 2.0 * (qx * qz - qw * qy); let m21 = 2.0 * (qy * qz + qw * qx); let m22 = 1.0 - 2.0 * (qx * qx + qy * qy);
    let vbx = m00 * vax + m10 * vay + m20 * vaz;
    let vby = m01 * vax + m11 * vay + m21 * vaz;
    let vbz = m02 * vax + m12 * vay + m22 * vaz;
    let vmag = (vax * vax + vay * vay + vaz * vaz).sqrt();
    let mach = vmag / atm.a;
    let qbar = 0.5 * rho * vmag * vmag;
    let s_ref = cfg.geometry.reference_area_m2;
    let (mut fax, mut fay, mut faz) = (0.0, 0.0, 0.0);
    if vmag > 1e-3 && site.aero {
        let ca = interp(&cfg.aero.ca_mach, &cfg.aero.ca_value, mach);
        let cna = cfg.aero.cn_alpha_per_rad;
        fax = -ca * qbar * s_ref * vbx / vmag;
        fay = -cna * qbar * s_ref * vby / vmag;
        faz = -cna * qbar * s_ref * vbz / vmag;
    }
    let arm = cfg.aero.x_cp_m - mp.x_cg;
    let tau_ay = -arm * faz; let tau_az = arm * fay;
    let eng = if stage == 1 { &cfg.engines.stage1 } else { &cfg.engines.stage2 };
    let mut thr = 0.0;
    let (mut dcx, mut dcy, mut dcz) = (m00, m10, m20);
    let az = site.launch_azimuth_deg * DEG;
    let (saz, caz) = (az.sin(), az.cos());
    let (dhx, dhy, dhz) = (saz * ex + caz * nx, saz * ey + caz * ny, saz * ez + caz * nz);
    if phase == PAD {
        let su = eng.startup_s;
        thr = ((t + su) / (0.8 * su)).max(0.0).min(1.0);
    } else if phase == S1_BURN || phase == S2_BURN {
        if phase == S1_BURN { thr = (1.0f64).min(1.0 - (qbar - gd.q_limit_pa) / gd.q_band_pa); }
        else { thr = ((t - ctx.t_ses) / eng.startup_s).max(0.0).min(1.0); }
        let n = eng.count;
        let f_unit = n * (eng.mdot0_kg_s * eng.ve_m_s + eng.pe_pa * eng.ae_m2);
        let thr_g = (gd.g_limit * G0 * mp.m + n * p_amb * eng.ae_m2) / f_unit;
        thr = thr.min(thr_g);
        if phase == S1_BURN || t - ctx.t_ses > eng.startup_s { thr = thr.max(eng.throttle_min); }
        if phase == S1_BURN {
            let (tv, tk, kick) = (gd.t_vertical_s, gd.kick_duration_s, gd.kick_angle_deg * DEG);
            if t < tv { dcx = ux; dcy = uy; dcz = uz; }
            else if t < tv + tk {
                let th = kick * smoothstep((t - tv) / tk);
                let (c, s) = (th.cos(), th.sin());
                dcx = c * ux + s * dhx; dcy = c * uy + s * dhy; dcz = c * uz + s * dhz;
            } else {
                let (c, s) = (kick.cos(), kick.sin());
                let (dkx, dky, dkz) = (c * ux + s * dhx, c * uy + s * dhy, c * uz + s * dhz);
                let vrn = (vrx * vrx + vry * vry + vrz * vrz).sqrt();
                let (dvx, dvy, dvz) = if vrn > 1e-12 { (vrx / vrn, vry / vrn, vrz / vrn) } else { (0.0, 0.0, 0.0) };
                let b = smoothstep((t - tv - tk) / gd.blend_s);
                let (mut x, mut yy, mut zz) = ((1.0 - b) * dkx + b * dvx, (1.0 - b) * dky + b * dvy, (1.0 - b) * dkz + b * dvz);
                let nn = (x * x + yy * yy + zz * zz).sqrt();
                if nn > 1e-12 { x /= nn; yy /= nn; zz /= nn; } else { x = 0.0; yy = 0.0; zz = 0.0; }
                dcx = x; dcy = yy; dcz = zz;
            }
        }
    }
    let (mut f_thr, mut mdot) = (0.0, 0.0);
    if thr > 0.0 {
        mdot = eng.count * thr * eng.mdot0_kg_s;
        f_thr = (eng.count * (thr * eng.mdot0_kg_s * eng.ve_m_s + (thr * eng.pe_pa - p_amb) * eng.ae_m2)).max(0.0);
    }
    if phase == COAST_S2 || phase == S2_BURN {
        let vzu = vx * ux + vy * uy + vz * uz;
        let (mut hx, mut hy, mut hz) = (vx - vzu * ux, vy - vzu * uy, vz - vzu * uz);
        let vh = (hx * hx + hy * hy + hz * hz).sqrt();
        if vh > 1e-12 { hx /= vh; hy /= vh; hz /= vh; } else { hx = 0.0; hy = 0.0; hz = 0.0; }
        let rt = RE + gd.target_alt_m;
        let (wn, ze) = (gd.s2_wn, gd.s2_zeta);
        let azc = wn * wn * (rt - rn) - 2.0 * ze * wn * vzu;
        let acc = if phase == S2_BURN && f_thr > 0.0 { (f_thr / mp.m).max(0.1) } else { 30.0 };
        let s = ((azc + MU / (rn * rn) - vh * vh / rn) / acc).max(gd.s2_sin_min).min(gd.s2_sin_max);
        let c = (1.0 - s * s).sqrt();
        dcx = c * hx + s * ux; dcy = c * hy + s * uy; dcz = c * hz + s * uz;
    } else if phase == ORBIT || phase == S2_DEPLETED {
        let vn = (vx * vx + vy * vy + vz * vz).sqrt();
        if vn > 1e-12 { dcx = vx / vn; dcy = vy / vn; dcz = vz / vn; } else { dcx = 0.0; dcy = 0.0; dcz = 0.0; }
    }
    let dbx = m00 * dcx + m10 * dcy + m20 * dcz;
    let dby = m01 * dcx + m11 * dcy + m21 * dcz;
    let dbz = m02 * dcx + m12 * dcy + m22 * dcz;
    let (axy, axz) = (-dbz, dby);
    let sn = axy.hypot(axz);
    let ang = sn.atan2(dbx);
    let (ey_, ez_) = if sn > 1e-12 { (axy / sn * ang, axz / sn * ang) } else { (0.0, 0.0) };
    let (wn, ze) = (gd.attitude_wn, gd.attitude_zeta);
    let mut aly = wn * wn * ey_ - 2.0 * ze * wn * wy;
    let mut alz = wn * wn * ez_ - 2.0 * ze * wn * wz;
    let tx = 0.0; let mut ty = 0.0; let mut tz = 0.0;
    let (mut tbx, mut tby, mut tbz) = (0.0, 0.0, 0.0);
    if f_thr > 1.0 {
        let l = mp.x_cg - eng.x_gimbal;
        let tyr = mp.iyy * aly - tau_ay;
        let tzr = mp.iyy * alz - tau_az;
        let mut dyy = -tzr / (l * f_thr);
        let mut dzz = tyr / (l * f_thr);
        let dm = dyy.hypot(dzz);
        let lim = (eng.gimbal_max_deg * DEG).tan();
        if dm > lim { dyy = dyy * lim / dm; dzz = dzz * lim / dm; }
        let nrm = (1.0 + dyy * dyy + dzz * dzz).sqrt();
        tbx = f_thr / nrm; tby = f_thr * dyy / nrm; tbz = f_thr * dzz / nrm;
        ty = l * tbz; tz = -l * tby;
    } else if stage == 2 && phase != PAD {
        let amax = gd.rcs_alpha_max;
        let am = aly.hypot(alz);
        if am > amax { aly = aly * amax / am; alz = alz * amax / am; }
        ty = mp.iyy * aly; tz = mp.iyy * alz;
    }
    ty += tau_ay; tz += tau_az;
    let (ox, oy, oz);
    if phase == PAD {
        let (vrrx, vrry) = (-we * ry, we * rx);
        dy[0] = vrrx; dy[1] = vrry; dy[2] = 0.0;
        dy[3] = -we * vrry; dy[4] = we * vrrx; dy[5] = 0.0;
        ox = m20 * we; oy = m21 * we; oz = m22 * we;
        dy[10] = 0.0; dy[11] = 0.0; dy[12] = 0.0;
    } else {
        let (fbx, fby, fbz) = (tbx + fax, tby + fay, tbz + faz);
        let fix = m00 * fbx + m01 * fby + m02 * fbz;
        let fiy = m10 * fbx + m11 * fby + m12 * fbz;
        let fiz = m20 * fbx + m21 * fby + m22 * fbz;
        dy[0] = vx; dy[1] = vy; dy[2] = vz;
        dy[3] = fix / mp.m + gx; dy[4] = fiy / mp.m + gy; dy[5] = fiz / mp.m + gz;
        ox = wx; oy = wy; oz = wz;
        let (i0, i1, i2) = (mp.ixx, mp.iyy, mp.iyy);
        let (iw0, iw1, iw2) = (i0 * wx, i1 * wy, i2 * wz);
        let (g0, g1, g2) = (wy * iw2 - wz * iw1, wz * iw0 - wx * iw2, wx * iw1 - wy * iw0);
        dy[10] = (tx - g0) / i0; dy[11] = (ty - g1) / i1; dy[12] = (tz - g2) / i2;
    }
    let (qw0, qx0, qy0, qz0) = (y[6], y[7], y[8], y[9]);
    dy[6] = 0.5 * (-qx0 * ox - qy0 * oy - qz0 * oz);
    dy[7] = 0.5 * (qw0 * ox + qy0 * oz - qz0 * oy);
    dy[8] = 0.5 * (qw0 * oy - qx0 * oz + qz0 * ox);
    dy[9] = 0.5 * (qw0 * oz + qx0 * oy - qy0 * ox);
    dy[13] = if stage == 1 { -mdot } else { 0.0 };
    dy[14] = if stage == 2 { -mdot } else { 0.0 };
}

pub struct Rk4 { k1: [f64; NSTATE], k2: [f64; NSTATE], k3: [f64; NSTATE], k4: [f64; NSTATE], tmp: [f64; NSTATE], pub atm: Atmo }

impl Rk4 {
    pub fn new() -> Self { Rk4 { k1: [0.0; NSTATE], k2: [0.0; NSTATE], k3: [0.0; NSTATE], k4: [0.0; NSTATE], tmp: [0.0; NSTATE], atm: Atmo::new() } }
    pub fn step(&mut self, t: f64, y: &mut [f64], ctx: &Ctx, cfg: &Config, h: f64) {
        evaluate(t, y, ctx, cfg, &mut self.atm, &mut self.k1);
        for i in 0..NSTATE { self.tmp[i] = y[i] + 0.5 * h * self.k1[i]; }
        let tmp = self.tmp;
        evaluate(t + 0.5 * h, &tmp, ctx, cfg, &mut self.atm, &mut self.k2);
        for i in 0..NSTATE { self.tmp[i] = y[i] + 0.5 * h * self.k2[i]; }
        let tmp = self.tmp;
        evaluate(t + 0.5 * h, &tmp, ctx, cfg, &mut self.atm, &mut self.k3);
        for i in 0..NSTATE { self.tmp[i] = y[i] + h * self.k3[i]; }
        let tmp = self.tmp;
        evaluate(t + h, &tmp, ctx, cfg, &mut self.atm, &mut self.k4);
        for i in 0..NSTATE { y[i] += h / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]); }
        let n = (y[6] * y[6] + y[7] * y[7] + y[8] * y[8] + y[9] * y[9]).sqrt();
        y[6] /= n; y[7] /= n; y[8] /= n; y[9] /= n;
    }
}

fn mat_to_quat(m: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    if tr > 0.0 { let s = (tr + 1.0).sqrt() * 2.0; return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]; }
    if m[0][0] > m[1][1] && m[0][0] > m[2][2] { let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0; return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]; }
    if m[1][1] > m[2][2] { let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0; return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]; }
    let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
    [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
}

pub fn initial_state(cfg: &Config) -> [f64; NSTATE] {
    let s = &cfg.site;
    let (lat, lon) = (s.latitude_deg * DEG, s.longitude_deg * DEG);
    let rr = RE + s.pad_altitude_m;
    let r = [rr * lat.cos() * lon.cos(), rr * lat.cos() * lon.sin(), rr * lat.sin()];
    let we = if s.rotating_earth { OMEGA_E } else { 0.0 };
    let v = [-we * r[1], we * r[0], 0.0];
    let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let up = [r[0] / rn, r[1] / rn, r[2] / rn];
    let en = up[1].hypot(up[0]);
    let east = [-up[1] / en, up[0] / en, 0.0];
    let north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
    let az = s.launch_azimuth_deg * DEG;
    let bx = up;
    let by = [az.sin() * east[0] + az.cos() * north[0], az.sin() * east[1] + az.cos() * north[1], az.sin() * east[2] + az.cos() * north[2]];
    let bz = [bx[1] * by[2] - bx[2] * by[1], bx[2] * by[0] - bx[0] * by[2], bx[0] * by[1] - bx[1] * by[0]];
    let m = [[bx[0], by[0], bz[0]], [bx[1], by[1], bz[1]], [bx[2], by[2], bz[2]]];
    let q = mat_to_quat(m);
    [r[0], r[1], r[2], v[0], v[1], v[2], q[0], q[1], q[2], q[3], m[2][0] * we, m[2][1] * we, m[2][2] * we, cfg.mass.stage1_prop_kg, cfg.mass.stage2_prop_kg]
}

fn energy(y: &[f64]) -> f64 { 0.5 * (y[3] * y[3] + y[4] * y[4] + y[5] * y[5]) - MU / (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt() }
fn alt(y: &[f64]) -> f64 { (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt() - RE }

#[derive(Clone, Copy, PartialEq)]
enum Ev { Meco, Fairing, Depl, Seco, Crash }

/// Full mission simulator (same event logic as frontend/src/physics/sim.ts).
#[wasm_bindgen]
pub struct WasmSim {
    cfg: Config, rk: Rk4, h: f64, t: f64, y: [f64; NSTATE], ctx: Ctx, lost: bool,
    events: Vec<(String, f64)>, steps: u64, eps_target: f64,
}

impl WasmSim {
    fn ev_value(&self, e: Ev, y: &[f64]) -> f64 {
        match e { Ev::Meco => y[13], Ev::Fairing => alt(y) - self.cfg.guidance.fairing_alt_m, Ev::Depl => y[14], Ev::Seco => energy(y) - self.eps_target, Ev::Crash => alt(y) + 5.0 }
    }
    fn ev_dir(e: Ev) -> f64 { match e { Ev::Meco | Ev::Depl | Ev::Crash => -1.0, _ => 1.0 } }
    fn active(&self, out: &mut Vec<Ev>) {
        out.clear();
        let p = self.ctx.phase;
        if p == S1_BURN { out.push(Ev::Meco); }
        if self.ctx.fairing && (p == COAST_S2 || p == S2_BURN) { out.push(Ev::Fairing); }
        if p == S2_BURN { out.push(Ev::Depl); out.push(Ev::Seco); }
        if p != PAD && p != ORBIT { out.push(Ev::Crash); }
    }
    fn emit(&mut self, n: &str) { self.events.push((n.to_string(), self.t)); }
    fn handle(&mut self, e: Ev) {
        match e {
            Ev::Meco => { self.y[13] = 0.0; self.ctx.t_meco = self.t; self.ctx.phase = COAST_SEP; self.emit("MECO"); }
            Ev::Fairing => { self.ctx.fairing = false; self.emit("FAIRING_SEP"); }
            Ev::Depl => { self.y[14] = 0.0; self.ctx.phase = S2_DEPLETED; self.ctx.t_seco = self.t; self.emit("S2_DEPLETION"); }
            Ev::Seco => { self.ctx.phase = ORBIT; self.ctx.t_seco = self.t; self.emit("SECO"); }
            Ev::Crash => { self.lost = true; self.emit("LOSS_OF_VEHICLE"); }
        }
    }
    fn next_sched(&self) -> f64 {
        let g = &self.cfg.guidance;
        match self.ctx.phase { PAD => 0.0, COAST_SEP => self.ctx.t_meco + g.sep_delay_s, COAST_S2 => self.ctx.t_sep + g.ses_delay_s, _ => f64::INFINITY }
    }
    fn on_sched(&mut self) {
        match self.ctx.phase {
            PAD => { self.ctx.phase = S1_BURN; self.emit("LIFTOFF"); }
            COAST_SEP => { self.ctx.phase = COAST_S2; self.ctx.t_sep = self.t; self.emit("STAGE_SEP"); }
            COAST_S2 => { self.ctx.phase = S2_BURN; self.ctx.t_ses = self.t; self.emit("SES"); }
            _ => {}
        }
    }
    fn step_once(&mut self, t_limit: f64, evs: &mut Vec<Ev>) {
        let mut h = self.h.min(t_limit - self.t);
        let ts = self.next_sched();
        let mut hit = false;
        if self.t + h >= ts - 1e-12 { h = ts - self.t; hit = true; }
        if h <= 1e-12 { if hit { self.on_sched(); } return; }
        self.active(evs);
        let mut g0 = [0.0f64; 6];
        for (k, e) in evs.iter().enumerate() { g0[k] = self.ev_value(*e, &self.y); }
        let y_prev = self.y;
        let ctx = self.ctx;
        self.rk.step(self.t, &mut self.y, &ctx, &self.cfg, h);
        let mut first: Option<usize> = None; let mut ffr = 2.0;
        for (k, e) in evs.iter().enumerate() {
            let g1 = self.ev_value(*e, &self.y);
            let d = Self::ev_dir(*e);
            let crossed = if d < 0.0 { g0[k] > 0.0 && g1 <= 0.0 } else { g0[k] < 0.0 && g1 >= 0.0 };
            if crossed { let fr = g0[k] / (g0[k] - g1); if fr < ffr { ffr = fr; first = Some(k); } }
        }
        if let Some(k) = first {
            let e = evs[k];
            let (mut a, mut ga, mut b, mut gb) = (0.0, g0[k], h, self.ev_value(e, &self.y));
            let mut hc = h * ffr;
            for _ in 0..6 {
                let mut yt = y_prev;
                self.rk.step(self.t, &mut yt, &ctx, &self.cfg, hc);
                let gc = self.ev_value(e, &yt);
                if gc.abs() < 1e-9 * (1.0 + ga.abs()) { break; }
                if (gc > 0.0) == (ga > 0.0) { a = hc; ga = gc; } else { b = hc; gb = gc; }
                hc = a + (b - a) * (ga / (ga - gb));
            }
            self.y = y_prev;
            self.rk.step(self.t, &mut self.y, &ctx, &self.cfg, hc);
            self.t += hc;
            self.steps += 1;
            self.handle(e);
        } else {
            self.t = if hit { ts } else { self.t + h };
            self.steps += 1;
            if hit { self.on_sched(); }
        }
    }
}

#[wasm_bindgen]
impl WasmSim {
    #[wasm_bindgen(constructor)]
    pub fn new(cfg_json: &str, h: f64) -> Result<WasmSim, JsValue> {
        let cfg: Config = serde_json::from_str(cfg_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let y = initial_state(&cfg);
        let eps = -MU / (2.0 * (RE + cfg.guidance.target_alt_m));
        let t0 = -cfg.engines.stage1.startup_s;
        let mut s = WasmSim { cfg, rk: Rk4::new(), h, t: t0, y, ctx: Ctx { phase: PAD, fairing: true, ..Default::default() }, lost: false, events: vec![], steps: 0, eps_target: eps };
        s.emit("IGNITION");
        Ok(s)
    }
    /// Advance to t_target. Returns number of RK4 steps taken.
    pub fn advance_to(&mut self, t_target: f64) -> u32 {
        let mut evs: Vec<Ev> = Vec::with_capacity(6);
        let s0 = self.steps;
        let mut n = 0;
        while self.t < t_target - 1e-9 && !self.lost && n < 10_000_000 { self.step_once(t_target, &mut evs); n += 1; }
        (self.steps - s0) as u32
    }
    pub fn t(&self) -> f64 { self.t }
    pub fn phase(&self) -> u32 { self.ctx.phase }
    pub fn fairing(&self) -> bool { self.ctx.fairing }
    pub fn lost(&self) -> bool { self.lost }
    pub fn steps(&self) -> f64 { self.steps as f64 }
    pub fn state(&self) -> Vec<f64> { self.y.to_vec() }
    pub fn ctx_times(&self) -> Vec<f64> { vec![self.ctx.t_meco, self.ctx.t_sep, self.ctx.t_ses, self.ctx.t_seco] }
    pub fn events_json(&self) -> String {
        let parts: Vec<String> = self.events.iter().map(|(n, t)| format!("[\"{}\",{}]", n, t)).collect();
        format!("[{}]", parts.join(","))
    }
}

/// Stand-alone RK4 stepper so the TS MissionSim can delegate its inner loop to WASM.
#[wasm_bindgen]
pub struct WasmStepper { cfg: Config, rk: Rk4 }

#[wasm_bindgen]
impl WasmStepper {
    #[wasm_bindgen(constructor)]
    pub fn new(cfg_json: &str) -> Result<WasmStepper, JsValue> {
        let cfg: Config = serde_json::from_str(cfg_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmStepper { cfg, rk: Rk4::new() })
    }
    pub fn rk4(&mut self, t: f64, y: &mut [f64], phase: u32, fairing: bool, t_ses: f64, h: f64) {
        let ctx = Ctx { phase, fairing, t_ses, ..Default::default() };
        self.rk.step(t, y, &ctx, &self.cfg, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mission_reaches_orbit() {
        let s = std::fs::read_to_string("../data/vehicle.json").unwrap();
        let mut sim = WasmSim::new(&s, 0.02).ok().unwrap();
        sim.advance_to(700.0);
        assert_eq!(sim.phase(), ORBIT);
    }
}
