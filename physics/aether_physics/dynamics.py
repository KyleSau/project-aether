"""6-DOF rigid-body flight dynamics, guidance and attitude control (authoritative reference).

State vector y (15):
  r[0:3]  position, ECI [m]          (z = Earth spin axis, x through launch meridian at t=0)
  v[3:6]  velocity, ECI [m/s]
  q[6:10] attitude quaternion (w,x,y,z), body -> ECI. Body x = vehicle long axis (nose)
  w[10:13] body angular rate [rad/s]
  mp1, mp2 [13], [14]  stage propellant masses [kg]

The same equations are implemented in frontend/src/physics (TypeScript) and
wasm/src (Rust) for real time; tests check agreement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .atmosphere import atmosphere
from .vehicle import mass_props, engine_thrust, ca_of_mach, wind_speed

MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08262668e-3
OMEGA_E = 7.2921150e-5
G0 = 9.80665

PAD, S1_BURN, COAST_SEP, COAST_S2, S2_BURN, ORBIT, S2_DEPLETED = 0, 1, 2, 3, 4, 5, 6
PHASE_NAMES = ["PAD", "S1_BURN", "COAST_SEP", "COAST_S2", "S2_BURN", "ORBIT", "S2_DEPLETED"]


@dataclass
class Ctx:
    phase: int = PAD
    fairing: bool = True
    t_meco: float = 0.0
    t_sep: float = 0.0
    t_ses: float = 0.0
    t_seco: float = 0.0


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a):
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _add(a, b, s=1.0):
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def mat_vec(m, v):
    return (_dot(m[0], v), _dot(m[1], v), _dot(m[2], v))


def mat_t_vec(m, v):
    return (
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    )


def mat_to_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((0.25 * s), (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def gravity(r, use_j2=True):
    x, y, z = r
    rn2 = x * x + y * y + z * z
    rn = math.sqrt(rn2)
    k = -MU / (rn2 * rn)
    if not use_j2:
        return (k * x, k * y, k * z)
    f = 1.5 * J2 * (RE * RE / rn2)
    zz = 5.0 * z * z / rn2
    return (k * x * (1 - f * (zz - 1)), k * y * (1 - f * (zz - 1)), k * z * (1 - f * (zz - 3)))


def smoothstep(x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return x * x * (3 - 2 * x)


def local_frame(r):
    up = _unit(r)
    east = _unit(_cross((0.0, 0.0, 1.0), up))
    north = _cross(up, east)
    return up, east, north


def initial_state(cfg):
    s = cfg["site"]
    lat = math.radians(s["latitude_deg"])
    lon = math.radians(s["longitude_deg"])
    rr = RE + s["pad_altitude_m"]
    r = (rr * math.cos(lat) * math.cos(lon), rr * math.cos(lat) * math.sin(lon), rr * math.sin(lat))
    we = (0.0, 0.0, OMEGA_E if s["rotating_earth"] else 0.0)
    v = _cross(we, r)
    up, east, north = local_frame(r)
    az = math.radians(s["launch_azimuth_deg"])
    dh = _add(_scale(east, math.sin(az)), north, math.cos(az))
    bx, by = up, dh
    bz = _cross(bx, by)
    m = ((bx[0], by[0], bz[0]), (bx[1], by[1], bz[1]), (bx[2], by[2], bz[2]))
    q = mat_to_quat(m)
    wb = mat_t_vec(m, we)
    ms = cfg["mass"]
    return list(r) + list(v) + list(q) + list(wb) + [ms["stage1_prop_kg"], ms["stage2_prop_kg"]]


def stage_of(phase):
    return 1 if phase <= COAST_SEP else 2


def evaluate(t, y, ctx: Ctx, cfg, aux=False):
    """Return dy/dt (list of 15) and optionally a telemetry dict."""
    site = cfg["site"]
    gd = cfg["guidance"]
    r = (y[0], y[1], y[2])
    v = (y[3], y[4], y[5])
    q = (y[6], y[7], y[8], y[9])
    w = (y[10], y[11], y[12])
    mp1 = max(y[13], 0.0)
    mp2 = max(y[14], 0.0)
    phase = ctx.phase
    stage = stage_of(phase)
    mp = mass_props(cfg, mp1, mp2, stage, ctx.fairing)
    rn = _norm(r)
    h = rn - RE
    T, P, rho, a_snd = atmosphere(h)
    up, east, north = local_frame(r)
    we = (0.0, 0.0, OMEGA_E if site["rotating_earth"] else 0.0)
    g = gravity(r, site["j2"])
    v_rel = _add(v, _cross(we, r), -1.0)
    wind = _scale(east, wind_speed(cfg, h))
    v_air = _add(v_rel, wind, -1.0)
    R = quat_to_mat(q)
    vab = mat_t_vec(R, v_air)
    V = _norm(v_air)
    mach = V / a_snd
    qbar = 0.5 * rho * V * V
    S = cfg["geometry"]["reference_area_m2"]
    if V > 1e-3 and site.get("aero", True):
        ca = ca_of_mach(cfg, mach)
        cna = cfg["aero"]["cn_alpha_per_rad"]
        fa = (-ca * qbar * S * vab[0] / V, -cna * qbar * S * vab[1] / V, -cna * qbar * S * vab[2] / V)
        alpha = math.atan2(math.hypot(vab[1], vab[2]), vab[0])
    else:
        fa = (0.0, 0.0, 0.0)
        alpha = 0.0
    arm = cfg["aero"]["x_cp_m"] - mp.x_cg
    tau_aero = (0.0, -arm * fa[2], arm * fa[1])

    eng = cfg["engines"]["stage1" if stage == 1 else "stage2"]
    # ---------------- guidance: throttle & commanded thrust direction (ECI) -------------
    thr = 0.0
    bx = (R[0][0], R[1][0], R[2][0])
    d_cmd = bx
    az = math.radians(site["launch_azimuth_deg"])
    dh = _add(_scale(east, math.sin(az)), north, math.cos(az))
    if phase == PAD:
        su = eng["startup_s"]
        thr = min(max((t + su) / (0.8 * su), 0.0), 1.0)
    elif phase == S1_BURN or phase == S2_BURN:
        if phase == S1_BURN:
            thr = 1.0
            thr_q = 1.0 - (qbar - gd["q_limit_pa"]) / gd["q_band_pa"]
            thr = min(thr, thr_q)
        else:
            thr = min(max((t - ctx.t_ses) / eng["startup_s"], 0.0), 1.0)
        n = eng["count"]
        f_unit = n * (eng["mdot0_kg_s"] * eng["ve_m_s"] + eng["pe_pa"] * eng["ae_m2"])
        thr_g = (gd["g_limit"] * G0 * mp.m + n * P * eng["ae_m2"]) / f_unit
        thr = min(thr, thr_g)
        if phase == S1_BURN or t - ctx.t_ses > eng["startup_s"]:
            thr = max(thr, eng["throttle_min"])
        if phase == S1_BURN:
            tv, tk, kick = gd["t_vertical_s"], gd["kick_duration_s"], math.radians(gd["kick_angle_deg"])
            if t < tv:
                d_cmd = up
            elif t < tv + tk:
                th = kick * smoothstep((t - tv) / tk)
                d_cmd = _add(_scale(up, math.cos(th)), dh, math.sin(th))
            else:
                dk = _add(_scale(up, math.cos(kick)), dh, math.sin(kick))
                dv = _unit(v_rel)
                b = smoothstep((t - tv - tk) / gd["blend_s"])
                d_cmd = _unit(_add(_scale(dk, 1 - b), dv, b))
    # thrust magnitude
    F, mdot = engine_thrust(eng, thr, P)
    if phase in (COAST_S2, S2_BURN):
        vz = _dot(v, up)
        vh_vec = _add(v, up, -vz)
        hdir = _unit(vh_vec)
        vh = _norm(vh_vec)
        rt = RE + gd["target_alt_m"]
        wn, ze = gd["s2_wn"], gd["s2_zeta"]
        az_cmd = wn * wn * (rt - rn) - 2 * ze * wn * vz
        acc = max(F / mp.m, 0.1) if phase == S2_BURN and F > 0 else 30.0
        s = (az_cmd + MU / (rn * rn) - vh * vh / rn) / acc
        s = min(max(s, gd["s2_sin_min"]), gd["s2_sin_max"])
        d_cmd = _add(_scale(hdir, math.sqrt(1 - s * s)), up, s)
    elif phase in (ORBIT, S2_DEPLETED):
        d_cmd = _unit(v)

    # ---------------- attitude control -----------------
    db = mat_t_vec(R, d_cmd)
    ax_y, ax_z = -db[2], db[1]
    sn = math.hypot(ax_y, ax_z)
    ang = math.atan2(sn, db[0])
    if sn > 1e-12:
        ey, ez = ax_y / sn * ang, ax_z / sn * ang
    else:
        ey = ez = 0.0
    wn, ze = gd["attitude_wn"], gd["attitude_zeta"]
    al_y = wn * wn * ey - 2 * ze * wn * w[1]
    al_z = wn * wn * ez - 2 * ze * wn * w[2]
    tau = [0.0, 0.0, 0.0]
    thrust_b = (0.0, 0.0, 0.0)
    gimbal = (0.0, 0.0)
    if F > 1.0:
        L = mp.x_cg - eng["x_gimbal"]
        ty = mp.iyy * al_y - tau_aero[1]
        tz = mp.iyy * al_z - tau_aero[2]
        dy = -tz / (L * F)
        dz = ty / (L * F)
        dm = math.hypot(dy, dz)
        lim = math.tan(math.radians(eng["gimbal_max_deg"]))
        if dm > lim:
            dy, dz = dy * lim / dm, dz * lim / dm
        nrm = math.sqrt(1 + dy * dy + dz * dz)
        thrust_b = (F / nrm, F * dy / nrm, F * dz / nrm)
        tau = [0.0, L * thrust_b[2], -L * thrust_b[1]]
        gimbal = (dy, dz)
    elif stage == 2 and phase != PAD:
        amax = gd["rcs_alpha_max"]
        am = math.hypot(al_y, al_z)
        if am > amax:
            al_y, al_z = al_y * amax / am, al_z * amax / am
        tau = [0.0, mp.iyy * al_y, mp.iyy * al_z]
    tau = (tau[0] + tau_aero[0], tau[1] + tau_aero[1], tau[2] + tau_aero[2])

    dy_ = [0.0] * 15
    if phase == PAD:
        vr = _cross(we, r)
        acc_in = _cross(we, vr)
        wb = mat_t_vec(R, we)
        dy_[0:3] = vr
        dy_[3:6] = acc_in
        wq = wb
        dw = (0.0, 0.0, 0.0)
    else:
        fb = (thrust_b[0] + fa[0], thrust_b[1] + fa[1], thrust_b[2] + fa[2])
        fi = mat_vec(R, fb)
        acc_in = (fi[0] / mp.m + g[0], fi[1] / mp.m + g[1], fi[2] / mp.m + g[2])
        dy_[0:3] = v
        dy_[3:6] = acc_in
        wq = w
        I = (mp.ixx, mp.iyy, mp.iyy)
        Iw = (I[0] * w[0], I[1] * w[1], I[2] * w[2])
        gyro = _cross(w, Iw)
        dw = ((tau[0] - gyro[0]) / I[0], (tau[1] - gyro[1]) / I[1], (tau[2] - gyro[2]) / I[2])
    qw, qx, qy, qz = q
    ox, oy, oz = wq
    dy_[6] = 0.5 * (-qx * ox - qy * oy - qz * oz)
    dy_[7] = 0.5 * (qw * ox + qy * oz - qz * oy)
    dy_[8] = 0.5 * (qw * oy - qx * oz + qz * ox)
    dy_[9] = 0.5 * (qw * oz + qx * oy - qy * ox)
    dy_[10:13] = dw
    if stage == 1:
        dy_[13] = -mdot
    else:
        dy_[14] = -mdot
    if not aux:
        return dy_
    sensed = _add(acc_in, g, -1.0)
    tel = dict(
        t=t, phase=phase, h=h, rn=rn, T=T, P=P, rho=rho, a=a_snd, V_air=V, mach=mach, qbar=qbar,
        thrust=F, mdot=mdot, throttle=thr, mass=mp.m, x_cg=mp.x_cg, ixx=mp.ixx, iyy=mp.iyy,
        alpha=alpha, g_load=_norm(sensed) / G0, v_inertial=_norm(v), v_rel=_norm(v_rel),
        gimbal_y=gimbal[0], gimbal_z=gimbal[1], fa=fa, thrust_b=thrust_b,
    )
    return dy_, tel
