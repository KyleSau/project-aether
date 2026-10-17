// Visual-effect drivers: plumes (reduced-order model), particle emitters (pad exhaust, sound-
// suppression steam [ARTISTIC], ascent trail, stage-separation puff), engine light and the
// vibration-driven camera shake model. All inputs come from simulation telemetry.
import * as THREE from 'three/webgpu';
import type { VehicleConfig } from '../physics/config';
import { atmosphere } from '../physics/atmosphere';
import { plumeState, type PlumeState } from './plumeModel';
import type { ParticleSystem } from './scene/Particles';
import { applyPlume, makePlumeMesh, type PlumeUniforms } from './scene/Plume';
import { GROUND_Y, MOUND_HALF, DECK_Y } from './scene/Launchpad';
import type { V3 } from './CameraRig';

const rnd = (a: number, b: number) => a + Math.random() * (b - a);

export interface EffectInput {
  t: number; // mission time
  dt: number; // mission-time step this frame
  base: V3; axis: V3; vel: V3; // active vehicle base (render), nose axis, earth-relative velocity (render)
  alt: number; pa: number;
  thrustS1: number; thrS1: number; s1Burning: boolean;
  thrustS2: number; thrS2: number; s2Burning: boolean;
  countdown: number; // seconds to T-0 (positive before launch)
  mach: number; qbar: number;
  s2Base: V3; s2Axis: V3;
  sepEvent: boolean;
}

export class Effects {
  particles: ParticleSystem;
  s1Cores: { mesh: THREE.Mesh; u: PlumeUniforms }[] = [];
  s1Outer: { mesh: THREE.Mesh; u: PlumeUniforms };
  s2Core: { mesh: THREE.Mesh; u: PlumeUniforms };
  s2Outer: { mesh: THREE.Mesh; u: PlumeUniforms };
  engineLight: THREE.PointLight;
  ps1: PlumeState | null = null;
  ps2: PlumeState | null = null;
  shake = 0;
  acoustic = 0;
  nearField = false;
  private trailAcc = 0;
  private padAcc = 0;
  private steamAcc = 0;
  private lastTrail: V3 | null = null;
  private thrustLog: [number, number][] = [];
  budget: number;

  constructor(private cfg: VehicleConfig, particles: ParticleSystem, s1Group: THREE.Group, s2Group: THREE.Group, enginePositions: THREE.Vector3[], s2ExitY: number) {
    this.particles = particles;
    this.budget = particles.capacity;
    for (const p of enginePositions) {
      const c = makePlumeMesh(true);
      c.mesh.position.copy(p);
      s1Group.add(c.mesh);
      this.s1Cores.push(c);
    }
    this.s1Outer = makePlumeMesh(false);
    s1Group.add(this.s1Outer.mesh);
    this.s2Core = makePlumeMesh(true);
    this.s2Core.mesh.position.y = s2ExitY;
    s2Group.add(this.s2Core.mesh);
    this.s2Outer = makePlumeMesh(false);
    this.s2Outer.mesh.position.y = s2ExitY;
    s2Group.add(this.s2Outer.mesh);
    this.engineLight = new THREE.PointLight(0xff9a4a, 0, 0, 2);
  }

  /** sound pressure at the pad camera arrives with acoustic delay */
  private delayedThrust(t: number, delay: number) {
    const tt = t - delay;
    const L = this.thrustLog;
    for (let i = L.length - 1; i >= 0; i--) if (L[i][0] <= tt) return L[i][1];
    return 0;
  }

  update(inp: EffectInput, camPos: V3) {
    const e1 = this.cfg.engines.stage1, e2 = this.cfg.engines.stage2;
    const P = this.particles;
    P.uTime.value = inp.t;
    this.thrustLog.push([inp.t, inp.thrustS1 + inp.thrustS2]);
    if (this.thrustLog.length > 4000) this.thrustLog.splice(0, 2000);
    // ---------------- plumes ----------------
    const s1On = inp.s1Burning && inp.thrS1 > 0.001;
    for (const c of this.s1Cores) c.mesh.visible = s1On;
    this.s1Outer.mesh.visible = s1On;
    if (s1On) {
      const ps = plumeState(e1, inp.thrS1, inp.pa);
      this.ps1 = ps;
      const re1 = Math.sqrt(e1.ae_m2 / Math.PI);
      const start = Math.min(1, inp.thrS1 * 1.2);
      const vacF = Math.min(1, Math.max(0, 1 - inp.pa / 101325));
      // individual cores: short and bright; length grows as ambient pressure falls
      for (const c of this.s1Cores) applyPlume(c.u, ps, re1, (6 + 26 * vacF) * start, 5 * start, 1);
      // merged envelope around the cluster
      const clusterR = 1.85;
      const L = (38 + 160 * Math.pow(vacF, 2)) * start;
      applyPlume(this.s1Outer.u, ps, re1, L, 1.6 * (1 - 0.8 * vacF) * start, (clusterR * 1.15) / re1);
    }
    const s2On = inp.s2Burning && inp.thrS2 > 0.001;
    this.s2Core.mesh.visible = this.s2Outer.mesh.visible = s2On;
    if (s2On) {
      const ps = plumeState(e2, inp.thrS2, inp.pa);
      this.ps2 = ps;
      const re2 = Math.sqrt(e2.ae_m2 / Math.PI);
      const st = Math.min(1, inp.thrS2 * 1.2);
      applyPlume(this.s2Core.u, ps, re2 * 0.9, 22 * st, 3 * st, 0.9);
      applyPlume(this.s2Outer.u, ps, re2, 240 * st, 0.35 * st, 1);
    }
    // ---------------- engine light ----------------
    const thrustTotal = (s1On ? inp.thrustS1 : 0) + (s2On ? inp.thrustS2 : 0);
    const lightBase = s1On ? inp.base : inp.s2Base;
    const lightAxis = s1On ? inp.axis : inp.s2Axis;
    const lp: V3 = [lightBase[0] - lightAxis[0] * 8, lightBase[1] - lightAxis[1] * 8, lightBase[2] - lightAxis[2] * 8];
    this.engineLight.userData.world = lp;
    // luminous intensity scaled with thrust (ARTISTIC calibration: ~1.4e7 cd for 7.4 MN at sea level)
    this.engineLight.intensity = thrustTotal * 0.0016 * (s1On ? 1 : 0.15) * (0.9 + 0.2 * Math.random());
    P.uLightPos.value.set(lp[0], lp[1], lp[2]);
    P.uLightPower.value = thrustTotal * 4e-4;

    // ---------------- emitters ----------------
    const dt = Math.max(inp.dt, 0);
    if (dt > 0) {
      this.emitPad(inp, dt);
      this.emitTrail(inp, dt);
      this.emitSteam(inp, dt);
      if (inp.sepEvent) this.sepPuff(inp);
    }
    P.flush();

    // ---------------- camera shake (approximate acoustic + buffet model) ----------------
    // Acoustic power ~ eta * 0.5 * F * ve (eta = 0.5 % acoustic efficiency), hemispherical spreading,
    // delayed by d / c. Shake amplitude ~ p_rms. Onboard: structural vibration ~ throttle + transonic buffet ~ qbar.
    const d = Math.max(30, Math.hypot(camPos[0] - inp.base[0], camPos[1] - inp.base[1], camPos[2] - inp.base[2]));
    const Fd = this.delayedThrust(inp.t, d / 343);
    const W = 0.005 * 0.5 * Fd * 2900;
    const I = W / (2 * Math.PI * d * d);
    const prms = Math.sqrt(I * 415);
    const acoustic = Math.min(1.5, prms / 1500);
    const transonic = Math.exp(-Math.pow((inp.mach - 1.0) / 0.25, 2));
    const buffet = (inp.qbar / 30000) * (0.35 + 0.65 * transonic);
    const near = d < 120 ? 1 : 0;
    this.acoustic = inp.pa < 50 ? 0 : acoustic * Math.min(1, inp.pa / 20000 + (d < 2000 ? 1 : 0));
    this.nearField = d < 2000;
    this.shake = acoustic + near * (0.08 * (inp.s1Burning ? inp.thrS1 : inp.thrS2) + 0.5 * buffet);
  }

  private emitPad(inp: EffectInput, dt: number) {
    const P = this.particles;
    const h = inp.alt;
    if (!inp.s1Burning || inp.thrustS1 <= 0 || h > 400) return;
    const thrFrac = inp.thrustS1 / 7.4e6;
    const lowF = Math.max(0, 1 - h / 400);
    // rate budget scales with capacity
    const rate = (this.budget / 40) * thrFrac * lowF; // particles / s
    this.padAcc += rate * dt;
    const n = Math.min(Math.floor(this.padAcc), 400);
    this.padAcc -= n;
    for (let i = 0; i < n; i++) {
      const tt = inp.t - Math.random() * dt;
      if (h < 25) {
        // deflected exhaust leaving both trench exits (north and south) - flow speed scales with thrust
        const sz = Math.random() < 0.5 ? -1 : 1;
        const hot = Math.random() < 0.35;
        P.spawn({
          x: rnd(-5, 5), y: GROUND_Y + rnd(1, 5), z: sz * (MOUND_HALF + rnd(0, 6)),
          vx: rnd(-14, 14), vy: rnd(4, 22), vz: sz * rnd(35, 95) * thrFrac,
          t: tt, life: rnd(18, 42), size0: rnd(6, 12), size1: rnd(45, 95), drag: rnd(0.12, 0.3),
          kind: hot ? 2 : Math.random() < 0.55 ? 1 : 0, buoy: rnd(0.3, 0.9), opacity: rnd(0.45, 0.8),
        });
      }
      if (h > 2) {
        // direct plume impingement on the deck after lift-off: radial wall jet
        const a = Math.random() * Math.PI * 2;
        const sp = rnd(25, 70) * Math.min(1, 60 / (h + 20));
        P.spawn({
          x: inp.base[0] + Math.cos(a) * rnd(4, 10), y: DECK_Y + rnd(0.5, 3), z: inp.base[2] + Math.sin(a) * rnd(4, 10),
          vx: Math.cos(a) * sp, vy: rnd(2, 10), vz: Math.sin(a) * sp,
          t: tt, life: rnd(15, 35), size0: rnd(5, 10), size1: rnd(35, 70), drag: rnd(0.15, 0.35),
          kind: Math.random() < 0.3 ? 2 : Math.random() < 0.5 ? 1 : 0, buoy: rnd(0.4, 1.0), opacity: rnd(0.4, 0.7),
        });
      }
    }
  }

  private emitTrail(inp: EffectInput, dt: number) {
    const P = this.particles;
    const burning = inp.s1Burning ? inp.thrustS1 > 0 : inp.s2Burning && inp.thrustS2 > 0;
    if (!burning || inp.alt > 48000 || inp.alt < 15) { this.lastTrail = null; return; }
    const ax = inp.s1Burning ? inp.axis : inp.s2Axis;
    const b = inp.s1Burning ? inp.base : inp.s2Base;
    const plumeL = inp.s1Burning ? 30 + 80 * (1 - inp.pa / 101325) : 40;
    const cur: V3 = [b[0] - ax[0] * plumeL * 0.7, b[1] - ax[1] * plumeL * 0.7, b[2] - ax[2] * plumeL * 0.7];
    const prev = this.lastTrail ?? cur;
    this.lastTrail = cur;
    const seg = Math.hypot(cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]);
    const rho = atmosphere(inp.alt).rho;
    const expand = Math.pow(1.225 / Math.max(rho, 1e-4), 0.33); // puffs grow as density drops
    const spacing = 2.5 * expand;
    this.trailAcc += seg / spacing;
    const n = Math.min(Math.floor(this.trailAcc), 300);
    this.trailAcc -= n;
    const thin = Math.max(0.15, Math.min(1, rho / 0.6));
    for (let i = 0; i < n; i++) {
      const f = Math.random();
      P.spawn({
        x: prev[0] + (cur[0] - prev[0]) * f + rnd(-2, 2), y: prev[1] + (cur[1] - prev[1]) * f + rnd(-2, 2), z: prev[2] + (cur[2] - prev[2]) * f + rnd(-2, 2),
        vx: inp.vel[0] * 0.05 + rnd(-3, 3), vy: inp.vel[1] * 0.05 + rnd(-3, 3), vz: inp.vel[2] * 0.05 + rnd(-3, 3),
        t: inp.t - (1 - f) * dt, life: rnd(70, 140), size0: 6 * expand, size1: rnd(40, 80) * expand, drag: 0.4,
        kind: inp.alt < 3000 && Math.random() < 0.15 ? 2 : Math.random() < 0.5 ? 1 : 0, buoy: 0.05, opacity: 0.55 * thin,
      });
    }
  }

  /** sound-suppression water deluge steam and LOX boil-off vapour (ARTISTIC) */
  private emitSteam(inp: EffectInput, dt: number) {
    const P = this.particles;
    const pre = inp.countdown > 0 && inp.countdown < 9;
    const post = inp.countdown <= 0 && inp.t < 25;
    if (!pre && !post) {
      // LOX vent wisps while waiting on the pad
      if (inp.countdown > 0) this.vent(inp, dt);
      return;
    }
    this.steamAcc += (pre ? 30 : 60) * dt * (this.budget / 16000);
    const n = Math.min(Math.floor(this.steamAcc), 200);
    this.steamAcc -= n;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      P.spawn({
        x: Math.cos(a) * rnd(6, 14), y: DECK_Y + rnd(0, 2), z: Math.sin(a) * rnd(6, 14),
        vx: Math.cos(a) * rnd(1, 6), vy: rnd(2, 6), vz: Math.sin(a) * rnd(1, 6),
        t: inp.t - Math.random() * dt, life: rnd(8, 18), size0: rnd(3, 6), size1: rnd(14, 30), drag: 0.5,
        kind: 1, buoy: rnd(0.4, 1.0), opacity: rnd(0.35, 0.6),
      });
    }
    this.vent(inp, dt);
  }

  private vent(inp: EffectInput, dt: number) {
    if (inp.countdown <= 0) return;
    const P = this.particles;
    if (Math.random() > dt * 12) return;
    const side = Math.random() < 0.5 ? 1 : -1;
    const y = Math.random() < 0.6 ? 38 : 55;
    P.spawn({
      x: 1.9 * side, y, z: rnd(-0.5, 0.5), vx: side * rnd(1.5, 4), vy: rnd(-1.5, 0.5), vz: rnd(-0.5, 0.5),
      t: inp.t, life: rnd(4, 7), size0: 0.8, size1: rnd(5, 9), drag: 0.6, kind: 1, buoy: -0.2, opacity: 0.35,
    });
  }

  private sepPuff(inp: EffectInput) {
    const P = this.particles;
    const b = inp.s2Base;
    for (let i = 0; i < 80; i++) {
      const a = Math.random() * Math.PI * 2;
      P.spawn({
        x: b[0], y: b[1], z: b[2], vx: inp.vel[0] + Math.cos(a) * rnd(5, 15), vy: inp.vel[1] + rnd(-5, 5), vz: inp.vel[2] + Math.sin(a) * rnd(5, 15),
        t: inp.t, life: rnd(3, 6), size0: 2, size1: rnd(10, 20), drag: 0.02, kind: 1, buoy: 0, opacity: 0.25,
      });
    }
  }

  reset() {
    this.particles.clear();
    this.thrustLog = [];
    this.lastTrail = null;
    this.trailAcc = this.padAcc = this.steamAcc = 0;
  }
}
