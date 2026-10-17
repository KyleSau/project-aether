// Atmospheric scattering sky (single scattering, Rayleigh + Mie, exponential atmosphere,
// Kasten-Young airmass toward the sun, Earth-shadow test) evaluated per pixel as the scene
// background, plus an altitude-aware aerial-perspective fog. APPROXIMATION of real scattering.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const {
  uniform, float, vec3, Fn, Loop, normalWorldGeometry, dot, sqrt, max, min, exp, pow, mix, smoothstep, clamp,
  select, positionWorld, positionView, length, acos, fog, mx_noise_float, abs, int,
} = T;

export const R_GROUND_KM = 6378.137;
export const R_TOP_KM = R_GROUND_KM + 100;
const BETA_R = new THREE.Vector3(5.8e-3, 13.5e-3, 33.1e-3); // 1/km at sea level
const BETA_M = 21e-3;
const HR = 8.0, HM = 1.2;

export class SkyModel {
  uSunDir = uniform(new THREE.Vector3(0.5, 0.3, -0.2).normalize());
  /** camera position relative to Earth centre, km */
  uCamKm = uniform(new THREE.Vector3(0, R_GROUND_KM, 0));
  uSunI = uniform(22);
  uExposure = uniform(1);
  /** local up at camera (render frame) */
  uUp = uniform(new THREE.Vector3(0, 1, 0));
  uCamAltM = uniform(0);
  uFogColor = uniform(new THREE.Color(0.6, 0.7, 0.85));
  uHaze = uniform(1.0);
  backgroundNode: THREE.Node;
  fogNode: THREE.Node;

  constructor(samples = 12) {
    const sunDir = this.uSunDir, cam = this.uCamKm, sunI = this.uSunI;
    const raySphere = (o: THREE.Node, d: THREE.Node, r: number) => {
      // returns vec2(t0, t1); t1 < t0 if miss
      const b = dot(o as never, d as never);
      const c = dot(o as never, o as never).sub(r * r);
      const disc = b.mul(b).sub(c);
      const sq = sqrt(max(disc, 0));
      return { t0: b.negate().sub(sq), t1: b.negate().add(sq), hit: disc.greaterThan(0) };
    };
    const airmass = (cosz: THREE.Node) => {
      const zdeg = acos(clamp(cosz as never, -1, 1)).mul(57.29578);
      return float(1).div(max(cosz as never, -0.2).add(float(0.50572).mul(pow(max(float(96.07995).sub(zdeg), 0.5), -1.6364))));
    };
    this.backgroundNode = Fn(() => {
      const d = normalWorldGeometry.normalize();
      const o = cam;
      const top = raySphere(o, d, R_TOP_KM);
      const gnd = raySphere(o, d, R_GROUND_KM);
      const tStart = max(top.t0, 0);
      const tEnd = select(gnd.hit.and(gnd.t0.greaterThan(0)), gnd.t0, top.t1).toVar();
      const segLen = max(tEnd.sub(tStart), 0);
      const ds = segLen.div(samples);
      const sumR = vec3(0).toVar();
      const sumM = vec3(0).toVar();
      const odR = float(0).toVar();
      const odM = float(0).toVar();
      Loop({ start: int(0), end: int(samples) }, ({ i }: { i: unknown }) => {
        const t = tStart.add(ds.mul(float(i).add(0.5)));
        const p = o.add(d.mul(t));
        const r = length(p);
        const h = max(r.sub(R_GROUND_KM), 0);
        const dr = exp(h.div(-HR)).mul(ds);
        const dm = exp(h.div(-HM)).mul(ds);
        odR.addAssign(dr);
        odM.addAssign(dm);
        const up = p.div(r);
        const mu = dot(up, sunDir);
        // Earth shadow: sun below the geometric horizon at this point
        const horizon = sqrt(max(float(1).sub(float(R_GROUND_KM).div(r).pow(2)), 0)).negate();
        const lit = smoothstep(horizon.sub(0.02), horizon.add(0.01), mu);
        const am = airmass(mu);
        const lR = exp(h.div(-HR)).mul(HR).mul(am);
        const lM = exp(h.div(-HM)).mul(HM).mul(am);
        const tau = vec3(BETA_R.x, BETA_R.y, BETA_R.z).mul(odR.add(lR)).add(vec3(BETA_M * 1.1).mul(odM.add(lM)));
        const att = exp(tau.negate()).mul(lit);
        sumR.addAssign(att.mul(dr));
        sumM.addAssign(att.mul(dm));
      });
      const mu = dot(d, sunDir);
      const phR = float(3 / (16 * Math.PI)).mul(mu.mul(mu).add(1));
      const g = 0.76;
      const phM = float((3 / (8 * Math.PI)) * ((1 - g * g) / (2 + g * g))).mul(mu.mul(mu).add(1)).div(pow(float(1 + g * g).sub(mu.mul(2 * g)), 1.5));
      const col = sumR.mul(vec3(BETA_R.x, BETA_R.y, BETA_R.z)).mul(phR).add(sumM.mul(BETA_M).mul(phM)).mul(sunI).toVar();
      // sun disc with transmittance along the view ray
      const trans = exp(vec3(BETA_R.x, BETA_R.y, BETA_R.z).mul(odR).add(odM.mul(BETA_M * 1.1)).negate());
      const disc = smoothstep(0.99996, 0.999985, mu).mul(select(gnd.hit.and(gnd.t0.greaterThan(0)), float(0), float(1)));
      col.addAssign(trans.mul(disc).mul(sunI.mul(120)));
      // stars (visible where the sky is dark)
      const lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
      const sn = mx_noise_float(d.mul(420));
      const star = smoothstep(0.72, 0.9, sn).mul(0.6).mul(float(1).sub(smoothstep(0.002, 0.05, lum)));
      col.addAssign(vec3(star).mul(select(gnd.hit.and(gnd.t0.greaterThan(0)), float(0), float(1))));
      // below-horizon ground seen through atmosphere (beyond the Earth mesh): dark albedo
      const gcol = vec3(0.05, 0.07, 0.09).mul(max(dot(o.add(d.mul(tEnd)).normalize(), sunDir), 0)).mul(sunI.mul(0.08)).mul(trans);
      col.addAssign(select(gnd.hit.and(gnd.t0.greaterThan(0)), gcol, vec3(0)));
      return col;
    })() as unknown as THREE.Node;

    // aerial perspective: optical depth of an exponential haze layer along the camera->fragment path
    const fogFactor = Fn(() => {
      const rel = positionWorld; // camera-relative (floating origin)
      const L = length(positionView);
      const dh = dot(rel, this.uUp);
      const H = float(1600); // haze scale height [m]
      const h0 = max(this.uCamAltM, 0);
      const h1 = max(h0.add(dh), 0);
      // integral of exp(-h/H) along a straight path, in an overflow-safe form
      const dhh = h1.sub(h0);
      const e0 = exp(h0.div(H).negate()), e1 = exp(h1.div(H).negate());
      const mean = select(abs(dhh).lessThan(1.0), e0, e0.sub(e1).mul(H).div(dhh));
      const tau = float(2.5e-5).mul(this.uHaze).mul(L).mul(mean);
      const factor = float(1).sub(exp(tau.negate()));
      return min(factor, 0.97);
    })();
    this.fogNode = fog(this.uFogColor, fogFactor) as unknown as THREE.Node;
    void mix;
  }

  /** CPU estimate of the horizon in-scatter colour (for fog / ambient), using the same model at coarse resolution */
  static horizonColor(sunElev: number): THREE.Color {
    const s = Math.max(Math.sin(sunElev), -0.1);
    const warm = Math.exp(-Math.max(s, 0.02) * 12);
    const c = new THREE.Color(0.55 + 0.35 * warm, 0.66 + 0.1 * warm, 0.86 - 0.3 * warm);
    const day = THREE.MathUtils.smoothstep(s, -0.08, 0.15);
    return c.multiplyScalar(0.05 + 0.95 * day);
  }
}
