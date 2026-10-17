// Exhaust plume rendering (TSL). Shape (expansion angle, jet width, shock-cell spacing,
// Mach disk) is driven per frame by the REDUCED-ORDER plume model (plumeModel.ts) using the
// simulated ambient pressure; luminosity/colour are an ARTISTIC APPROXIMATION.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const {
  uniform, float, vec3, positionLocal, normalView, positionViewDirection, abs, dot, exp, max, min, cos, sin,
  pow, mix, smoothstep, mx_noise_float, time, clamp, atan, Fn, sqrt,
} = T;
import type { PlumeState } from '../plumeModel';

type U = any; // eslint-disable-line

export interface PlumeUniforms {
  len: U; re: U; tanT: U; rj: U; spread: U; cell: U; diamonds: U; intensity: U; vac: U; ripple: U;
  mdX: U; md: U; seed: U; core: U;
}

const TWO_PI = Math.PI * 2;

export function makePlumeMesh(core: boolean): { mesh: THREE.Mesh; u: PlumeUniforms } {
  const geo = new THREE.CylinderGeometry(1, 1, 1, 40, 96, true);
  geo.translate(0, -0.5, 0); // y in [-1, 0]
  const u: PlumeUniforms = {
    len: uniform(20), re: uniform(0.5), tanT: uniform(0.1), rj: uniform(0.6), spread: uniform(0.05),
    cell: uniform(4), diamonds: uniform(0), intensity: uniform(0), vac: uniform(0), ripple: uniform(0),
    mdX: uniform(2), md: uniform(0), seed: uniform(Math.random() * 100), core: uniform(core ? 1 : 0),
  };
  const x = positionLocal.y.negate().mul(u.len); // axial distance from exit [m]
  const radius = Fn(() => {
    const dr = u.rj.sub(u.re);
    const k = abs(u.tanT).div(max(abs(dr), float(0.02)));
    const r0 = u.re.add(dr.mul(float(1).sub(exp(x.mul(k).negate())))).add(u.spread.mul(x));
    const rip = float(1).add(u.ripple.mul(sin(x.div(u.cell).mul(TWO_PI))).mul(exp(x.div(u.cell.mul(4)).negate())));
    return max(r0.mul(rip), float(0.05));
  })();
  const ang = atan(positionLocal.z, positionLocal.x);
  const wob = mx_noise_float(vec3(x.mul(0.15), ang.mul(1.5), time.mul(9).add(u.seed))).mul(0.06).mul(x.div(u.len).add(0.1));
  const rr = radius.mul(float(1).add(wob));
  const mat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
  mat.positionNode = vec3(positionLocal.x.mul(rr), positionLocal.y.mul(u.len), positionLocal.z.mul(rr));
  // brightness
  const s = x.div(u.len);
  const axial = exp(s.mul(u.core.mul(3).add(2.2)).negate());
  const ph = x.div(u.cell).mul(TWO_PI);
  const dia = pow(max(cos(ph), float(0)), float(8)).mul(exp(x.div(u.cell.mul(5)).negate())).mul(u.diamonds);
  const disk = exp(pow(x.sub(u.mdX).div(max(u.cell.mul(0.08), float(0.15))), float(2)).negate()).mul(u.md).mul(1.5);
  const facing = abs(dot(normalView, positionViewDirection));
  const thick = pow(facing, mix(float(2.4), float(1.0), u.vac));
  const flick = mx_noise_float(vec3(x.mul(0.35).sub(time.mul(60)), ang.mul(2), u.seed)).mul(0.25).add(0.9);
  const tailFade = float(1).sub(smoothstep(0.55, 1.0, s));
  const headFade = smoothstep(0.0, 0.015, s);
  const I = u.intensity.mul(axial.mul(float(1).add(dia.mul(2.5))).add(disk)).mul(thick).mul(flick).mul(tailFade).mul(headFade);
  // colour: RP-1 core white-yellow -> orange envelope; vacuum plume pale and translucent
  const hot = vec3(1.0, 0.86, 0.62);
  const warm = vec3(1.0, 0.46, 0.14);
  const vacC = vec3(0.65, 0.72, 1.0);
  const col0 = mix(warm, hot, clamp(axial.mul(u.core.add(0.6)), 0, 1));
  const col = mix(col0, mix(vacC, hot, axial), u.vac.mul(0.7));
  // fade the plume's own geometry near the very wide vacuum skirt
  const vacFade = mix(float(1), exp(s.mul(2.5).negate()), u.vac);
  mat.colorNode = col.mul(I).mul(vacFade);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 10;
  return { mesh, u };
}

/** Apply a reduced-order plume state to a plume's uniforms. */
export function applyPlume(u: PlumeUniforms, ps: PlumeState, exitRadius: number, lenScale: number, intensity: number, radiusScale = 1) {
  const re = exitRadius * radiusScale;
  const rj = Math.min(Math.max((ps.dj / 2) * radiusScale, re * 0.55), re * 45);
  const vac = Math.min(1, Math.max(0, 1 - Math.log10(Math.max(ps.pa, 1)) / 3.5));
  u.re.value = re;
  u.rj.value = rj;
  u.tanT.value = Math.tan(Math.min(Math.abs(ps.thetaB), 1.3)) * Math.sign(ps.thetaB || 1);
  u.spread.value = 0.035 + 0.12 * vac;
  u.cell.value = Math.max(ps.cellLength * radiusScale, re * 0.8);
  u.diamonds.value = ps.diamonds;
  u.ripple.value = Math.min(0.18, 0.12 * Math.abs(Math.log(ps.ratio))) * ps.diamonds;
  u.md.value = ps.machDisk ? ps.diamonds : 0;
  u.mdX.value = ps.machDiskX * radiusScale;
  u.vac.value = vac;
  u.len.value = lenScale;
  u.intensity.value = intensity;
}

export { sqrt, min };
