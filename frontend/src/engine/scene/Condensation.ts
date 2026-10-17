// Transonic condensation ("vapour cone") and humid-layer shroud: ARTISTIC APPROXIMATION.
// Intensity model (stated assumption): I = RH(h) * exp(-((M - 1.0)/0.12)^2) * min(q/15 kPa, 1),
// where RH(h) is an assumed humidity profile (moist boundary layer below 3 km and a moist layer
// near 6-8 km). Not a thermodynamic condensation calculation.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const { uniform, float, vec3, vec4, positionGeometry, normalView, positionViewDirection, abs, dot, mx_noise_float, time, smoothstep, atan, pow } = T;

export function humidity(h: number) {
  const bl = Math.exp(-Math.pow(h / 2500, 2));
  const layer = 0.7 * Math.exp(-Math.pow((h - 7000) / 1200, 2));
  return Math.min(1, 0.25 + 0.75 * Math.max(bl, layer)) * (h < 14000 ? 1 : 0);
}

export function vaporIntensity(mach: number, h: number, qbar: number) {
  return humidity(h) * Math.exp(-Math.pow((mach - 1.0) / 0.12, 2)) * Math.min(qbar / 15000, 1);
}

export class Condensation {
  mesh: THREE.Mesh;
  uI = uniform(0);
  constructor(xShoulder = 57.5, r0 = 2.75) {
    // cone opening aft from the fairing shoulder (body +Y = nose)
    const geo = new THREE.CylinderGeometry(r0, r0 * 3.6, 16, 48, 24, true);
    geo.translate(0, xShoulder - 8, 0);
    const mat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: false });
    const yN = positionGeometry.y.sub(xShoulder - 16).div(16); // 0 aft .. 1 shoulder
    const ang = atan(positionGeometry.z, positionGeometry.x);
    const n = mx_noise_float(vec3(ang.mul(3), yN.mul(4).add(time.mul(3)), float(1.7))).mul(0.5).add(0.5);
    const facing = abs(dot(normalView, positionViewDirection));
    const rim = pow(float(1).sub(facing), float(1.5));
    const a = this.uI.mul(smoothstep(0.0, 0.5, yN)).mul(smoothstep(1.0, 0.85, yN)).mul(n).mul(rim.mul(0.8).add(0.2)).mul(0.75);
    mat.colorNode = vec4(vec3(0.95, 0.96, 1.0), a);
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 8;
    this.mesh.visible = false;
  }
  update(mach: number, h: number, qbar: number) {
    const i = vaporIntensity(mach, h, qbar);
    this.uI.value = i;
    this.mesh.visible = i > 0.01;
  }
}
