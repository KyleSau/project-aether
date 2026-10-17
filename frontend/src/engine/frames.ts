// Coordinate frames.
// Sim: ECI (z = spin axis). ECEF coincides with ECI at sim start t0.
// Render world: pad-fixed ENU tangent frame mapped to three.js axes
//   X = east, Y = up, Z = south.  Origin = vehicle base on the launch mount.
// All world positions are kept in float64 and rendered camera-relative (floating origin).
import * as THREE from 'three/webgpu';
import type { VehicleConfig } from '../physics/config';
import { OMEGA_E, RE } from '../physics/dynamics';

export class Frames {
  padEcef: [number, number, number];
  E: [number, number, number];
  N: [number, number, number];
  U: [number, number, number];
  wE: number;
  t0: number;
  constructor(cfg: VehicleConfig, t0: number) {
    const lat = (cfg.site.latitude_deg * Math.PI) / 180, lon = (cfg.site.longitude_deg * Math.PI) / 180;
    const r = RE + cfg.site.pad_altitude_m;
    this.padEcef = [r * Math.cos(lat) * Math.cos(lon), r * Math.cos(lat) * Math.sin(lon), r * Math.sin(lat)];
    this.U = [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
    this.E = [-Math.sin(lon), Math.cos(lon), 0];
    this.N = [-Math.sin(lat) * Math.cos(lon), -Math.sin(lat) * Math.sin(lon), Math.cos(lat)];
    this.wE = cfg.site.rotating_earth ? OMEGA_E : 0;
    this.t0 = t0;
  }
  /** rotate an ECI vector into ECEF at time t */
  eciToEcefVec(v: ArrayLike<number>, t: number, out: number[] = [0, 0, 0]) {
    const th = this.wE * (t - this.t0);
    const c = Math.cos(th), s = Math.sin(th);
    const x = v[0], y = v[1];
    out[0] = c * x + s * y;
    out[1] = -s * x + c * y;
    out[2] = v[2];
    return out;
  }
  /** ECEF direction -> render direction */
  ecefToRenderVec(d: ArrayLike<number>, out: number[] = [0, 0, 0]) {
    const { E, N, U } = this;
    const e = E[0] * d[0] + E[1] * d[1] + E[2] * d[2];
    const n = N[0] * d[0] + N[1] * d[1] + N[2] * d[2];
    const u = U[0] * d[0] + U[1] * d[1] + U[2] * d[2];
    out[0] = e; out[1] = u; out[2] = -n;
    return out;
  }
  eciToRenderVec(v: ArrayLike<number>, t: number, out: number[] = [0, 0, 0]) {
    const tmp = this.eciToEcefVec(v, t);
    return this.ecefToRenderVec(tmp, out);
  }
  /** ECI position -> render world position (float64) */
  eciToRenderPos(r: ArrayLike<number>, t: number, out: number[] = [0, 0, 0]) {
    const p = this.eciToEcefVec(r, t);
    p[0] -= this.padEcef[0]; p[1] -= this.padEcef[1]; p[2] -= this.padEcef[2];
    return this.ecefToRenderVec(p, out);
  }
  /** body->ECI quaternion -> three.js quaternion for a model built along +Y (nose), +X = body y, +Z = -body z */
  bodyQuatToRender(q: ArrayLike<number>, t: number, out: THREE.Quaternion) {
    const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
    const bx = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
    const by = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)];
    const bz = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
    const rx = this.eciToRenderVec(bx, t), ry = this.eciToRenderVec(by, t), rz = this.eciToRenderVec(bz, t);
    const m = new THREE.Matrix4().makeBasis(
      new THREE.Vector3(ry[0], ry[1], ry[2]),
      new THREE.Vector3(rx[0], rx[1], rx[2]),
      new THREE.Vector3(-rz[0], -rz[1], -rz[2]),
    );
    return out.setFromRotationMatrix(m);
  }
  /** Earth centre in render coordinates */
  earthCenter(): [number, number, number] {
    const c = this.ecefToRenderVec([-this.padEcef[0], -this.padEcef[1], -this.padEcef[2]]);
    return [c[0], c[1], c[2]];
  }
}
