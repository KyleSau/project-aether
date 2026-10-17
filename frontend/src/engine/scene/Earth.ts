// Procedural Earth (ARTISTIC APPROXIMATION of land/ocean/cloud cover) with view-dependent haze.
// The launch site's coastline (ocean to the east) is matched to the local pad terrain.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const {
  uniform, float, vec3, positionGeometry, normalize, dot, max, mix, smoothstep, mx_fractal_noise_float, pow, clamp,
  positionWorld, exp, length, abs, select, Fn, reflect,
} = T;
import { RE } from '../../physics/dynamics';

export class Earth {
  mesh: THREE.Mesh;
  uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  uSunI = uniform(1);
  uCamAlt = uniform(0);
  constructor(segments = 256) {
    const geo = new THREE.SphereGeometry(1, segments, segments / 2);
    const mat = new THREE.MeshBasicNodeMaterial({ fog: false });
    const color = Fn(() => {
      const d = normalize(positionGeometry);
      // local tangent coordinates around the pad (pad at +Y)
      const eastKm = d.x.mul(RE / 1000);
      const northKm = d.z.negate().mul(RE / 1000);
      const distKm = length(vec3(eastKm, northKm, 0));
      const near = float(1).sub(smoothstep(250, 700, distKm)).mul(smoothstep(0, 0.2, d.y));
      const n1 = mx_fractal_noise_float(d.mul(2.2), 5, 2.0, 0.5, 1.0);
      const n2 = mx_fractal_noise_float(d.mul(14.0), 4, 2.0, 0.5, 1.0);
      const cont = n1.add(n2.mul(0.25));
      const landGlobal = smoothstep(0.02, 0.08, cont);
      const coastNoise = mx_fractal_noise_float(vec3(northKm.mul(0.01), float(3.3), float(0.7)), 4, 2.0, 0.5, 1.0).mul(25);
      const landLocal = smoothstep(2.0, -2.0, eastKm.sub(1.6).sub(coastNoise.mul(smoothstep(5, 60, distKm))));
      const land = mix(landGlobal, landLocal, near);
      const lat = abs(d.y.mul(0.6).add(d.z.mul(-0.8))); // rough latitude proxy relative to a tilted pole
      const veg = mix(vec3(0.07, 0.13, 0.05), vec3(0.32, 0.27, 0.16), smoothstep(-0.1, 0.25, n2));
      const ice = smoothstep(0.82, 0.9, lat);
      const landCol = mix(veg, vec3(0.85, 0.88, 0.92), ice);
      const ocean = vec3(0.012, 0.05, 0.11);
      const surf = mix(ocean, landCol, land);
      // lighting
      const ndl = dot(d, this.uSunDir);
      const lit = clamp(ndl.mul(1.1).add(0.05), 0, 1);
      const viewDir = normalize(positionWorld).negate();
      const spec = pow(max(dot(reflect(this.uSunDir.negate(), d), viewDir), 0), 60).mul(float(1).sub(land)).mul(0.6).mul(lit);
      // clouds (global), thinner near the pad so the local sky clouds dominate
      const cn = mx_fractal_noise_float(d.mul(9.0).add(vec3(3.1, 0.2, 1.7)), 5, 2.0, 0.55, 1.0);
      const cloud = smoothstep(0.08, 0.35, cn).mul(0.85).mul(float(1).sub(near.mul(0.6)));
      const base = mix(surf.mul(lit), vec3(0.92).mul(clamp(ndl.mul(1.2).add(0.1), 0, 1)), cloud).add(spec);
      // haze: optical depth grows toward the limb
      const mu = max(dot(d, viewDir), 0.02);
      const hazeAmt = float(1).sub(exp(float(-0.18).div(mu))).mul(smoothstep(-5000, 30000, this.uCamAlt).mul(0.7).add(0.3));
      const hazeCol = vec3(0.35, 0.55, 0.95).mul(clamp(ndl.add(0.15), 0, 1));
      const night = vec3(0.004, 0.006, 0.012);
      return mix(base, hazeCol, hazeAmt).mul(this.uSunI).add(select(ndl.lessThan(0), night, vec3(0)));
    })();
    mat.colorNode = color;
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1;
  }
}

export { exp };
