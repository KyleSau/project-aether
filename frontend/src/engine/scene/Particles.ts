// Stateless ("analytic") GPU particles: the CPU writes only newly spawned particles into a ring
// buffer; the vertex stage evaluates each particle's position in closed form from its spawn
// state and the mission clock (drag-relaxed velocity + buoyancy + wind + turbulence).
// Works identically on the WebGPU and WebGL2 backends (no compute / transform feedback).
// Smoke/steam appearance is an ARTISTIC APPROXIMATION.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const {
  uniform, instancedBufferAttribute, float, vec3, vec4, exp, max, mix, smoothstep, uv, length, mx_noise_float,
  clamp, dot, normalize, Fn, select, pow, sin, cos,
} = T;

export interface SpawnSpec {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  t: number; life: number;
  size0: number; size1: number;
  /** drag relaxation rate [1/s] */
  drag: number;
  /** 0 = smoke (grey-brown), 1 = steam (white), 2 = hot exhaust (emissive -> smoke) */
  kind: number;
  /** buoyant rise acceleration [m/s^2] */
  buoy: number;
  opacity: number;
}

export class ParticleSystem {
  readonly capacity: number;
  sprite: THREE.Sprite;
  private a: THREE.InstancedBufferAttribute; // p0.xyz, t0
  private b: THREE.InstancedBufferAttribute; // v0.xyz, life
  private c: THREE.InstancedBufferAttribute; // size0, size1, drag, kind
  private d: THREE.InstancedBufferAttribute; // buoy, opacity, seed, -
  private cursor = 0;
  private dirtyFrom = -1;
  private dirtyCount = 0;
  uTime = uniform(0);
  uSunDir = uniform(new THREE.Vector3(0.3, 0.8, 0.2));
  uSunColor = uniform(new THREE.Color(1, 0.95, 0.85));
  uAmbient = uniform(new THREE.Color(0.35, 0.42, 0.55));
  uLightPos = uniform(new THREE.Vector3());
  uLightColor = uniform(new THREE.Color(1, 0.55, 0.2));
  uLightPower = uniform(0);
  uWind = uniform(new THREE.Vector3(4, 0, 0));
  uGround = uniform(-16);
  spawnedTotal = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    const mk = () => {
      const at = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
      at.setUsage(THREE.DynamicDrawUsage);
      return at;
    };
    this.a = mk(); this.b = mk(); this.c = mk(); this.d = mk();
    // initialise as dead
    for (let i = 0; i < capacity; i++) { this.a.array[i * 4 + 3] = -1e9; this.b.array[i * 4 + 3] = 1; }
    const A = instancedBufferAttribute(this.a), B = instancedBufferAttribute(this.b);
    const C = instancedBufferAttribute(this.c), D = instancedBufferAttribute(this.d);
    const age = this.uTime.sub(A.w);
    const life = B.w;
    const f = age.div(life);
    const alive = age.greaterThanEqual(0).and(f.lessThan(1));
    const drag = C.z;
    const tau = float(1).sub(exp(drag.mul(age).negate())).div(max(drag, float(1e-3)));
    const seed = D.z;
    const pos = Fn(() => {
      const p = A.xyz.add(B.xyz.mul(tau)).add(this.uWind.mul(age)).toVar();
      // buoyant rise saturating with age (entrainment)
      p.y.addAssign(D.x.mul(age).mul(age).mul(0.5).div(float(1).add(age.mul(0.08))));
      // large-scale turbulence (curl-free noise displacement, growing with age)
      const n = vec3(
        mx_noise_float(vec3(seed, age.mul(0.07), float(1.3))),
        mx_noise_float(vec3(seed.add(17.1), age.mul(0.07), float(4.1))),
        mx_noise_float(vec3(seed.add(41.7), age.mul(0.07), float(7.7))),
      );
      p.addAssign(n.mul(age.mul(1.6)).mul(C.y.div(40).add(0.3)));
      const sz = mix(C.x, C.y, float(1).sub(pow(float(1).sub(clamp(f, 0, 1)), float(2.2))));
      p.y.assign(max(p.y, this.uGround.add(sz.mul(0.28))));
      return p;
    })();
    const size = mix(C.x, C.y, float(1).sub(pow(float(1).sub(clamp(f, 0, 1)), float(2.2))));
    const mat = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false });
    mat.positionNode = pos;
    mat.scaleNode = select(alive, size, float(0));
    mat.rotationNode = seed.mul(6.283);
    // puff mask
    const q = uv().sub(0.5);
    const rr = length(q).mul(2);
    const nz = mx_noise_float(vec3(q.x.mul(3.2), q.y.mul(3.2), seed.add(age.mul(0.05)))).mul(0.5).add(0.5);
    const mask = clamp(float(1).sub(rr).mul(1.4).sub(nz.mul(0.55)), 0, 1);
    // lighting: fake normal from sprite uv (sphere-ish), sun + sky ambient + engine light
    const nrm = normalize(vec3(q.x.mul(2), float(0.6), q.y.mul(-2)));
    const sunTerm = clamp(dot(nrm, this.uSunDir).mul(0.5).add(0.6), 0, 1.2);
    const kind = C.w;
    const smoke = vec3(0.56, 0.53, 0.5);
    const steam = vec3(0.95, 0.96, 0.98);
    const base = select(kind.lessThan(0.5), smoke, steam);
    const toL = this.uLightPos.sub(pos);
    const dl2 = dot(toL, toL);
    const lightFall = this.uLightPower.div(dl2.add(400));
    const lit = base.mul(this.uSunColor.mul(sunTerm).add(this.uAmbient)).add(base.mul(this.uLightColor).mul(lightFall));
    // hot exhaust: emissive early in life, cooling to smoke
    const heat = select(kind.greaterThan(1.5), exp(age.mul(-1.6)), float(0));
    const hotCol = vec3(1.0, 0.45, 0.12).mul(heat.mul(9));
    const fadeIn = smoothstep(0.0, 0.04, f);
    const fadeOut = float(1).sub(smoothstep(0.55, 1.0, f));
    const thin = float(1).div(float(1).add(size.div(120))); // large puffs get optically thinner
    mat.colorNode = vec4(lit.add(hotCol), mask.mul(D.y).mul(fadeIn).mul(fadeOut).mul(thin));
    this.sprite = new THREE.Sprite(mat);
    this.sprite.count = capacity;
    this.sprite.frustumCulled = false;
    this.sprite.renderOrder = 5;
    void sin; void cos;
  }

  spawn(s: SpawnSpec) {
    const i = this.cursor;
    const A = this.a.array as Float32Array, B = this.b.array as Float32Array, C = this.c.array as Float32Array, D = this.d.array as Float32Array;
    const o = i * 4;
    A[o] = s.x; A[o + 1] = s.y; A[o + 2] = s.z; A[o + 3] = s.t;
    B[o] = s.vx; B[o + 1] = s.vy; B[o + 2] = s.vz; B[o + 3] = s.life;
    C[o] = s.size0; C[o + 1] = s.size1; C[o + 2] = s.drag; C[o + 3] = s.kind;
    D[o] = s.buoy; D[o + 1] = s.opacity; D[o + 2] = Math.random() * 1000; D[o + 3] = 0;
    if (this.dirtyFrom < 0) { this.dirtyFrom = i; this.dirtyCount = 0; }
    this.dirtyCount++;
    this.cursor = (i + 1) % this.capacity;
    this.spawnedTotal++;
  }

  /** upload newly written ranges */
  flush() {
    if (this.dirtyFrom < 0) return;
    const start = this.dirtyFrom, count = Math.min(this.dirtyCount, this.capacity);
    for (const at of [this.a, this.b, this.c, this.d]) {
      at.clearUpdateRanges();
      if (start + count <= this.capacity) at.addUpdateRange(start * 4, count * 4);
      else {
        at.addUpdateRange(start * 4, (this.capacity - start) * 4);
        at.addUpdateRange(0, (start + count - this.capacity) * 4);
      }
      at.needsUpdate = true;
    }
    this.dirtyFrom = -1;
    this.dirtyCount = 0;
  }

  /** kill all particles (restart) */
  clear() {
    const A = this.a.array as Float32Array;
    for (let i = 0; i < this.capacity; i++) A[i * 4 + 3] = -1e9;
    this.a.clearUpdateRanges();
    this.a.needsUpdate = true;
    this.cursor = 0;
    this.dirtyFrom = -1;
  }
}
