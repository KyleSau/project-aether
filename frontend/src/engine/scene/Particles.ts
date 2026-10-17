// GPU particle systems for exhaust smoke, steam and dust (appearance = ARTISTIC APPROXIMATION).
//
//  * AnalyticParticles (WebGL2 fallback, also usable on WebGPU): stateless. The CPU writes only
//    newly spawned particles into a ring buffer; the vertex stage evaluates each particle in
//    closed form (drag-relaxed velocity + buoyancy + wind + turbulence). No compute, no transform
//    feedback.
//  * ComputeParticles (WebGPU): the same spawn ring, but particle state (position/velocity) is
//    integrated every frame by a WebGPU compute shader, which allows ground collision with
//    deflection/spreading and noise-driven turbulence forces.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const {
  uniform, instancedBufferAttribute, storage, instancedArray, instanceIndex, float, vec3, vec4, exp, max, mix, smoothstep,
  uv, length, mx_noise_float, clamp, dot, normalize, Fn, select, pow, If, abs,
} = T;

export interface SpawnSpec {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  t: number; life: number;
  size0: number; size1: number;
  /** drag relaxation rate [1/s] */
  drag: number;
  /** 0 = smoke (grey), 1 = steam/condensate (white), 2 = hot exhaust (emissive -> smoke) */
  kind: number;
  /** buoyant rise acceleration [m/s^2] */
  buoy: number;
  opacity: number;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type N = any;

abstract class ParticleBase {
  readonly capacity: number;
  sprite!: THREE.Sprite;
  protected a: THREE.BufferAttribute; // p0.xyz, t0
  protected b: THREE.BufferAttribute; // v0.xyz, life
  protected c: THREE.BufferAttribute; // size0, size1, drag, kind
  protected d: THREE.BufferAttribute; // buoy, opacity, seed, -
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
  abstract readonly kind: string;

  constructor(capacity: number, mk: (n: number) => THREE.BufferAttribute) {
    this.capacity = capacity;
    this.a = mk(capacity); this.b = mk(capacity); this.c = mk(capacity); this.d = mk(capacity);
    for (let i = 0; i < capacity; i++) { (this.a.array as Float32Array)[i * 4 + 3] = -1e9; (this.b.array as Float32Array)[i * 4 + 3] = 1; }
  }

  /** shared sprite material: lighting, puff mask, fades */
  protected buildMaterial(pos: N, age: N, A: N, B: N, C: N, D: N, aliveExtra: N = null) {
    const life = B.w;
    const f = age.div(life);
    let alive = age.greaterThanEqual(0).and(f.lessThan(1));
    if (aliveExtra) alive = alive.and(aliveExtra);
    const size = mix(C.x, C.y, float(1).sub(pow(float(1).sub(clamp(f, 0, 1)), float(2.2))));
    const seed = D.z;
    const mat = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false });
    mat.positionNode = pos;
    mat.scaleNode = select(alive, size, float(0));
    mat.rotationNode = seed.mul(6.283);
    const q = uv().sub(0.5);
    const rr = length(q).mul(2);
    const nz = mx_noise_float(vec3(q.x.mul(3.2), q.y.mul(3.2), seed.add(age.mul(0.05)))).mul(0.5).add(0.5);
    const mask = clamp(float(1).sub(rr).mul(1.4).sub(nz.mul(0.55)), 0, 1);
    const nrm = normalize(vec3(q.x.mul(2), float(0.6), q.y.mul(-2)));
    const sunTerm = clamp(dot(nrm, this.uSunDir).mul(0.5).add(0.6), 0, 1.2);
    const kind = C.w;
    const smoke = vec3(0.56, 0.53, 0.5);
    const steam = vec3(0.95, 0.96, 0.98);
    const base = select(kind.lessThan(0.5), smoke, steam);
    const toL = this.uLightPos.sub(pos);
    const lightFall = this.uLightPower.div(dot(toL, toL).add(400));
    const lit = base.mul(this.uSunColor.mul(sunTerm).add(this.uAmbient)).add(base.mul(this.uLightColor).mul(lightFall));
    const heat = select(kind.greaterThan(1.5), exp(age.mul(-1.6)), float(0));
    const hotCol = vec3(1.0, 0.45, 0.12).mul(heat.mul(9));
    const fadeIn = smoothstep(0.0, 0.04, f);
    const fadeOut = float(1).sub(smoothstep(0.55, 1.0, f));
    const thin = float(1).div(float(1).add(size.div(120)));
    mat.colorNode = vec4(lit.add(hotCol), mask.mul(D.y).mul(fadeIn).mul(fadeOut).mul(thin));
    const sprite = new THREE.Sprite(mat);
    sprite.count = this.capacity;
    sprite.frustumCulled = false;
    sprite.renderOrder = 5;
    this.sprite = sprite;
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

  clear() {
    const A = this.a.array as Float32Array;
    for (let i = 0; i < this.capacity; i++) A[i * 4 + 3] = -1e9;
    this.a.clearUpdateRanges();
    this.a.needsUpdate = true;
    this.cursor = 0;
    this.dirtyFrom = -1;
  }

  /** per-frame GPU work (compute); no-op for the analytic system */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  step(_renderer: THREE.WebGPURenderer, _dt: number) {}
}

export class AnalyticParticles extends ParticleBase {
  readonly kind = 'analytic (vertex-shader closed form)';
  constructor(capacity: number) {
    super(capacity, (n) => {
      const at = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4);
      at.setUsage(THREE.DynamicDrawUsage);
      return at;
    });
    const A = instancedBufferAttribute(this.a), B = instancedBufferAttribute(this.b);
    const C = instancedBufferAttribute(this.c), D = instancedBufferAttribute(this.d);
    const age = this.uTime.sub(A.w);
    const f = age.div(B.w);
    const drag = C.z;
    const tau = float(1).sub(exp(drag.mul(age).negate())).div(max(drag, float(1e-3)));
    const seed = D.z;
    const pos = Fn(() => {
      const p = A.xyz.add(B.xyz.mul(tau)).add(this.uWind.mul(age)).toVar();
      p.y.addAssign(D.x.mul(age).mul(age).mul(0.5).div(float(1).add(age.mul(0.08))));
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
    this.buildMaterial(pos, age, A, B, C, D);
  }
}

export class ComputeParticles extends ParticleBase {
  readonly kind = 'WebGPU compute (stateful integration)';
  private updateNode: N;
  private uDt = uniform(0);
  private posBuf: N;
  constructor(capacity: number) {
    super(capacity, (n) => new THREE.StorageInstancedBufferAttribute(new Float32Array(n * 4), 4));
    const SA = storage(this.a, 'vec4', capacity), SB = storage(this.b, 'vec4', capacity);
    const SC = storage(this.c, 'vec4', capacity), SD = storage(this.d, 'vec4', capacity);
    const P = instancedArray(capacity, 'vec4'); // xyz position, w = spawn-time tag
    const V = instancedArray(capacity, 'vec4'); // xyz velocity
    this.posBuf = P;
    const uTime = this.uTime, uDt = this.uDt, uWind = this.uWind, uGround = this.uGround;
    this.updateNode = Fn(() => {
      const i = instanceIndex;
      const a = SA.element(i), b = SB.element(i), c = SC.element(i), d = SD.element(i);
      const p = P.element(i), v = V.element(i);
      const age = uTime.sub(a.w);
      If(age.lessThan(0).or(age.greaterThan(b.w)), () => {
        p.w.assign(-1e9);
      }).Else(() => {
        If(abs(p.w.sub(a.w)).greaterThan(1e-4), () => {
          // first frame of a newly spawned particle: initialise state from the spawn ring
          p.xyz.assign(a.xyz);
          v.xyz.assign(b.xyz);
          p.w.assign(a.w);
        });
        const seed = d.z;
        const turb = vec3(
          mx_noise_float(vec3(p.x.mul(0.015), p.y.mul(0.015), seed.mul(0.01).add(age.mul(0.05)))),
          mx_noise_float(vec3(p.y.mul(0.015), p.z.mul(0.015), seed.mul(0.01).add(3.1))),
          mx_noise_float(vec3(p.z.mul(0.015), p.x.mul(0.015), seed.mul(0.01).add(7.7))),
        ).mul(c.y.div(40).add(0.3)).mul(2.2);
        const buoy = vec3(0, d.x.div(float(1).add(age.mul(0.08))), 0);
        const acc = uWind.sub(v.xyz).mul(c.z).add(buoy).add(turb);
        v.xyz.addAssign(acc.mul(uDt));
        p.xyz.addAssign(v.xyz.mul(uDt));
        // ground contact: deflect and spread along the surface (pad / terrain)
        const f = age.div(b.w);
        const size = mix(c.x, c.y, float(1).sub(pow(float(1).sub(clamp(f, 0, 1)), float(2.2))));
        const floor = uGround.add(size.mul(0.28));
        If(p.y.lessThan(floor), () => {
          p.y.assign(floor);
          const vy = abs(v.y);
          v.y.assign(vy.mul(0.15));
          const hor = vec3(v.x, 0, v.z);
          const hl = max(length(hor), float(1e-3));
          v.x.addAssign(v.x.div(hl).mul(vy.mul(0.6)));
          v.z.addAssign(v.z.div(hl).mul(vy.mul(0.6)));
        });
      });
    })().compute(capacity);
    const A = SA.element(instanceIndex), B = SB.element(instanceIndex), C = SC.element(instanceIndex), D = SD.element(instanceIndex);
    const Pn = P.element(instanceIndex);
    const age = this.uTime.sub(A.w);
    this.buildMaterial(Pn.xyz, age, A, B, C, D, abs(Pn.w.sub(A.w)).lessThan(1e-3));
  }

  step(renderer: THREE.WebGPURenderer, dt: number) {
    if (dt <= 0) return;
    const n = Math.min(8, Math.ceil(dt / 0.05));
    const h = dt / n;
    const t1 = this.uTime.value as number;
    for (let k = 0; k < n; k++) {
      this.uDt.value = h;
      this.uTime.value = t1 - dt + h * (k + 1);
      renderer.compute(this.updateNode);
    }
    this.uTime.value = t1;
  }
}

export type ParticleSystem = ParticleBase;
