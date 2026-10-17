// Cloud layers (ARTISTIC APPROXIMATION): fair-weather cumulus near 2 km and a thin cirrus deck
// near 9.5 km, as instanced soft sprites lit by the sun. Positions are relative to the pad.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const { uniform, instancedBufferAttribute, float, vec3, vec4, uv, length, mx_noise_float, clamp, dot, normalize, smoothstep, positionView, mix, pow } = T;

export class Clouds {
  sprite: THREE.Sprite;
  uSunDir = uniform(new THREE.Vector3(0, 1, 0));
  uSunColor = uniform(new THREE.Color(1, 1, 1));
  uAmbient = uniform(new THREE.Color(0.4, 0.45, 0.55));

  constructor(clusters: number, seed = 7) {
    let s = seed;
    const rand = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const pts: number[] = [];
    const prm: number[] = [];
    for (let c = 0; c < clusters; c++) {
      const cirrus = c % 5 === 4;
      const ang = rand() * Math.PI * 2;
      const rad = 1500 + Math.sqrt(rand()) * 28000;
      const cx = Math.cos(ang) * rad, cz = Math.sin(ang) * rad;
      const base = cirrus ? 9200 + rand() * 900 : 1700 + rand() * 500;
      const n = cirrus ? 5 : 9 + Math.floor(rand() * 7);
      const scale = cirrus ? 1600 : 650 + rand() * 700;
      for (let k = 0; k < n; k++) {
        const ox = (rand() - 0.5) * scale * (cirrus ? 3.2 : 1.6);
        const oz = (rand() - 0.5) * scale * (cirrus ? 1.2 : 1.6);
        const oy = cirrus ? rand() * 60 : Math.pow(rand(), 1.8) * scale * 0.55;
        const size = cirrus ? scale * (0.9 + rand()) : scale * (0.55 + rand() * 0.6) * (1 - oy / (scale * 1.6));
        pts.push(cx + ox, base + oy - 16, cz + oz, size);
        prm.push(rand() * 1000, cirrus ? 1 : 0, cirrus ? 0.3 : 0.62, (oy / scale));
      }
    }
    const count = pts.length / 4;
    const A = instancedBufferAttribute(new THREE.InstancedBufferAttribute(new Float32Array(pts), 4));
    const B = instancedBufferAttribute(new THREE.InstancedBufferAttribute(new Float32Array(prm), 4));
    const mat = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false });
    mat.positionNode = A.xyz;
    mat.scaleNode = A.w;
    mat.rotationNode = B.x;
    const q = uv().sub(0.5);
    const r = length(q).mul(2);
    const n1 = mx_noise_float(vec3(q.x.mul(2.5), q.y.mul(2.5), B.x)).mul(0.5).add(0.5);
    const n2 = mx_noise_float(vec3(q.x.mul(7.0), q.y.mul(7.0), B.x.add(3))).mul(0.5).add(0.5);
    const dens = clamp(float(1).sub(r).mul(1.6).sub(n1.mul(0.6)).sub(n2.mul(0.25)), 0, 1);
    // shading: brighter top / sun side, darker base
    const nrm = normalize(vec3(q.x.mul(1.6), q.y.mul(-1.6).add(0.4), float(0.6)));
    const sunTerm = clamp(dot(nrm, this.uSunDir).mul(0.6).add(0.55), 0.15, 1.3);
    const height = B.w;
    const base = mix(vec3(0.62, 0.65, 0.72), vec3(1.0), clamp(height.mul(1.2).add(0.3), 0, 1));
    const col = base.mul(this.uSunColor.mul(sunTerm).add(this.uAmbient.mul(0.8)));
    // fade when the camera is inside / very close to a puff
    const dist = length(positionView);
    const near = smoothstep(A.w.mul(0.35), A.w.mul(1.1), dist);
    mat.colorNode = vec4(col, pow(dens, float(1.3)).mul(B.z).mul(near));
    this.sprite = new THREE.Sprite(mat);
    this.sprite.count = count;
    this.sprite.frustumCulled = false;
    this.sprite.renderOrder = 2;
  }
}
