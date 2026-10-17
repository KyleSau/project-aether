// Procedural launch complex: pad mound with flame trench, launch mount, lattice service tower,
// lightning masts, water tower, propellant farm, roads, assembly building, coastline.
// Render frame: X east, Y up, Z south; vehicle base (mount top) at y = 0.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const { texture, uv, float, vec2, mix, mx_noise_float, positionGeometry, vec3, uniform, length, smoothstep } = T;

export const DECK_Y = -8; // pad deck elevation relative to vehicle base
export const GROUND_Y = -16; // surrounding terrain
export const TRENCH_HALF_W = 7;
export const MOUND_HALF = 60;

const loader = new THREE.TextureLoader();
function tex(name: string, repeat: number, srgb = false) {
  const t = loader.load(`./tex/${name}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function pbr(name: string, repeat: number, opts: Partial<THREE.MeshStandardNodeMaterialParameters> = {}) {
  return new THREE.MeshStandardNodeMaterial({
    map: tex(`${name}_diff.jpg`, repeat, true),
    normalMap: tex(`${name}_nor_gl.jpg`, repeat),
    roughnessMap: tex(`${name}_rough.jpg`, repeat),
    ...opts,
  });
}

export class Launchpad {
  group = new THREE.Group();
  accessArm = new THREE.Group();
  /** 0 = arm at vehicle, 1 = fully retracted */
  armRetract = 0;
  lampMats: THREE.MeshStandardNodeMaterial[] = [];
  waterMat: THREE.MeshStandardNodeMaterial;
  lightsOn = uniform(0);

  constructor(detail = 2) {
    const g = this.group;
    const concrete = pbr('concrete_floor_02', 6, { roughness: 1 });
    const concreteBig = pbr('concrete_floor_02', 24, { roughness: 1 });
    const asphalt = pbr('asphalt_02', 40);
    const steel = new THREE.MeshStandardNodeMaterial({ color: 0x6d7178, roughness: 0.55, metalness: 0.7 });
    const steelDark = new THREE.MeshStandardNodeMaterial({ color: 0x3a3d42, roughness: 0.6, metalness: 0.6 });
    const orange = new THREE.MeshStandardNodeMaterial({ color: 0xb8541d, roughness: 0.6, metalness: 0.3 });
    const whitePaint = new THREE.MeshStandardNodeMaterial({ color: 0xe8e8e4, roughness: 0.5, metalness: 0.1 });

    // ---------------- terrain: grass with macro variation ----------------
    const grassMap = tex('sparse_grass_diff.jpg', 1, true);
    const grassNor = tex('sparse_grass_nor_gl.jpg', 1);
    const grass = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 });
    const guv = uv().mul(8000);
    const macro = mx_noise_float(positionGeometry.mul(0.0012)).mul(0.5).add(0.5);
    const macro2 = mx_noise_float(positionGeometry.mul(0.006)).mul(0.5).add(0.5);
    const gcol = texture(grassMap, guv).rgb;
    const nearCol = mix(gcol.mul(vec3(0.62, 0.86, 0.45)), gcol.mul(vec3(0.95, 1.0, 0.62)), macro).mul(macro2.mul(0.35).add(0.75));
    // blend to the Earth-sphere land colour with distance so the terrain disc edge disappears
    const farD = length(vec2(positionGeometry.x, positionGeometry.y));
    grass.colorNode = mix(nearCol, vec3(0.16, 0.19, 0.1).mul(macro.mul(0.4).add(0.8)), smoothstep(2500, 22000, farD));
    grass.normalMap = grassNor;
    grassNor.repeat.set(8000, 8000);
    // terrain disc follows Earth curvature (drop = r^2 / 2R) so it meets the global Earth sphere
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000, 96, 96), grass);
    grass.positionNode = vec3(positionGeometry.x, positionGeometry.y, positionGeometry.z.sub(positionGeometry.x.mul(positionGeometry.x).add(positionGeometry.y.mul(positionGeometry.y)).div(2 * 6378137)));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = GROUND_Y;
    ground.receiveShadow = true;
    g.add(ground);

    // ---------------- pad mound (two halves split by the trench) ----------------
    const h = DECK_Y - GROUND_Y;
    for (const side of [-1, 1]) {
      const w = MOUND_HALF - TRENCH_HALF_W;
      const blk = new THREE.Mesh(new THREE.BoxGeometry(w, h, MOUND_HALF * 2), concrete);
      blk.position.set(side * (TRENCH_HALF_W + w / 2), GROUND_Y + h / 2, 0);
      blk.receiveShadow = true;
      blk.castShadow = true;
      g.add(blk);
      // slope skirt on the outer side
      const slope = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(h * 3, h), 0.5, MOUND_HALF * 2), concreteBig);
      slope.position.set(side * (MOUND_HALF + h * 1.5), GROUND_Y + h / 2, 0);
      slope.rotation.z = side * Math.atan2(h, h * 3) * -1;
      slope.receiveShadow = true;
      g.add(slope);
    }
    // north/south slopes for both halves
    for (const sz of [-1, 1]) for (const sx of [-1, 1]) {
      const w = MOUND_HALF - TRENCH_HALF_W;
      const slope = new THREE.Mesh(new THREE.BoxGeometry(w, 0.5, Math.hypot(h * 3, h)), concreteBig);
      slope.position.set(sx * (TRENCH_HALF_W + w / 2), GROUND_Y + h / 2, sz * (MOUND_HALF + h * 1.5));
      slope.rotation.x = sz * Math.atan2(h, h * 3);
      slope.receiveShadow = true;
      g.add(slope);
    }
    // trench floor + walls
    const trenchFloor = new THREE.Mesh(new THREE.BoxGeometry(TRENCH_HALF_W * 2, 0.4, MOUND_HALF * 2 + 60), concrete);
    trenchFloor.position.set(0, GROUND_Y - 0.2, 0);
    trenchFloor.receiveShadow = true;
    g.add(trenchFloor);
    // deck slab bridging the trench around the mount (with exhaust hole)
    const deckShape = new THREE.Shape();
    deckShape.moveTo(-TRENCH_HALF_W, -16); deckShape.lineTo(TRENCH_HALF_W, -16); deckShape.lineTo(TRENCH_HALF_W, 16); deckShape.lineTo(-TRENCH_HALF_W, 16); deckShape.closePath();
    const hole = new THREE.Path();
    hole.absarc(0, 0, 5.2, 0, Math.PI * 2, true);
    deckShape.holes.push(hole);
    const deck = new THREE.Mesh(new THREE.ExtrudeGeometry(deckShape, { depth: 3, bevelEnabled: false }), concrete);
    deck.rotation.x = -Math.PI / 2;
    deck.position.y = DECK_Y - 3;
    deck.receiveShadow = true; deck.castShadow = true;
    g.add(deck);
    // flame deflector wedge under the mount (splits flow north/south)
    const wedgeShape = new THREE.Shape();
    wedgeShape.moveTo(-9, 0); wedgeShape.lineTo(0, 6.5); wedgeShape.lineTo(9, 0); wedgeShape.closePath();
    const wedge = new THREE.Mesh(new THREE.ExtrudeGeometry(wedgeShape, { depth: TRENCH_HALF_W * 2 - 0.4, bevelEnabled: false }), steelDark);
    wedge.rotation.y = Math.PI / 2;
    wedge.position.set(-TRENCH_HALF_W + 0.2, GROUND_Y, 0);
    g.add(wedge);

    // ---------------- launch mount ----------------
    const mount = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.TorusGeometry(4.6, 0.55, 12, 48), steel);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -0.6;
    mount.add(ring);
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
      const leg = new THREE.Mesh(new THREE.BoxGeometry(1.2, -DECK_Y, 1.2), steel);
      leg.position.set(Math.cos(a) * 5.2, DECK_Y / 2, Math.sin(a) * 5.2);
      mount.add(leg);
    }
    for (let i = 0; i < 4; i++) {
      // hold-down clamps
      const a = (i / 4) * Math.PI * 2;
      const c = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.4, 1.0), orange);
      c.position.set(Math.cos(a) * 2.4, -0.2, Math.sin(a) * 2.4);
      c.lookAt(0, -0.2, 0);
      mount.add(c);
    }
    // tail service masts (umbilicals)
    for (const s of [-1, 1]) {
      const tsm = new THREE.Mesh(new THREE.BoxGeometry(1.6, 9, 1.6), steelDark);
      tsm.position.set(-4.2, 4.5 - 0.5, s * 3.6);
      mount.add(tsm);
    }
    mount.traverse((o) => { o.castShadow = true; o.receiveShadow = true; });
    g.add(mount);

    // ---------------- lattice service tower ----------------
    const towerX = -21, towerH = 96, half = 4.5, level = 4;
    const nLev = Math.floor(towerH / level);
    const beam = new THREE.BoxGeometry(1, 1, 1);
    const mats: THREE.Matrix4[] = [];
    const tmp = new THREE.Object3D();
    const addBeam = (a: THREE.Vector3, b: THREE.Vector3, t: number) => {
      const d = b.clone().sub(a);
      tmp.position.copy(a).add(b).multiplyScalar(0.5);
      tmp.scale.set(t, d.length(), t);
      tmp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
      tmp.updateMatrix();
      mats.push(tmp.matrix.clone());
    };
    const corners = [[-half, -half], [half, -half], [half, half], [-half, half]];
    const y0 = DECK_Y;
    for (const [cx, cz] of corners) addBeam(new THREE.Vector3(towerX + cx, y0, cz), new THREE.Vector3(towerX + cx, y0 + towerH, cz), 0.7);
    for (let l = 0; l <= nLev; l++) {
      const y = y0 + l * level;
      for (let k = 0; k < 4; k++) {
        const [ax, az] = corners[k], [bx, bz] = corners[(k + 1) % 4];
        addBeam(new THREE.Vector3(towerX + ax, y, az), new THREE.Vector3(towerX + bx, y, bz), 0.35);
        if (l < nLev && detail >= 1) {
          addBeam(new THREE.Vector3(towerX + ax, y, az), new THREE.Vector3(towerX + bx, y + level, bz), 0.22);
          if (detail >= 2) addBeam(new THREE.Vector3(towerX + bx, y, bz), new THREE.Vector3(towerX + ax, y + level, az), 0.22);
        }
      }
    }
    // crane boom on top
    addBeam(new THREE.Vector3(towerX - 6, y0 + towerH + 3, 0), new THREE.Vector3(towerX + 14, y0 + towerH + 3, 0), 1.0);
    addBeam(new THREE.Vector3(towerX, y0 + towerH, 0), new THREE.Vector3(towerX, y0 + towerH + 8, 0), 0.8);
    addBeam(new THREE.Vector3(towerX, y0 + towerH + 8, 0), new THREE.Vector3(towerX + 14, y0 + towerH + 3, 0), 0.25);
    const inst = new THREE.InstancedMesh(beam, steel, mats.length);
    mats.forEach((m, i) => inst.setMatrixAt(i, m));
    inst.castShadow = true; inst.receiveShadow = true;
    g.add(inst);
    // platforms + elevator core
    for (let l = 3; l <= nLev; l += 3) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(half * 2 + 1.4, 0.25, half * 2 + 1.4), steelDark);
      p.position.set(towerX, y0 + l * level, 0);
      p.castShadow = true; p.receiveShadow = true;
      g.add(p);
    }
    const core = new THREE.Mesh(new THREE.BoxGeometry(3, towerH, 3), new THREE.MeshStandardNodeMaterial({ color: 0x8f9296, roughness: 0.7, metalness: 0.3 }));
    core.position.set(towerX - 1, y0 + towerH / 2, -1);
    core.castShadow = true;
    g.add(core);
    // lightning rod
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.3, 22, 8), steel);
    rod.position.set(towerX, y0 + towerH + 11, 0);
    g.add(rod);

    // crew access arm (pivots at the tower face)
    const arm = new THREE.Mesh(new THREE.BoxGeometry(14.5, 3.2, 2.6), steelDark);
    arm.position.set(7.25, 0, 0);
    arm.castShadow = true;
    this.accessArm.add(arm);
    const room = new THREE.Mesh(new THREE.BoxGeometry(3.2, 3.6, 3.2), whitePaint);
    room.position.set(12.6, 0, 0);
    this.accessArm.add(room);
    this.accessArm.position.set(towerX + half, 50, 0.0);
    g.add(this.accessArm);

    // ---------------- lightning masts + catenary wires ----------------
    const mastPos = [[-95, -90], [95, -70], [10, 120]];
    const tops: THREE.Vector3[] = [];
    for (const [x, z] of mastPos) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 1.4, 120, 10), whitePaint);
      m.position.set(x, GROUND_Y + 60, z);
      m.castShadow = true;
      g.add(m);
      tops.push(new THREE.Vector3(x, GROUND_Y + 120, z));
    }
    const wirePts: number[] = [];
    for (let i = 0; i < tops.length; i++) {
      const a = tops[i], b = tops[(i + 1) % tops.length];
      for (let k = 0; k < 24; k++) {
        const s0 = k / 24, s1 = (k + 1) / 24;
        for (const s of [s0, s1]) {
          const p = a.clone().lerp(b, s);
          p.y -= 18 * 4 * s * (1 - s);
          wirePts.push(p.x, p.y, p.z);
        }
      }
      // wire to the tower top
      wirePts.push(a.x, a.y, a.z, towerX, y0 + towerH + 21, 0);
    }
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.Float32BufferAttribute(wirePts, 3));
    g.add(new THREE.LineSegments(wg, new THREE.LineBasicNodeMaterial({ color: 0x222222 })));

    // ---------------- water tower ----------------
    const wt = new THREE.Group();
    const tank = new THREE.Mesh(new THREE.SphereGeometry(9, 24, 16), whitePaint);
    tank.position.y = 52;
    wt.add(tank);
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, 48, 6), steel);
      leg.position.set(Math.cos(a) * 6.5, 24, Math.sin(a) * 6.5);
      wt.add(leg);
    }
    const riser = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 48, 10), steel);
    riser.position.y = 24;
    wt.add(riser);
    wt.position.set(-300, GROUND_Y, -260);
    wt.traverse((o) => { o.castShadow = true; o.receiveShadow = true; });
    g.add(wt);

    // ---------------- propellant farm (generic spheres / cylinders) ----------------
    const lox = new THREE.Mesh(new THREE.SphereGeometry(11, 24, 16), whitePaint);
    lox.position.set(-190, GROUND_Y + 13, 140);
    lox.castShadow = true;
    g.add(lox);
    for (let i = 0; i < 3; i++) {
      const c = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, 26, 16), whitePaint);
      c.rotation.z = Math.PI / 2;
      c.position.set(-200, GROUND_Y + 4, -110 + i * 9);
      c.castShadow = true;
      g.add(c);
    }
    // ---------------- assembly building + roads ----------------
    const hangar = new THREE.Mesh(new THREE.BoxGeometry(70, 48, 110), new THREE.MeshStandardNodeMaterial({ color: 0xd6d8da, roughness: 0.6, metalness: 0.2 }));
    hangar.position.set(-820, GROUND_Y + 24, 60);
    hangar.castShadow = true; hangar.receiveShadow = true;
    g.add(hangar);
    const road = new THREE.Mesh(new THREE.PlaneGeometry(760, 14), asphalt);
    road.rotation.x = -Math.PI / 2;
    road.position.set(-430, GROUND_Y + 0.05, 30);
    road.receiveShadow = true;
    g.add(road);
    const apron = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), asphalt);
    apron.rotation.x = -Math.PI / 2;
    apron.position.set(0, GROUND_Y + 0.03, 0);
    apron.receiveShadow = true;
    g.add(apron);

    // ---------------- flood-light masts ----------------
    for (const [x, z] of [[-70, 70], [70, 70], [70, -70], [-70, -70]]) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.6, 40, 8), steel);
      pole.position.set(x, GROUND_Y + 20, z);
      g.add(pole);
      const lm = new THREE.MeshStandardNodeMaterial({ color: 0x222222, roughness: 0.4 });
      lm.emissiveNode = vec3(1.0, 0.92, 0.75).mul(float(this.lightsOn as unknown as THREE.Node).mul(4));
      this.lampMats.push(lm);
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(4, 2, 0.8), lm);
      lamp.position.set(x, GROUND_Y + 40, z);
      lamp.lookAt(0, 0, 0);
      g.add(lamp);
    }

    // ---------------- coastline: beach + ocean (east) ----------------
    const curved = (cx: number) => vec3(positionGeometry.x, positionGeometry.y, positionGeometry.z.sub(positionGeometry.x.add(cx).mul(positionGeometry.x.add(cx)).add(positionGeometry.y.mul(positionGeometry.y)).div(2 * 6378137)));
    const sandMat = new THREE.MeshStandardNodeMaterial({ color: 0xcdbb94, roughness: 0.95 });
    sandMat.positionNode = curved(1500);
    const sand = new THREE.Mesh(new THREE.PlaneGeometry(400, 60000, 1, 96), sandMat);
    sand.rotation.x = -Math.PI / 2;
    sand.position.set(1500, GROUND_Y + 0.08, 0);
    sand.receiveShadow = true;
    g.add(sand);
    this.waterMat = new THREE.MeshStandardNodeMaterial({ color: 0x0d3a52, roughness: 0.08, metalness: 0.0 });
    const wuv = uv().mul(vec2(3000, 6000));
    const n1 = mx_noise_float(vec3(wuv.x, wuv.y, float(0)).mul(0.9));
    const n2 = mx_noise_float(vec3(wuv.x, wuv.y, float(0)).mul(3.1));
    this.waterMat.normalNode = vec3(n1.mul(0.08).add(n2.mul(0.04)), n2.mul(0.06), 1).normalize() as unknown as THREE.Node;
    this.waterMat.positionNode = curved(1700 + 15000);
    const ocean = new THREE.Mesh(new THREE.PlaneGeometry(30000, 60000, 64, 96), this.waterMat);
    ocean.rotation.x = -Math.PI / 2;
    ocean.position.set(1700 + 15000, GROUND_Y + 0.12, 0);
    ocean.receiveShadow = true;
    g.add(ocean);
  }

  update(dt: number, retract: boolean) {
    const target = retract ? 1 : 0;
    this.armRetract += Math.sign(target - this.armRetract) * Math.min(Math.abs(target - this.armRetract), dt / 8);
    this.accessArm.rotation.y = this.armRetract * 1.9;
  }
}
