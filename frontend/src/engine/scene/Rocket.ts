// Procedural two-stage vehicle (generic, unbranded). Geometry in body coordinates:
// +Y = body x (nose), y = 0 at stage-1 nozzle exit plane. Dimensions follow data/vehicle.json.
import * as THREE from 'three/webgpu';
import T from '../tsl';
const { uniform, float, vec3, mix, smoothstep, positionLocal, color } = T;
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { VehicleConfig } from '../../physics/config';

function panelTexture(w: number, h: number, rings: number, seams: number, base = '#e9ebee'): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d')!;
  g.fillStyle = base;
  g.fillRect(0, 0, w, h);
  // subtle large-scale paint variation
  for (let i = 0; i < 300; i++) {
    g.fillStyle = `rgba(${200 + Math.random() * 30},${200 + Math.random() * 30},${205 + Math.random() * 30},0.05)`;
    g.fillRect(Math.random() * w, Math.random() * h, 20 + Math.random() * 80, 10 + Math.random() * 60);
  }
  g.strokeStyle = 'rgba(90,95,105,0.55)';
  g.lineWidth = 2;
  for (let i = 1; i < rings; i++) {
    const y = (i / rings) * h;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
  g.strokeStyle = 'rgba(90,95,105,0.25)';
  g.lineWidth = 1;
  for (let i = 0; i < seams; i++) {
    const x = (i / seams) * w;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

function bellProfile(rThroat: number, rExit: number, len: number, n = 24): THREE.Vector2[] {
  // lathe profile from exit (y=0) to throat/chamber (y=len): parabolic-ish bell
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i <= n; i++) {
    const s = i / n; // 0 exit -> 1 throat
    const r = rThroat + (rExit - rThroat) * Math.pow(1 - s, 1.6);
    pts.push(new THREE.Vector2(r, s * len));
  }
  // short combustion chamber
  pts.push(new THREE.Vector2(rThroat * 1.6, len + 0.25));
  pts.push(new THREE.Vector2(rThroat * 1.6, len + 0.65));
  pts.push(new THREE.Vector2(rThroat * 0.6, len + 0.8));
  return pts;
}

export class RocketModel {
  s1 = new THREE.Group();
  s2 = new THREE.Group();
  fairL = new THREE.Group();
  fairR = new THREE.Group();
  /** engine glow 0..1 for each stage (drives emissive heat on nozzles) */
  s1Glow = uniform(0);
  s2Glow = uniform(0);
  s1EnginePositions: THREE.Vector3[] = [];
  s2NozzleExitY: number;
  s1ExitRadius: number;
  s2ExitRadius: number;
  /** true once the textured model has replaced the procedural shells */
  modelLoaded = false;
  private s2Body: THREE.Mesh;

  constructor(cfg: VehicleConfig, detail = 2) {
    const g = cfg.geometry;
    const R = g.radius_m;
    const seg = detail >= 1 ? 64 : 32;
    const white = new THREE.MeshStandardNodeMaterial({ map: panelTexture(512, 1024, 14, 8), roughness: 0.42, metalness: 0.05 });
    const white2 = new THREE.MeshStandardNodeMaterial({ map: panelTexture(512, 512, 6, 8), roughness: 0.42, metalness: 0.05 });
    const dark = new THREE.MeshStandardNodeMaterial({ color: 0x17181b, roughness: 0.55, metalness: 0.2 });
    const metal = new THREE.MeshStandardNodeMaterial({ color: 0x8a8d93, roughness: 0.35, metalness: 0.9 });
    // ---- stage 1 ----
    const s1Len = 42.0 - 2.2;
    const body1 = new THREE.Mesh(new THREE.CylinderGeometry(R, R, s1Len, seg, 1, true), white);
    body1.position.y = 2.2 + s1Len / 2;
    this.s1.add(body1);
    const inter = new THREE.Mesh(new THREE.CylinderGeometry(R * 1.002, R * 1.002, 3.2, seg, 1, true), dark);
    inter.position.y = 42 + 1.6;
    this.s1.add(inter);
    const skirt = new THREE.Mesh(new THREE.CylinderGeometry(R, R * 1.02, 2.2, seg, 1, true), dark);
    skirt.position.y = 1.1;
    this.s1.add(skirt);
    // aft heat shield
    const hs = new THREE.Mesh(new THREE.CircleGeometry(R * 0.99, seg), new THREE.MeshStandardNodeMaterial({ color: 0x2a2b2e, roughness: 0.9 }));
    hs.rotation.x = Math.PI / 2;
    hs.position.y = 1.55;
    this.s1.add(hs);
    // raceway + chines (generic detail)
    const race = new THREE.Mesh(new THREE.BoxGeometry(0.28, s1Len - 2, 0.22), white2);
    race.position.set(0, 2.2 + s1Len / 2, R + 0.08);
    this.s1.add(race);
    // 9 nozzles: 8 around + 1 centre
    const e1 = cfg.engines.stage1;
    this.s1ExitRadius = Math.sqrt(e1.ae_m2 / Math.PI);
    const nozzleMat = this.nozzleMaterial(this.s1Glow);
    const bell1 = new THREE.LatheGeometry(bellProfile(this.s1ExitRadius * 0.38, this.s1ExitRadius, 1.35), 32);
    const pos: THREE.Vector3[] = [new THREE.Vector3(0, 0, 0)];
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      pos.push(new THREE.Vector3(Math.cos(a) * 1.28, 0, Math.sin(a) * 1.28));
    }
    for (const p of pos) {
      const m = new THREE.Mesh(bell1, nozzleMat);
      m.position.copy(p);
      m.castShadow = true;
      this.s1.add(m);
    }
    this.s1EnginePositions = pos;
    // ---- stage 2 ----
    const s2Len = g.stage2.x1 - g.stage2.x0;
    const body2 = new THREE.Mesh(new THREE.CylinderGeometry(R, R, s2Len, seg, 1, true), white2);
    body2.position.y = g.stage2.x0 + s2Len / 2;
    this.s2.add(body2);
    this.s2Body = body2;
    const e2 = cfg.engines.stage2;
    this.s2ExitRadius = Math.sqrt(e2.ae_m2 / Math.PI);
    const bell2Len = 2.9;
    const bell2 = new THREE.LatheGeometry(bellProfile(0.22, this.s2ExitRadius, bell2Len, 32), 48);
    const nz2 = new THREE.Mesh(bell2, this.nozzleMaterial(this.s2Glow, true));
    this.s2NozzleExitY = 42.4;
    nz2.position.y = this.s2NozzleExitY;
    this.s2.add(nz2);
    const s2base = new THREE.Mesh(new THREE.CircleGeometry(R * 0.99, seg), dark);
    s2base.rotation.x = Math.PI / 2;
    s2base.position.y = g.stage2.x0 + 0.01;
    this.s2.add(s2base);
    // payload adapter ring (visible after fairing sep) and payload (generic box satellite)
    const pa = new THREE.Mesh(new THREE.CylinderGeometry(0.9, R, 1.2, seg), metal);
    pa.position.y = g.stage2.x1 + 0.6;
    this.s2.add(pa);
    const sat = new THREE.Mesh(new THREE.BoxGeometry(2.2, 4.5, 2.2), new THREE.MeshStandardNodeMaterial({ color: 0xc9a227, roughness: 0.3, metalness: 0.8 }));
    sat.position.y = g.stage2.x1 + 1.2 + 2.25;
    this.s2.add(sat);
    const top2 = new THREE.Mesh(new THREE.CircleGeometry(R, seg), white2);
    top2.rotation.x = -Math.PI / 2;
    top2.position.y = g.stage2.x1;
    this.s2.add(top2);
    // ---- fairing halves ----
    const fr = g.fairing.radius_m;
    const prof: THREE.Vector2[] = [];
    prof.push(new THREE.Vector2(R, g.fairing.x0 - 0.5));
    prof.push(new THREE.Vector2(fr, g.fairing.x0 + 0.6));
    const cylTop = g.fairing.x0 + 6.0;
    prof.push(new THREE.Vector2(fr, cylTop));
    const ogL = g.fairing.x1 - cylTop;
    for (let i = 1; i <= 20; i++) {
      const s = i / 20;
      const r = fr * Math.sqrt(Math.max(0, 1 - s * s)) * (1 - 0.05 * s);
      prof.push(new THREE.Vector2(Math.max(r, 0.02), cylTop + s * ogL));
    }
    const fairMat = new THREE.MeshStandardNodeMaterial({ map: panelTexture(512, 512, 4, 2), roughness: 0.38, metalness: 0.05, side: THREE.DoubleSide });
    const halfA = new THREE.Mesh(new THREE.LatheGeometry(prof, 48, 0, Math.PI), fairMat);
    const halfB = new THREE.Mesh(new THREE.LatheGeometry(prof, 48, Math.PI, Math.PI), fairMat);
    this.fairL.add(halfA);
    this.fairR.add(halfB);
    for (const grp of [this.s1, this.s2, this.fairL, this.fairR]) {
      grp.traverse((o) => { if ((o as THREE.Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    }
  }

  /**
   * Swap the procedural stage-1, stage-2 and fairing shells for the textured model built by
   * tools/rocket_model/process_meshy.py (same body frame: metres, +Y nose, y = 0 at the stage-1 exit plane).
   * The stage-2 vacuum nozzle, caps, payload and plume anchors stay procedural. Keeps the procedural
   * vehicle if the file is missing or fails to load.
   */
  async loadModel(url: string): Promise<boolean> {
    let gltf: { scene: THREE.Object3D };
    try {
      gltf = await new GLTFLoader().loadAsync(url);
    } catch (e) {
      console.warn('rocket model not loaded, using procedural vehicle', e);
      return false;
    }
    const part = (name: string) => gltf.scene.getObjectByName(name);
    const s1 = part('S1'), noz = part('S1_Nozzles'), s2 = part('S2'), fl = part('FairingL'), fr = part('FairingR');
    if (!s1 || !noz || !s2 || !fl || !fr) return false;
    const toNode = (m: THREE.Material, glow?: ReturnType<typeof uniform>) => {
      const src = m as THREE.MeshStandardMaterial;
      const n = new THREE.MeshStandardNodeMaterial();
      n.setValues({
        name: src.name, color: src.color, map: src.map, normalMap: src.normalMap, normalScale: src.normalScale,
        roughnessMap: src.roughnessMap, metalnessMap: src.metalnessMap, roughness: src.roughness,
        metalness: src.metalness, side: THREE.DoubleSide,
      });
      if (glow) {
        // same heat model as the procedural bells: glow toward the throat, scaled by engine state
        const hot = smoothstep(float(0.4), float(2.4), positionLocal.y);
        n.emissiveNode = mix(color(0x000000), vec3(1.0, 0.32, 0.08).mul(3.0), hot.mul(glow as unknown as THREE.Node));
      }
      return n;
    };
    // a part with two materials (S1: body + interstage) arrives as a group of primitive meshes
    const prep = (part: THREE.Object3D, glow?: ReturnType<typeof uniform>) => {
      part.removeFromParent();
      part.position.set(0, 0, 0);
      part.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map((m) => toNode(m, glow)) : toNode(mesh.material, glow);
        mesh.castShadow = mesh.receiveShadow = true;
      });
      return part;
    };
    let parts: THREE.Object3D[];
    try {
      parts = [prep(s1), prep(noz, this.s1Glow), prep(s2), prep(fl), prep(fr)];
    } catch (e) {
      console.warn('rocket model materials failed, using procedural vehicle', e);
      return false;
    }
    // stage 1: the model replaces everything (the plume anchors are added later by Effects)
    for (const c of [...this.s1.children]) this.s1.remove(c);
    this.s1.add(parts[0], parts[1]);
    const ep = JSON.parse((noz.userData.engine_positions as string) ?? '[]') as number[][];
    if (ep.length === this.s1EnginePositions.length) ep.forEach((p, i) => this.s1EnginePositions[i].set(p[0], p[1], p[2]));
    // stage 2: replace the barrel, keep the vacuum nozzle, end caps and payload
    this.s2.remove(this.s2Body);
    this.s2.add(parts[2]);
    for (const [grp, part] of [[this.fairL, parts[3]], [this.fairR, parts[4]]] as const) {
      for (const c of [...grp.children]) grp.remove(c);
      grp.add(part);
    }
    this.modelLoaded = true;
    return true;
  }

  private nozzleMaterial(glow: ReturnType<typeof uniform>, vac = false) {
    const m = new THREE.MeshStandardNodeMaterial({ color: vac ? 0x3b3027 : 0x2c2b2a, roughness: 0.35, metalness: 0.85, side: THREE.DoubleSide });
    // heat glow concentrated near the throat (top of bell), scaled by engine state
    const y = positionLocal.y;
    const hot = smoothstep(float(0.2), float(vac ? 2.9 : 1.35), y);
    m.emissiveNode = mix(color(0x000000), vec3(1.0, 0.32, 0.08).mul(3.0), hot.mul(glow as unknown as THREE.Node));
    return m;
  }
}
