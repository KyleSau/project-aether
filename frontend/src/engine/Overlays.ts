// Engineering-mode 3D overlays: force / velocity vectors, CG & CP markers (all from sim telemetry).
import * as THREE from 'three/webgpu';
import type { Engine } from './Engine';
import type { V3 } from './CameraRig';

function arrow(color: number) {
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicNodeMaterial({ color, depthTest: false, transparent: true, opacity: 0.9, fog: false });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 1, 8), mat);
  shaft.position.y = 0.5;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.8, 2.4, 12), mat);
  g.add(shaft, head);
  g.userData = { shaft, head };
  g.renderOrder = 50;
  shaft.renderOrder = head.renderOrder = 50;
  return g;
}

function setArrow(g: THREE.Group, origin: THREE.Vector3, dir: THREE.Vector3, length: number, width = 1) {
  const { shaft, head } = g.userData as { shaft: THREE.Mesh; head: THREE.Mesh };
  const L = Math.max(length, 0.01);
  g.position.copy(origin);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  shaft.scale.set(width, Math.max(L - 2.4 * width, 0.01), width);
  head.scale.setScalar(width);
  head.position.y = Math.max(L - 1.2 * width, 1.2 * width);
}

export class Overlays {
  group = new THREE.Group();
  thrust = arrow(0xff7a1a);
  vel = arrow(0x46d2ff);
  drag = arrow(0xff3b6b);
  grav = arrow(0x9dff6a);
  cg: THREE.Mesh;
  cp: THREE.Mesh;
  constructor(scene: THREE.Scene) {
    this.cg = new THREE.Mesh(new THREE.SphereGeometry(1.2, 16, 12), new THREE.MeshBasicNodeMaterial({ color: 0xffff00, depthTest: false, fog: false }));
    this.cp = new THREE.Mesh(new THREE.TorusGeometry(2.4, 0.25, 8, 24), new THREE.MeshBasicNodeMaterial({ color: 0xff00ff, depthTest: false, fog: false }));
    this.cg.renderOrder = this.cp.renderOrder = 51;
    this.group.add(this.thrust, this.vel, this.drag, this.grav, this.cg, this.cp);
    this.group.visible = false;
    scene.add(this.group);
  }

  update(e: Engine, base: V3, quat: THREE.Quaternion, cam: V3) {
    this.group.visible = e.engineering;
    if (!e.engineering) return;
    const tel = e.mission.tel;
    const t = e.mission.view.t;
    const fr = e.frames;
    const o = new THREE.Vector3(base[0] - cam[0], base[1] - cam[1], base[2] - cam[2]);
    const axis = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    const cgP = o.clone().addScaledVector(axis, tel.x_cg);
    const cpP = o.clone().addScaledVector(axis, e.cfg.aero.x_cp_m);
    this.cg.position.copy(cgP);
    this.cp.position.copy(cpP);
    this.cp.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
    const scale = 12 / 1e6; // 12 m per MN
    // thrust (body -> render)
    const tb = tel.thrust_b;
    const tr = new THREE.Vector3(tb[1], tb[0], -tb[2]).applyQuaternion(quat);
    this.thrust.visible = tel.thrust > 1;
    if (this.thrust.visible) setArrow(this.thrust, cgP, tr.clone().normalize(), 8 + tel.thrust * scale, 1.2);
    // aerodynamic force
    const fa = tel.fa;
    const far = new THREE.Vector3(fa[1], fa[0], -fa[2]).applyQuaternion(quat);
    this.drag.visible = far.length() > 100;
    if (this.drag.visible) setArrow(this.drag, cpP, far.clone().normalize(), 4 + far.length() * scale * 8, 0.9);
    // velocity (earth-relative), 20 m per 100 m/s capped
    const vr = [e.mission.view.y[3] + 7.292115e-5 * e.mission.view.y[1], e.mission.view.y[4] - 7.292115e-5 * e.mission.view.y[0], e.mission.view.y[5]];
    const vv = fr.eciToRenderVec(vr, t);
    const vl = Math.hypot(vv[0], vv[1], vv[2]);
    this.vel.visible = vl > 1;
    if (this.vel.visible) setArrow(this.vel, cgP, new THREE.Vector3(vv[0], vv[1], vv[2]), Math.min(10 + vl * 0.02, 160), 0.9);
    // gravity (weight)
    const g = fr.eciToRenderVec(tel.grav, t);
    const gw = Math.hypot(g[0], g[1], g[2]) * tel.mass;
    setArrow(this.grav, cgP, new THREE.Vector3(g[0], g[1], g[2]), 4 + gw * scale, 0.9);
  }
}
