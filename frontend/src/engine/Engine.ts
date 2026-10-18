// Renderer / scene orchestration. three.js WebGPURenderer (WebGPU backend, or its WebGL2
// backend when WebGPU is unavailable or ?renderer=webgl). Floating origin: the camera sits at
// the origin and every object is placed at (world - cameraWorld) computed in float64.
import * as THREE from 'three/webgpu';
import { pass, uniform, float, vec3 } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import type { VehicleConfig } from '../physics/config';
import { NSTATE, ORBIT, S1_BURN, S2_BURN, PAD, COAST_SEP, RE } from '../physics/dynamics';
import { keplerPath } from '../physics/orbit';
import { Frames } from './frames';
import { MissionController, COUNTDOWN_S } from './MissionController';
import { CameraRig, type CamMode, type V3 } from './CameraRig';
import { RocketModel } from './scene/Rocket';
import { Launchpad, GROUND_Y } from './scene/Launchpad';
import { Earth } from './scene/Earth';
import { SkyModel, R_GROUND_KM } from './scene/Sky';
import { Effects } from './Effects';
import { Clouds } from './scene/Clouds';
import { Condensation } from './scene/Condensation';
import { LaunchAudio } from './Audio';
import { AnalyticParticles, ComputeParticles } from './scene/Particles';
import { AutoScaler, detectTier, makeSettings, type QualitySettings } from './quality';
import type { Stepper } from '../physics/sim';

export type TimeOfDay = 'day' | 'dawn' | 'night';

export interface EngineInfo {
  backend: string;
  tier: string;
  physics: string;
  fps: number;
  frameMs: number;
  particles: number;
  particleMode: string;
  vehicleModel?: string;
  motionBlur?: boolean;
  drawCalls: number;
  triangles: number;
  pixelRatio: number;
}

interface Placed { obj: THREE.Object3D; world: V3 }

export class Engine {
  renderer!: THREE.WebGPURenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 0.5, 5e7);
  mission: MissionController;
  rig = new CameraRig();
  frames: Frames;
  q!: QualitySettings;
  info: EngineInfo = { backend: '?', tier: '?', physics: 'TS', fps: 0, frameMs: 0, particles: 0, particleMode: '', drawCalls: 0, triangles: 0, pixelRatio: 1 };
  rocket!: RocketModel;
  pad!: Launchpad;
  earth!: Earth;
  sky!: SkyModel;
  fx!: Effects;
  sun = new THREE.DirectionalLight(0xffffff, 3);
  hemi = new THREE.HemisphereLight(0x9ab8ff, 0x3a3428, 0.6);
  floods: THREE.SpotLight[] = [];
  pipeline!: THREE.RenderPipeline;
  bloomNode: ReturnType<typeof bloom> | null = null;
  exposure = uniform(1);
  private placed: Placed[] = [];
  private camWorld: V3 = [0, 0, 0];
  private last = performance.now();
  private fpsAcc = 0;
  private fpsN = 0;
  private scaler = new AutoScaler(true);
  private trajLine!: THREE.Line;
  private trajCount = 0;
  private orbitLine!: THREE.Line;
  private orbitTimer = 0;
  private lastSepCount = 0;
  private eyeAdapt = 1;
  timeOfDay: TimeOfDay = 'dawn';
  mblurAmount = uniform(0.5);
  engineering = false;
  onFrame?: () => void;
  private shakeT = 0;
  overlays!: import('./Overlays').Overlays;
  clouds!: Clouds;
  vapor!: Condensation;
  audio = new LaunchAudio();

  constructor(private canvas: HTMLCanvasElement, public cfg: VehicleConfig, makeStepper: (c: VehicleConfig) => Stepper | undefined, physicsLabel: string) {
    this.mission = new MissionController(cfg, makeStepper);
    this.frames = new Frames(cfg, this.mission.sim.t);
    this.info.physics = physicsLabel;
  }

  async init() {
    const params = new URLSearchParams(location.search);
    const forceWebGL = params.get('renderer') === 'webgl' || !('gpu' in navigator);
    this.renderer = new THREE.WebGPURenderer({ canvas: this.canvas, antialias: false, forceWebGL, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
    await this.renderer.init();
    const backend = (this.renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'WebGPU' : 'WebGL2';
    this.info.backend = backend;
    this.q = makeSettings(detectTier(backend === 'WebGPU'));
    const prOverride = parseFloat(params.get('pr') ?? '');
    if (prOverride > 0) { this.q.pixelRatio = prOverride; this.scaler.enabled = false; }
    this.info.tier = this.q.tier;
    this.renderer.setPixelRatio(this.q.pixelRatio);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.shadowMap.enabled = this.q.shadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // ---------- sky + fog ----------
    this.sky = new SkyModel(this.q.skySamples);
    this.scene.backgroundNode = this.sky.backgroundNode;
    this.scene.fogNode = this.sky.fogNode;

    // ---------- lights ----------
    this.sun.castShadow = this.q.shadows;
    this.sun.shadow.mapSize.set(this.q.shadowMapSize, this.q.shadowMapSize);
    const sc = this.sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = -160; sc.right = 160; sc.top = 160; sc.bottom = -160; sc.near = 10; sc.far = 6000;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.05;
    this.scene.add(this.sun, this.sun.target, this.hemi);

    // ---------- world ----------
    this.earth = new Earth(this.q.tier === 'low' ? 160 : 320);
    this.earth.mesh.scale.setScalar(RE + 0);
    this.scene.add(this.earth.mesh);
    this.pad = new Launchpad(this.q.towerDetail);
    this.scene.add(this.pad.group);
    this.place(this.pad.group, [0, 0, 0]);
    this.rocket = new RocketModel(this.cfg, this.q.towerDetail);
    if (params.get('model') !== 'procedural') await this.rocket.loadModel('./models/rocket.glb');
    this.info.vehicleModel = this.rocket.modelLoaded ? 'textured' : 'procedural';
    this.scene.add(this.rocket.s1, this.rocket.s2, this.rocket.fairL, this.rocket.fairR);
    const useCompute = backend === 'WebGPU' && params.get('particles') !== 'analytic';
    const parts = useCompute ? new ComputeParticles(this.q.maxParticles) : new AnalyticParticles(this.q.maxParticles);
    this.info.particleMode = parts.kind;
    this.fx = new Effects(this.cfg, parts, this.rocket.s1, this.rocket.s2, this.rocket.s1EnginePositions, this.rocket.s2NozzleExitY);
    this.scene.add(this.fx.particles.sprite);
    this.place(this.fx.particles.sprite, [0, 0, 0]);
    this.clouds = new Clouds(this.q.cloudCount);
    this.scene.add(this.clouds.sprite);
    this.place(this.clouds.sprite, [0, 0, 0]);
    this.vapor = new Condensation();
    this.rocket.s2.add(this.vapor.mesh);
    this.fx.engineLight.castShadow = this.q.engineLightShadows;
    this.fx.engineLight.shadow.mapSize.set(512, 512);
    this.scene.add(this.fx.engineLight);
    for (const [x, z] of [[-70, 70], [70, 70]]) {
      const s = new THREE.SpotLight(0xfff0d8, 0, 0, 0.35, 0.5, 2);
      s.userData.world = [x, GROUND_Y + 40, z];
      s.userData.target = [0, 40, 0];
      this.floods.push(s);
      this.scene.add(s, s.target);
    }
    // ---------- trajectory + orbit lines ----------
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(30000 * 3), 3));
    tg.setDrawRange(0, 0);
    this.trajLine = new THREE.Line(tg, new THREE.LineBasicNodeMaterial({ color: 0x46d2ff, transparent: true, opacity: 0.85, fog: false }));
    this.trajLine.frustumCulled = false;
    this.scene.add(this.trajLine);
    this.place(this.trajLine, [0, 0, 0]);
    const og = new THREE.BufferGeometry();
    og.setAttribute('position', new THREE.BufferAttribute(new Float32Array(600 * 3), 3));
    og.setDrawRange(0, 0);
    this.orbitLine = new THREE.Line(og, new THREE.LineBasicNodeMaterial({ color: 0xffc04d, transparent: true, opacity: 0.9, fog: false }));
    this.orbitLine.frustumCulled = false;
    this.scene.add(this.orbitLine);
    this.place(this.orbitLine, [0, 0, 0]);
    const { Overlays } = await import('./Overlays');
    this.overlays = new Overlays(this.scene);

    // ---------- post-processing ----------
    this.pipeline = new THREE.RenderPipeline(this.renderer);
    const scenePass = pass(this.scene, this.camera, { samples: this.q.msaa } as never);
    let col = scenePass.getTextureNode('output');
    if (this.q.motionBlur && params.get('mblur') === '1') {
      try {
        const T = (await import('./tsl')).default;
        const { motionBlur } = await import('three/examples/jsm/tsl/display/MotionBlur.js');
        scenePass.setMRT(T.mrt({ output: T.output, velocity: T.velocity }));
        const vel = scenePass.getTextureNode('velocity');
        col = motionBlur(col, vel.mul(this.mblurAmount), T.int(8)) as never;
        this.info.motionBlur = true;
      } catch (e) { console.warn('motion blur unavailable', e); }
    }
    if (this.q.bloom) {
      this.bloomNode = bloom(col, 0.7, 0.55, 0.9);
      this.pipeline.outputNode = col.add(this.bloomNode).mul(this.exposure) as never;
    } else this.pipeline.outputNode = col.mul(this.exposure) as never;

    this.setTimeOfDay(this.timeOfDay);
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.installControls();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  private place(obj: THREE.Object3D, world: V3) {
    this.placed.push({ obj, world });
  }

  setTimeOfDay(tod: TimeOfDay) {
    this.timeOfDay = tod;
    const elev = tod === 'day' ? 38 : tod === 'dawn' ? 6 : -14;
    const az = tod === 'day' ? 140 : tod === 'dawn' ? 100 : 255; // degrees from north, clockwise (dawn: sun rising in the east, downrange)
    const e = (elev * Math.PI) / 180, a = (az * Math.PI) / 180;
    // render frame: x east, y up, z south
    const dir = new THREE.Vector3(Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a)).normalize();
    this.sky.uSunDir.value.copy(dir);
    this.earth.uSunDir.value.copy(dir);
    this.fx.particles.uSunDir.value.copy(dir);
    this.clouds.uSunDir.value.copy(dir);
    const day = THREE.MathUtils.smoothstep(Math.sin(e), -0.1, 0.2);
    const warm = Math.exp(-Math.max(Math.sin(e), 0.02) * 8);
    const sunCol = new THREE.Color(1, 0.93 - 0.35 * warm, 0.85 - 0.6 * warm);
    this.sun.color.copy(sunCol);
    this.sun.intensity = 3.2 * day;
    this.hemi.intensity = 0.15 + 0.6 * day;
    this.hemi.color.set(day > 0.5 ? 0x9ab8ff : 0x3a4a78);
    this.fx.particles.uSunColor.value.copy(sunCol).multiplyScalar(day * 1.2);
    this.fx.particles.uAmbient.value.set(0.22, 0.27, 0.36).multiplyScalar(0.12 + 0.88 * day);
    this.clouds.uSunColor.value.copy(sunCol).multiplyScalar(day * 1.1 + 0.02);
    this.clouds.uAmbient.value.set(0.35, 0.42, 0.55).multiplyScalar(0.08 + 0.92 * day);
    this.sky.uFogColor.value.copy(SkyModel.horizonColor(e));
    this.pad.lightsOn.value = tod === 'night' ? 1 : tod === 'dawn' ? 0.6 : 0;
    for (const f of this.floods) f.intensity = tod === 'night' ? 4e4 : tod === 'dawn' ? 8e3 : 0;
    this.earth.uSunI.value = 1;
    this.scene.environment = makeEnvironment(dir, sunCol, day);
    this.scene.environmentIntensity = 0.6;
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(this.q.pixelRatio * this.scaler.scale);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.info.pixelRatio = this.q.pixelRatio * this.scaler.scale;
  }

  // ---------------------------------------------------------------- input (free / orbital camera)
  private installControls() {
    const el = this.canvas;
    const pts = new Map<number, { x: number; y: number }>();
    let pinch = 0;
    el.addEventListener('pointerdown', (e) => { pts.set(e.pointerId, { x: e.clientX, y: e.clientY }); el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointerup', (e) => { pts.delete(e.pointerId); pinch = 0; });
    el.addEventListener('pointercancel', (e) => { pts.delete(e.pointerId); pinch = 0; });
    el.addEventListener('pointermove', (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      if (pts.size === 1) {
        if (this.rig.mode !== 'free' && this.rig.mode !== 'orbital') this.rig.setMode('free');
        this.rig.rotate(e.clientX - p.x, e.clientY - p.y);
      } else if (pts.size === 2) {
        p.x = e.clientX; p.y = e.clientY;
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0) {
          if (this.rig.mode !== 'free' && this.rig.mode !== 'orbital') this.rig.setMode('free');
          this.rig.zoom(pinch / d);
        }
        pinch = d;
        return;
      }
      p.x = e.clientX; p.y = e.clientY;
    });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.rig.mode !== 'free' && this.rig.mode !== 'orbital') this.rig.setMode('free');
      this.rig.zoom(Math.exp(e.deltaY * 0.001));
    }, { passive: false });
  }

  launch() {
    this.audio.start();
    this.mission.launch();
  }

  setCamera(m: CamMode) {
    this.rig.setMode(m);
    if (m === 'free') { this.rig.dist = this.mission.view.phase >= 3 ? 60 : 140; }
  }

  restart() {
    this.mission.reset();
    this.fx.reset();
    this.trajCount = 0;
    this.lastSepCount = 0;
    (this.orbitLine.geometry as THREE.BufferGeometry).setDrawRange(0, 0);
  }

  // ---------------------------------------------------------------- per frame
  private bench: { dts: number[]; cpu: number[]; until: number; resolve: (v: unknown) => void } | null = null;
  /** Measure frame statistics for `seconds` (used by the benchmark procedure in docs/AETHER.md). */
  benchmark(seconds: number): Promise<unknown> {
    return new Promise((resolve) => { this.bench = { dts: [], cpu: [], until: performance.now() + seconds * 1000, resolve }; });
  }
  private finishBench() {
    const b = this.bench!;
    this.bench = null;
    const sorted = [...b.dts].sort((x, y) => x - y);
    const mean = b.dts.reduce((a, c) => a + c, 0) / b.dts.length;
    const cpuMean = b.cpu.reduce((a, c) => a + c, 0) / b.cpu.length;
    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    const ri = this.renderer.info as unknown as { memory?: { geometries?: number; textures?: number } };
    // GPU memory estimate: particle buffers + colour/depth targets (approximate, see docs)
    const w = this.renderer.domElement.width, h = this.renderer.domElement.height;
    const particleBytes = this.fx.particles.capacity * 16 * (this.info.particleMode.startsWith('WebGPU') ? 6 : 4);
    const targets = w * h * (8 /*rgba16f*/ + 4 /*depth*/) * Math.max(1, this.q.msaa) + w * h * 8 * 1.5 /*bloom chain*/;
    b.resolve({
      ...this.info, frames: b.dts.length, meanFrameMs: mean, fps: 1000 / mean, p95FrameMs: sorted[Math.floor(sorted.length * 0.95)],
      p99FrameMs: sorted[Math.floor(sorted.length * 0.99)], cpuFrameMs: cpuMean, width: w, height: h,
      jsHeapMB: mem ? mem.usedJSHeapSize / 1048576 : null, geometries: ri.memory?.geometries, textures: ri.memory?.textures,
      gpuMemEstimateMB: (particleBytes + targets) / 1048576,
    });
  }

  private frame() {
    const now = performance.now();
    if (this.bench) {
      this.bench.dts.push(now - this.last);
      if (now > this.bench.until) this.finishBench();
    }
    const dtReal = Math.min((now - this.last) / 1000, 0.1);
    const frameMs = now - this.last;
    this.last = now;
    this.fpsAcc += frameMs; this.fpsN++;
    if (this.fpsAcc > 500) {
      this.info.fps = (1000 * this.fpsN) / this.fpsAcc;
      this.info.frameMs = this.fpsAcc / this.fpsN;
      this.fpsAcc = 0; this.fpsN = 0;
    }
    if (this.scaler.sample(frameMs)) this.resize();
    const M = this.mission;
    const metBefore = M.met;
    M.update(dtReal);
    const v = M.view;
    const ft = v.t;
    const tel = M.tel;
    const fr = this.frames;

    // vehicle pose from simulation
    const base = fr.eciToRenderPos(v.y, ft) as V3;
    const quat = fr.bodyQuatToRender(v.y.subarray(6, 10), ft, new THREE.Quaternion());
    const axis = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
    const side = new THREE.Vector3(1, 0, 0).applyQuaternion(quat);
    const separated = v.phase > COAST_SEP;
    // spent stage + fairings
    let s1Base = base, s1Quat = quat;
    if (separated && M.s1) {
      s1Base = fr.eciToRenderPos(M.s1.y, ft) as V3;
      s1Quat = fr.bodyQuatToRender(M.s1.y.subarray(6, 10), ft, new THREE.Quaternion());
    }
    this.setPose(this.rocket.s1, s1Base, s1Quat);
    this.rocket.s1.visible = !separated || !!(M.s1 && M.s1.active);
    this.setPose(this.rocket.s2, base, quat);
    if (v.fairing || !M.fairingL) {
      this.setPose(this.rocket.fairL, base, quat);
      this.setPose(this.rocket.fairR, base, quat);
      this.rocket.fairL.visible = this.rocket.fairR.visible = v.fairing;
    } else {
      for (const [g, d] of [[this.rocket.fairL, M.fairingL], [this.rocket.fairR, M.fairingR!]] as const) {
        this.setPose(g, fr.eciToRenderPos(d.y, ft) as V3, fr.bodyQuatToRender(d.y.subarray(6, 10), ft, new THREE.Quaternion()));
        g.visible = d.active;
      }
    }
    // Earth-relative velocity in render frame
    const vr = [v.y[3] + 7.292115e-5 * v.y[1], v.y[4] - 7.292115e-5 * v.y[0], v.y[5]];
    const velR = fr.eciToRenderVec(vr, ft) as V3;

    // effects
    const s1Burning = v.phase === S1_BURN || v.phase === PAD;
    const s2Burning = v.phase === S2_BURN;
    const sepCount = M.sim.events.filter((e) => e.name === 'STAGE_SEP').length;
    const sepEvent = sepCount > this.lastSepCount && separated;
    this.lastSepCount = sepCount;
    const dtMission = M.paused ? 0 : M.met - metBefore;
    const ax: V3 = [axis.x, axis.y, axis.z];
    const s2Base: V3 = [base[0] + ax[0] * this.rocket.s2NozzleExitY, base[1] + ax[1] * this.rocket.s2NozzleExitY, base[2] + ax[2] * this.rocket.s2NozzleExitY];
    const lagCam = this.camWorld;
    this.fx.update({
      t: M.state === 'flight' || M.replay ? ft : M.met, dt: dtMission, base, axis: ax, vel: velR, alt: tel.h, pa: tel.P,
      thrustS1: s1Burning ? tel.thrust : 0, thrS1: s1Burning ? tel.throttle : 0, s1Burning: s1Burning && M.state !== 'idle' && (M.state !== 'countdown' || M.met > -3),
      thrustS2: s2Burning ? tel.thrust : 0, thrS2: s2Burning ? tel.throttle : 0, s2Burning,
      countdown: M.state === 'idle' ? COUNTDOWN_S + 1 : -M.met, mach: tel.mach, qbar: tel.qbar, s2Base, s2Axis: ax, sepEvent,
    }, lagCam);
    this.rocket.s1Glow.value = s1Burning ? tel.throttle : Math.max(0, (this.rocket.s1Glow.value as number) - dtReal * 0.2);
    this.rocket.s2Glow.value = s2Burning ? tel.throttle : Math.max(0, (this.rocket.s2Glow.value as number) - dtReal * 0.1);
    this.pad.update(dtReal * (M.paused ? 0 : M.warp), M.state !== 'idle');
    this.vapor.update(tel.mach, tel.h, tel.qbar);
    this.vapor.mesh.visible = this.vapor.mesh.visible && v.phase <= S2_BURN;
    this.audio.update(this.fx.acoustic, this.fx.nearField, M.paused);

    // camera
    const out = this.rig.update({
      base, axis: ax, side: [side.x, side.y, side.z], s1Base: separated ? s1Base : null, separated,
      earthCenter: fr.earthCenter(), velDir: velR, alt: tel.h,
    }, dtReal);
    // keep cameras above terrain
    if (this.rig.mode === 'free' && out.pos[1] < GROUND_Y + 2 && tel.h < 5000) out.pos[1] = GROUND_Y + 2;
    this.camWorld = out.pos;
    this.camera.position.set(0, 0, 0);
    this.camera.up.set(out.up[0], out.up[1], out.up[2]);
    // camera shake (approximate model, see Effects.update)
    this.shakeT += dtReal;
    const sh = this.fx.shake * (M.paused ? 0 : 1);
    const jitter = (k: number) => (Math.sin(this.shakeT * (37 + k * 11)) * 0.6 + Math.sin(this.shakeT * (71 + k * 7)) * 0.4);
    const tgt = new THREE.Vector3(out.target[0] - out.pos[0], out.target[1] - out.pos[1], out.target[2] - out.pos[2]);
    const tl = tgt.length();
    const amp = sh * 0.0025 * tl * (this.camera.fov / 40);
    tgt.x += jitter(1) * amp; tgt.y += jitter(2) * amp; tgt.z += jitter(3) * amp;
    this.camera.lookAt(tgt);
    this.camera.fov = out.fov;
    this.camera.near = Math.max(0.3, Math.min(50, (tel.h > 1e5 && this.rig.mode === 'orbital') ? 100 : 0.3));
    this.camera.updateProjectionMatrix();

    // floating origin
    const c = this.camWorld;
    for (const p of this.placed) p.obj.position.set(p.world[0] - c[0], p.world[1] - c[1], p.world[2] - c[2]);
    const ec = fr.earthCenter();
    this.earth.mesh.position.set(ec[0] - c[0], ec[1] - c[1] + GROUND_Y - 3, ec[2] - c[2]);
    const lw = this.fx.engineLight.userData.world as V3;
    this.fx.engineLight.position.set(lw[0] - c[0], lw[1] - c[1], lw[2] - c[2]);
    for (const f of this.floods) {
      const w = f.userData.world as V3, t = f.userData.target as V3;
      f.position.set(w[0] - c[0], w[1] - c[1], w[2] - c[2]);
      f.target.position.set(t[0] - c[0], t[1] - c[1], t[2] - c[2]);
    }
    // sun + shadow camera follow the focus (vehicle or pad)
    const focus: V3 = tel.h < 3000 ? [base[0] * 0.5, base[1] * 0.5 + 20, base[2] * 0.5] : base;
    const sd = this.sky.uSunDir.value;
    const fz = [focus[0] - c[0], focus[1] - c[1], focus[2] - c[2]];
    this.sun.target.position.set(fz[0], fz[1], fz[2]);
    this.sun.position.set(fz[0] + sd.x * 3000, fz[1] + sd.y * 3000, fz[2] + sd.z * 3000);
    const ext = tel.h < 1500 ? 170 : 120;
    const scam = this.sun.shadow.camera as THREE.OrthographicCamera;
    if (scam.right !== ext) { scam.left = -ext; scam.right = ext; scam.top = ext; scam.bottom = -ext; scam.updateProjectionMatrix(); }

    // sky uniforms: camera relative to Earth centre in km
    const camRelEarth = [c[0] - ec[0], c[1] - (ec[1] + GROUND_Y - 3), c[2] - ec[2]];
    const ckm = new THREE.Vector3(camRelEarth[0], camRelEarth[1], camRelEarth[2]).multiplyScalar(1e-3);
    this.sky.uCamKm.value.copy(ckm);
    const camAlt = ckm.length() * 1000 - RE;
    this.sky.uCamAltM.value = camAlt;
    this.sky.uUp.value.copy(ckm).normalize();
    this.earth.uCamAlt.value = camAlt;
    void R_GROUND_KM;

    // trajectory line from recorded simulation history (ECI -> render at each frame's own time)
    this.updateTrajectory(c);
    this.orbitTimer -= dtReal;
    if (this.orbitTimer <= 0) { this.updateOrbitLine(v.y, ft); this.orbitTimer = 0.5; }
    this.orbitLine.visible = v.phase >= S2_BURN || this.engineering || this.rig.mode === 'orbital';
    this.trajLine.visible = this.engineering || this.rig.mode === 'orbital' || this.rig.mode === 'aerial' || this.rig.mode === 'tracking';
    this.overlays.update(this, base, quat, c);

    // exposure adaptation (approximate eye/camera adaptation to sky + engine light at camera)
    const dCam = Math.max(20, Math.hypot(c[0] - base[0], c[1] - base[1], c[2] - base[2]));
    const sunUp = Math.max(0.02, THREE.MathUtils.smoothstep(this.sky.uSunDir.value.y, -0.15, 0.25));
    const engineLum = Math.min(2.5, (this.fx.engineLight.intensity / (dCam * dCam)) * 0.15 * (this.rig.mode === 'onboard' ? 0.3 : 1));
    const target = 1 / (0.35 + 0.65 * sunUp + engineLum) * (camAlt > 60000 ? 0.85 : 1);
    this.eyeAdapt += (target - this.eyeAdapt) * (1 - Math.exp(-dtReal * 1.5));
    this.exposure.value = this.eyeAdapt;

    this.fx.particles.step(this.renderer, Math.max(0, Math.min(dtMission, 5)));
    this.pipeline.render();
    const ri = this.renderer.info.render as unknown as { drawCalls?: number; calls?: number; triangles?: number };
    this.info.drawCalls = ri.drawCalls ?? ri.calls ?? 0;
    this.info.triangles = ri.triangles ?? 0;
    this.info.particles = this.fx.particles.capacity;
    this.onFrame?.();
    if (this.bench) this.bench.cpu.push(performance.now() - now);
  }

  private setPose(g: THREE.Object3D, pos: V3, q: THREE.Quaternion) {
    const c = this.camWorld;
    g.position.set(pos[0] - c[0], pos[1] - c[1], pos[2] - c[2]);
    g.quaternion.copy(q);
  }

  private updateTrajectory(_c: V3) {
    const H = this.mission.sim.history;
    const viewT = this.mission.view.t;
    const geo = this.trajLine.geometry as THREE.BufferGeometry;
    const attr = geo.getAttribute('position') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const max = arr.length / 3;
    const stride = Math.max(1, Math.ceil(H.length / max));
    let n = 0;
    if (this.trajCount > H.length) this.trajCount = 0;
    for (let i = 0; i < H.length && n < max; i += stride) {
      if (H[i].t > viewT) break;
      const p = this.frames.eciToRenderPos(H[i].y, H[i].t);
      arr[n * 3] = p[0]; arr[n * 3 + 1] = p[1]; arr[n * 3 + 2] = p[2];
      n++;
    }
    geo.setDrawRange(0, n);
    attr.needsUpdate = true;
    this.trajCount = n;
  }

  private updateOrbitLine(y: Float64Array, t: number) {
    const geo = this.orbitLine.geometry as THREE.BufferGeometry;
    const attr = geo.getAttribute('position') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const el = this.mission.orbitEl;
    if (!el || this.mission.tel.h < 20000) { geo.setDrawRange(0, 0); return; }
    const n = arr.length / 3;
    const dur = el.period && isFinite(el.period) ? el.period : 3000;
    const path = keplerPath(y.subarray(0, 3), y.subarray(3, 6), n, dur);
    let k = 0;
    for (let i = 0; i < n; i++) {
      const r = [path[i * 3], path[i * 3 + 1], path[i * 3 + 2]];
      if (Math.hypot(r[0], r[1], r[2]) < RE) break; // stop at ground impact (suborbital prediction)
      const p = this.frames.eciToRenderPos(r, t); // inertial orbit drawn at the current Earth orientation
      arr[k * 3] = p[0]; arr[k * 3 + 1] = p[1]; arr[k * 3 + 2] = p[2];
      k++;
    }
    geo.setDrawRange(0, k);
    attr.needsUpdate = true;
  }

  stateVector(): Float64Array { return this.mission.view.y.slice(0, NSTATE); }
  phaseIsOrbit() { return this.mission.view.phase === ORBIT; }
}

/** Procedural HDR equirect environment: sky gradient + sun lobe + ground (used for PBR IBL). */
function makeEnvironment(sunDir: THREE.Vector3, sunCol: THREE.Color, day: number): THREE.DataTexture {
  const W = 128, H = 64;
  const data = new Float32Array(W * H * 4);
  const hz = SkyModel.horizonColor(Math.asin(sunDir.y));
  for (let j = 0; j < H; j++) {
    const v = 1 - (j + 0.5) / H;
    const phi = v * Math.PI - Math.PI / 2; // elevation
    for (let i = 0; i < W; i++) {
      const th = ((i + 0.5) / W) * Math.PI * 2;
      const d = new THREE.Vector3(Math.cos(phi) * Math.sin(th), Math.sin(phi), Math.cos(phi) * Math.cos(th));
      const o = (j * W + i) * 4;
      let r: number, g: number, b: number;
      if (phi > 0) {
        const t = Math.pow(1 - phi / (Math.PI / 2), 3);
        r = (0.12 + (hz.r - 0.12) * t) * (0.15 + 0.85 * day);
        g = (0.25 + (hz.g - 0.25) * t) * (0.15 + 0.85 * day);
        b = (0.6 + (hz.b - 0.6) * t) * (0.15 + 0.85 * day);
      } else {
        r = 0.09 * day + 0.004; g = 0.085 * day + 0.004; b = 0.07 * day + 0.005;
      }
      const s = Math.max(0, d.dot(sunDir));
      const lobe = Math.pow(s, 64) * 30 * day + Math.pow(s, 6) * 0.6 * day;
      data[o] = r + lobe * sunCol.r; data[o + 1] = g + lobe * sunCol.g; data[o + 2] = b + lobe * sunCol.b; data[o + 3] = 1;
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.needsUpdate = true;
  return tex;
}

export { float, vec3 };
