import { useEffect, useRef, useState, useCallback } from 'react';
import type { Engine, TimeOfDay } from './engine/Engine';
import { CAM_MODES, type CamMode } from './engine/CameraRig';
import { PHASE_NAMES, ORBIT, S2_DEPLETED } from './physics/dynamics';
import { COUNTDOWN_S } from './engine/MissionController';
import { EngPanel } from './ui/EngPanel';
import { boot } from './boot';

const fmt = (v: number, d = 0) => (isFinite(v) ? v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');
const fmtMet = (t: number) => {
  const s = t < 0 ? 'T-' : 'T+';
  const a = Math.abs(t);
  const m = Math.floor(a / 60), sec = a - m * 60;
  return `${s}${String(m).padStart(2, '0')}:${sec.toFixed(1).padStart(4, '0')}`;
};
const PHASE_LABEL: Record<string, string> = {
  PAD: 'ON PAD', S1_BURN: 'STAGE 1 BURN', COAST_SEP: 'MECO / COAST', COAST_S2: 'STAGE SEPARATION', S2_BURN: 'STAGE 2 BURN', ORBIT: 'ORBIT', S2_DEPLETED: 'S2 DEPLETED',
};
const WARPS = [0.25, 1, 2, 5, 10, 50];

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const [cam, setCam] = useState<CamMode>('pad');
  const [telOpen, setTelOpen] = useState(() => window.innerWidth > 700);
  const [eng, setEng] = useState(false);
  const [tod, setTod] = useState<TimeOfDay>('dusk');
  const [help, setHelp] = useState(false);

  useEffect(() => {
    let alive = true;
    boot(canvasRef.current!).then((e) => { if (alive) { setEngine(e); (window as unknown as { aether: Engine }).aether = e; } }).catch((e) => { console.error(e); setErr(String(e?.stack || e)); });
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 100);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    if (!engine) return;
    const k = (ev: KeyboardEvent) => {
      if (ev.key === ' ') { engine.mission.state === 'idle' ? engine.mission.launch() : (engine.mission.paused = !engine.mission.paused); }
      const i = parseInt(ev.key, 10);
      if (i >= 1 && i <= CAM_MODES.length) { engine.setCamera(CAM_MODES[i - 1].id); setCam(CAM_MODES[i - 1].id); }
      if (ev.key === 'e') { engine.engineering = !engine.engineering; setEng(engine.engineering); }
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [engine]);

  const M = engine?.mission;
  const tel = M?.tel;
  const setCamera = useCallback((m: CamMode) => { engine?.setCamera(m); setCam(m); }, [engine]);
  // the canvas drag switches to free camera; keep UI in sync
  const camNow = engine?.rig.mode ?? cam;

  const lastEvent = M?.sim.events.length ? M.sim.events[M.sim.events.length - 1] : null;
  const phase = M ? PHASE_NAMES[M.view.phase] : 'PAD';
  const el = M?.orbitEl;
  const inOrbit = M && (M.view.phase === ORBIT) && el && el.periapsis_alt > 150e3;
  const metShown = M ? (M.state === 'idle' ? -COUNTDOWN_S : M.met) : -COUNTDOWN_S;
  const prop1 = M ? Math.max(0, M.view.y[13]) / (engine!.cfg.mass.stage1_prop_kg) : 1;
  const prop2 = M ? Math.max(0, M.view.y[14]) / (engine!.cfg.mass.stage2_prop_kg) : 1;
  const att = M ? attitude(engine!) : { pitch: 90, heading: 90 };

  return (
    <div className="app">
      <canvas ref={canvasRef} className="view" />
      {err && <div className="error"><b>Renderer failed to start.</b><pre>{err}</pre><p>Try <a href="?renderer=webgl">WebGL2 mode</a>.</p></div>}
      {!engine && !err && <div className="loading">Initialising renderer and physics…</div>}

      <header className="topbar">
        <div className="brand">AETHER <span>orbital launch simulator</span></div>
        <div className="met">{fmtMet(metShown)}</div>
        <div className="phase">{M?.replay ? 'REPLAY · ' : ''}{PHASE_LABEL[phase] ?? phase}{lastEvent && M?.state !== 'idle' ? <em> · {lastEvent.name.replace('_', ' ')}</em> : null}</div>
        <div className="chips">
          <span className="chip" title="Graphics backend">{engine?.info.backend ?? '…'}</span>
          <span className="chip" title="Physics integrator">{engine?.info.physics ?? ''}</span>
          <span className="chip" title="Quality tier / frame rate">{engine?.info.tier} {engine ? fmt(engine.info.fps, 0) : ''} fps</span>
          <button className="chip btn" onClick={() => setHelp(!help)}>?</button>
        </div>
      </header>

      {inOrbit && <div className="banner">ORBIT ACHIEVED · {fmt(el!.periapsis_alt / 1e3, 1)} × {fmt(el!.apoapsis_alt / 1e3, 1)} km · i {fmt(el!.i_deg, 2)}°</div>}
      {M && M.view.phase === S2_DEPLETED && <div className="banner warn">S2 PROPELLANT DEPLETED · orbit not achieved</div>}
      {M?.sim.lost && <div className="banner warn">LOSS OF VEHICLE</div>}

      <section className={`telemetry ${telOpen ? 'open' : 'closed'}`}>
        <button className="tel-toggle" onClick={() => setTelOpen(!telOpen)}>{telOpen ? '▾ TELEMETRY' : '▸ TELEMETRY'}</button>
        <div className="tel-strip">
          <Item k="ALT" v={`${fmt((tel?.h ?? 0) / 1e3, 2)} km`} />
          <Item k="VEL" v={`${fmt(tel?.v_rel ?? 0)} m/s`} />
          <Item k="MACH" v={fmt(tel?.mach ?? 0, 2)} />
        </div>
        {telOpen && tel && (
          <div className="tel-grid">
            <Item k="Altitude" v={`${fmt(tel.h / 1e3, 2)} km`} />
            <Item k="Velocity (rel)" v={`${fmt(tel.v_rel)} m/s`} />
            <Item k="Velocity (inertial)" v={`${fmt(tel.v_inertial)} m/s`} />
            <Item k="Acceleration" v={`${fmt(tel.g_load * 9.80665, 1)} m/s²`} />
            <Item k="g-load" v={`${fmt(tel.g_load, 2)} g`} />
            <Item k="Mach" v={fmt(tel.mach, 2)} />
            <Item k="Dyn. pressure" v={`${fmt(tel.qbar / 1e3, 1)} kPa`} />
            <Item k="Max-Q" v={M!.maxQ.qbar > 0 ? `${fmt(M!.maxQ.qbar / 1e3, 1)} kPa @ T+${fmt(M!.maxQ.t, 0)}s` : '—'} />
            <Item k="Thrust" v={`${fmt(tel.thrust / 1e3)} kN`} />
            <Item k="Throttle" v={`${fmt(tel.throttle * 100)} %`} />
            <Item k="Engines" v={engineStatus(M!.view.phase, tel.throttle, M!.state, metShown)} />
            <Item k="Mass" v={`${fmt(tel.mass / 1e3, 1)} t`} />
            <Item k="Downrange" v={`${fmt(M!.downrange / 1e3, 1)} km`} />
            <Item k="Pitch / Hdg" v={`${fmt(att.pitch, 1)}° / ${fmt(att.heading, 1)}°`} />
            <Item k="AoA" v={`${fmt((tel.alpha * 180) / Math.PI, 2)}°`} />
            <Item k="Ambient p" v={`${fmt(tel.P / 1e3, 3)} kPa`} />
            <div className="bars">
              <Bar label="S1 PROP" f={prop1} />
              <Bar label="S2 PROP" f={prop2} />
            </div>
            {el && tel.h > 20000 && (
              <div className="orbit">
                <div className="sub">PREDICTED ORBIT (two-body)</div>
                <Item k="Apoapsis" v={isFinite(el.apoapsis_alt) ? `${fmt(el.apoapsis_alt / 1e3, 1)} km` : 'escape'} />
                <Item k="Periapsis" v={`${fmt(el.periapsis_alt / 1e3, 1)} km`} />
                <Item k="Inclination" v={`${fmt(el.i_deg, 2)}°`} />
                <Item k="Eccentricity" v={fmt(el.e, 4)} />
                <Item k="Period" v={isFinite(el.period) ? `${fmt(el.period / 60, 1)} min` : '—'} />
              </div>
            )}
          </div>
        )}
      </section>

      {eng && engine && <EngPanel engine={engine} />}

      <footer className="controls">
        {M?.state === 'idle' && !M.replay ? (
          <button className="launch" onClick={() => M.launch()}>LAUNCH</button>
        ) : (
          <div className="row">
            <button onClick={() => { if (M) M.paused = !M.paused; }}>{M?.paused ? '▶' : '❚❚'}</button>
            {WARPS.map((w) => (
              <button key={w} className={M?.warp === w ? 'on' : ''} onClick={() => { if (M) M.warp = w; }}>{w < 1 ? '¼' : w}×</button>
            ))}
            <button className={M?.replay ? 'on' : ''} onClick={() => { if (!M) return; M.replay = !M.replay; M.replayT = M.sim.history[0].t; }}>Replay</button>
            <button onClick={() => engine?.restart()}>↺</button>
          </div>
        )}
        {M?.replay && (
          <input className="scrub" type="range" min={M.sim.history[0].t} max={M.sim.t} step={0.1} value={M.replayT}
            onChange={(e) => { M.replayT = parseFloat(e.target.value); }} />
        )}
        <div className="row cams">
          {CAM_MODES.map((c) => (
            <button key={c.id} className={camNow === c.id ? 'on' : ''} onClick={() => setCamera(c.id)}>{c.label}</button>
          ))}
        </div>
        <div className="row small">
          <button className={eng ? 'on' : ''} onClick={() => { if (!engine) return; engine.engineering = !engine.engineering; setEng(engine.engineering); }}>Engineering</button>
          {(['day', 'dusk', 'night'] as TimeOfDay[]).map((t) => (
            <button key={t} className={tod === t ? 'on' : ''} onClick={() => { engine?.setTimeOfDay(t); setTod(t); }}>{t}</button>
          ))}
        </div>
      </footer>

      {help && (
        <div className="help" onClick={() => setHelp(false)}>
          <h3>Project Aether</h3>
          <p>The vehicle's position and attitude come from a 6-DOF simulation (RK4, validated against a SciPy solve_ivp reference). Drag to orbit (free camera), pinch / wheel to zoom. Keys: space = launch/pause, 1–7 cameras, E = engineering.</p>
          <p><b>Visual labels:</b> <span className="tag rom">REDUCED-ORDER MODEL</span> plume shape (isentropic + Prandtl–Meyer + shock-cell spacing from simulated ambient pressure); <span className="tag art">ARTISTIC APPROXIMATION</span> plume luminosity, smoke, sound-suppression steam, condensation, clouds, Earth surface; <span className="tag sol">SOLVER DATA</span> trajectory, telemetry, orbit elements.</p>
          <p>{engine?.cfg.disclaimer}</p>
        </div>
      )}
      <div className="labels">
        <span className="tag sol">TRAJECTORY: SOLVER DATA</span>
        <span className="tag rom">PLUME SHAPE: ROM</span>
        <span className="tag art">SMOKE/STEAM/CLOUDS: ARTISTIC</span>
      </div>
    </div>
  );
}

function Item({ k, v }: { k: string; v: string }) {
  return <div className="item"><span className="k">{k}</span><span className="v">{v}</span></div>;
}
function Bar({ label, f }: { label: string; f: number }) {
  return <div className="bar"><span>{label}</span><div className="track"><div className="fill" style={{ width: `${Math.max(0, Math.min(1, f)) * 100}%` }} /></div><em>{(f * 100).toFixed(0)}%</em></div>;
}
function engineStatus(phase: number, thr: number, state: string, met: number) {
  if (state === 'idle') return 'SAFED';
  if (state === 'countdown' && met < -3) return 'CHILLDOWN';
  if (phase === 0) return 'IGNITION';
  if (phase === 1) return `S1 ×9 RUN ${(thr * 100).toFixed(0)}%`;
  if (phase === 2 || phase === 3) return 'SHUTDOWN';
  if (phase === 4) return `S2 RUN ${(thr * 100).toFixed(0)}%`;
  return 'SHUTDOWN';
}
function attitude(e: Engine) {
  // pitch above local horizontal & heading of the body axis (render frame: x east, y up, z south)
  const v = e.mission.view;
  const q = v.y.subarray(6, 10);
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  const bx = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
  const r = v.y.subarray(0, 3);
  const rn = Math.hypot(r[0], r[1], r[2]);
  const up = [r[0] / rn, r[1] / rn, r[2] / rn];
  const s = bx[0] * up[0] + bx[1] * up[1] + bx[2] * up[2];
  const east = [-up[1], up[0], 0];
  const en = Math.hypot(east[0], east[1]);
  east[0] /= en; east[1] /= en;
  const north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
  const ce = bx[0] * east[0] + bx[1] * east[1];
  const cn = bx[0] * north[0] + bx[1] * north[1] + bx[2] * north[2];
  return { pitch: (Math.asin(Math.max(-1, Math.min(1, s))) * 180) / Math.PI, heading: ((Math.atan2(ce, cn) * 180) / Math.PI + 360) % 360 };
}
