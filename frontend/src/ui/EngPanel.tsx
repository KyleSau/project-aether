import { useMemo } from 'react';
import type { Engine } from '../engine/Engine';
import { atmosphere } from '../physics/atmosphere';
import { extra } from '../boot';

const f = (v: number, d = 1) => (isFinite(v) ? v.toFixed(d) : '—');

interface Thermo {
  propellant: string; method: string; cantera_version: string;
  stages: Record<string, { pc_bar: number; of_ratio: number; eps: number; Tc_K: number; gamma_c: number; mw_c: number; cstar_m_s: number; Isp_vac_s: number; Isp_sl_s: number; Te_K: number; pe_bar: number; Me: number; gamma_e: number; mode: string }>;
  reference_check?: { quantity: string; computed: number; reference: number; tol_pct: number; pass: boolean; source: string }[];
}

export function EngPanel({ engine }: { engine: Engine }) {
  const M = engine.mission;
  const tel = M.tel;
  const ps = M.view.phase <= 2 ? engine.fx.ps1 : engine.fx.ps2;
  const profile = useMemo(() => {
    const pts: { h: number; T: number; P: number; rho: number }[] = [];
    for (let h = 0; h <= 120000; h += 2000) { const a = atmosphere(h); pts.push({ h, T: a.T, P: a.P, rho: a.rho }); }
    return pts;
  }, []);
  const W = 220, H = 150;
  const hx = (h: number) => H - (h / 120000) * H;
  const tx = (T: number) => ((T - 160) / (380 - 160)) * W;
  const px = (P: number) => ((Math.log10(Math.max(P, 1e-3)) + 3) / 8.1) * W;
  const thermo = extra.thermo as Thermo | null;
  const normalForce = Math.hypot(tel.fa[1], tel.fa[2]);
  const bendProxy = tel.qbar * tel.alpha; // q-alpha [Pa rad]
  return (
    <aside className="engpanel">
      <h4>ENGINEERING <span className="tag sol">SOLVER DATA</span></h4>
      <div className="kv"><span>Phase</span><b>{M.view.phase}</b><span>x_cg</span><b>{f(tel.x_cg, 2)} m</b></div>
      <div className="kv"><span>Iyy</span><b>{(tel.iyy / 1e6).toFixed(1)}e6 kg·m²</b><span>Gimbal</span><b>{f((Math.hypot(tel.gimbal_y, tel.gimbal_z) * 180) / Math.PI, 2)}°</b></div>
      <div className="kv"><span>Axial aero</span><b>{f(-tel.fa[0] / 1e3, 1)} kN</b><span>Normal aero</span><b>{f(normalForce / 1e3, 1)} kN</b></div>
      <div className="kv"><span>q·α</span><b>{f(bendProxy / 1e3, 2)} kPa·rad</b><span>T_amb</span><b>{f(tel.T, 1)} K</b></div>
      <div className="sub">Atmosphere (USSA-1976): T (orange) · log P (blue) · vehicle (dot)</div>
      <svg width={W} height={H} className="chart">
        <polyline fill="none" stroke="#ffa64d" strokeWidth="1.5" points={profile.map((p) => `${tx(p.T)},${hx(p.h)}`).join(' ')} />
        <polyline fill="none" stroke="#5ab8ff" strokeWidth="1.5" points={profile.map((p) => `${px(p.P)},${hx(p.h)}`).join(' ')} />
        {tel.h < 120000 && <circle cx={tx(tel.T)} cy={hx(tel.h)} r="4" fill="#fff" />}
        {[0, 40, 80, 120].map((k) => <text key={k} x="2" y={hx(k * 1000) - 2} fontSize="9" fill="#9ab">{k} km</text>)}
      </svg>
      <h4>PLUME <span className="tag rom">REDUCED-ORDER MODEL</span></h4>
      {ps ? (
        <>
          <div className="kv"><span>Regime</span><b>{ps.regime}</b><span>pe/pa</span><b>{ps.ratio > 1e4 ? '∞' : f(ps.ratio, 2)}</b></div>
          <div className="kv"><span>M exit</span><b>{f(ps.Me, 2)}</b><span>M jet</span><b>{ps.Mj > 50 ? '>50' : f(ps.Mj, 2)}</b></div>
          <div className="kv"><span>Boundary angle</span><b>{f((ps.thetaB * 180) / Math.PI, 1)}°</b><span>Jet Ø</span><b>{ps.dj > 1e3 ? '>1 km' : f(ps.dj, 2) + ' m'}</b></div>
          <div className="kv"><span>Shock cell</span><b>{ps.cellLength > 1e3 ? '—' : f(ps.cellLength, 2) + ' m'}</b><span>Diamonds</span><b>{ps.diamonds > 0.05 ? `${f(ps.diamonds * 100, 0)}%` : 'none'}</b></div>
          <div className="kv"><span>Mach disk</span><b>{ps.machDisk ? `${f(ps.machDiskX, 1)} m` : 'no'}</b><span>p_amb</span><b>{f(ps.pa / 1e3, 3)} kPa</b></div>
        </>
      ) : <div className="muted">engines not running</div>}
      <h4>EXHAUST THERMOCHEMISTRY <span className="tag sol">CANTERA</span></h4>
      {thermo ? (
        <div className="thermo">
          <div className="muted">{thermo.propellant} · {thermo.method} · Cantera {thermo.cantera_version}</div>
          {Object.entries(thermo.stages).map(([k, s]) => (
            <div key={k} className="kv wide"><span>{k}</span><b>Tc {f(s.Tc_K, 0)} K · γc {f(s.gamma_c, 3)} · MW {f(s.mw_c, 2)} · c* {f(s.cstar_m_s, 0)} m/s · Isp vac {f(s.Isp_vac_s, 1)} s · pe {f(s.pe_bar, 3)} bar · Me {f(s.Me, 2)}</b></div>
          ))}
          {thermo.reference_check?.map((r) => (
            <div key={r.quantity} className="kv wide"><span>{r.pass ? '✓' : '✗'} {r.quantity}</span><b>{f(r.computed, 1)} vs {f(r.reference, 1)} ({r.source}, ±{r.tol_pct}%)</b></div>
          ))}
        </div>
      ) : <div className="muted">thermo.json not present (run backend/precompute.py)</div>}
      <h4>THERMAL <span className="tag rom">ESTIMATE</span></h4>
      {extra.thermal ? (() => { const t = extra.thermal as { model: string; peak_heat_flux_kW_m2: number; peak_t: number; peak_h_m: number; total_heat_load_MJ_m2: number };
        return <div className="muted">{t.model}: peak {f(t.peak_heat_flux_kW_m2, 1)} kW/m² at T+{f(t.peak_t, 0)} s ({f(t.peak_h_m / 1000, 1)} km), load {f(t.total_heat_load_MJ_m2, 2)} MJ/m² (reference trajectory)</div>; })() : <div className="muted">thermal.json not present</div>}
      <h4>PHYSICS BENCHMARK</h4>
      {extra.benchmarks ? (() => { const b = extra.benchmarks as { ts_full: { stepsPerSec: number }; wasm_full: { stepsPerSec: number }; accuracy: { ts_final_pos_err_vs_python_m: number } };
        return <div className="muted">TS {(b.ts_full.stepsPerSec / 1e3).toFixed(0)}k steps/s · WASM {(b.wasm_full.stepsPerSec / 1e3).toFixed(0)}k steps/s · |Δr| vs Python {f(b.accuracy.ts_final_pos_err_vs_python_m, 2)} m (Node/V8)</div>; })() : null}
      <h4>EVENTS</h4>
      <div className="events">{M.sim.events.map((e, i) => <div key={i}>T{e.t >= 0 ? '+' : ''}{f(e.t, 2)} s · {e.name}</div>)}</div>
      <h4>CFD <span className="tag na">NOT CONNECTED</span></h4>
      <div className="muted">OpenFOAM adapter + axisymmetric rhoCentralFoam case generator ready (backend/openfoam). No OpenFOAM installation is available, so no CFD has been run and no solver field data is displayed. Needs: WSL2 Ubuntu + OpenFOAM (or Docker image) with user approval.</div>
    </aside>
  );
}
