import { useMemo } from 'react';
import type { Engine } from '../engine/Engine';
import { extra } from '../boot';
import { RE, OMEGA_E } from '../physics/dynamics';

function downrangeKm(y: ArrayLike<number>, t: number, t0: number, p0: ArrayLike<number>, rot: boolean) {
  const th = (rot ? OMEGA_E : 0) * (t - t0);
  const c = Math.cos(th), s = Math.sin(th);
  const x = c * y[0] + s * y[1], yy = -s * y[0] + c * y[1], z = y[2];
  const dot = (x * p0[0] + yy * p0[1] + z * p0[2]) / (Math.hypot(x, yy, z) * Math.hypot(p0[0], p0[1], p0[2]));
  return (RE * Math.acos(Math.min(1, Math.max(-1, dot)))) / 1000;
}

/** Altitude vs downrange: live 6-DOF history (solid) over the Python solve_ivp reference (dashed). */
export function TrajPlot({ engine }: { engine: Engine }) {
  const M = engine.mission;
  const H = M.sim.history;
  const t0 = H[0].t, p0 = H[0].y, rot = engine.cfg.site.rotating_earth;
  const ref = useMemo(() => {
    const r = extra.reference as { samples: { t: number; h: number; y: number[]; phase: number }[] } | null;
    if (!r) return [] as [number, number][];
    return r.samples.filter((s) => s.phase <= 5).map((s) => [downrangeKm(s.y, s.t, t0, p0, rot), s.h / 1000] as [number, number]);
  }, [t0, p0, rot]);
  const live: [number, number][] = [];
  const step = Math.max(1, Math.floor(H.length / 300));
  for (let i = 0; i < H.length; i += step) {
    if (H[i].t > M.view.t) break;
    live.push([downrangeKm(H[i].y, H[i].t, t0, p0, rot), (Math.hypot(H[i].y[0], H[i].y[1], H[i].y[2]) - RE) / 1000]);
  }
  const W = 270, Hh = 110;
  const xmax = Math.max(400, ...ref.map((p) => p[0]).slice(-1), live.length ? live[live.length - 1][0] : 0);
  const ymax = 260;
  const X = (v: number) => (v / xmax) * (W - 24) + 22;
  const Y = (v: number) => Hh - 12 - (v / ymax) * (Hh - 18);
  const path = (pts: [number, number][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(Math.max(0, p[1])).toFixed(1)}`).join('');
  return (
    <svg width={W} height={Hh} className="traj">
      <rect x="22" y="6" width={W - 24} height={Hh - 18} fill="rgba(0,0,0,0.25)" />
      {[0, 100, 200].map((a) => <g key={a}><line x1="22" x2={W} y1={Y(a)} y2={Y(a)} stroke="rgba(120,170,220,0.15)" /><text x="0" y={Y(a) + 3} fontSize="8" fill="#8ea2b8">{a}</text></g>)}
      <text x={W - 70} y={Hh - 2} fontSize="8" fill="#8ea2b8">downrange {xmax.toFixed(0)} km</text>
      {ref.length > 0 && <path d={path(ref)} fill="none" stroke="#ffb24d" strokeDasharray="3 3" strokeWidth="1" />}
      <path d={path(live)} fill="none" stroke="#46d2ff" strokeWidth="1.6" />
      <text x="26" y="14" fontSize="8" fill="#46d2ff">live 6-DOF</text>
      <text x="80" y="14" fontSize="8" fill="#ffb24d">reference (solve_ivp)</text>
    </svg>
  );
}
