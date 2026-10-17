// Reduced-order exhaust-plume model (REDUCED-ORDER MODEL, not CFD).
// Isentropic exit/jet relations + Prandtl-Meyer initial jet-boundary turning angle +
// Pack's shock-cell spacing. Mirrors physics/aether_physics/plume_rom.py, which is the
// validated reference and also writes data/plume_table.json.
import type { EngineCfg } from '../physics/config';

export interface PlumeState {
  pe: number; pa: number; ratio: number; Me: number; Mj: number;
  thetaB: number; // initial jet-boundary angle vs. axis [rad] (lip angle + PM turning)
  dj: number; // fully-expanded jet diameter [m]
  cellLength: number; // shock-cell spacing [m]
  diamonds: number; // 0..1 visibility of shock-diamond structure
  machDisk: boolean;
  machDiskX: number; // distance of first Mach disk / strong reflection from exit [m]
  regime: 'over-expanded' | 'near-ideal' | 'under-expanded' | 'vacuum';
}

export function prandtlMeyer(M: number, g: number) {
  if (M <= 1) return 0;
  const a = Math.sqrt((g + 1) / (g - 1));
  return a * Math.atan(Math.sqrt(((g - 1) / (g + 1)) * (M * M - 1))) - Math.atan(Math.sqrt(M * M - 1));
}
export function areaRatio(M: number, g: number) {
  return (1 / M) * Math.pow((2 / (g + 1)) * (1 + ((g - 1) / 2) * M * M), (g + 1) / (2 * (g - 1)));
}
export function machFromPressureRatio(p0_over_p: number, g: number) {
  return Math.sqrt((2 / (g - 1)) * (Math.pow(p0_over_p, (g - 1) / g) - 1));
}

export function plumeState(e: EngineCfg, throttle: number, pa: number): PlumeState {
  const g = e.gamma_exit ?? 1.22;
  const pc = (e.chamber_pressure_pa ?? 9.7e6) * Math.max(throttle, 0.05);
  const pe = e.pe_pa * Math.max(throttle, 0.05);
  const Me = machFromPressureRatio(pc / pe, g);
  const paEff = Math.max(pa, 1e-3);
  const ratio = pe / paEff;
  const Mj = machFromPressureRatio(Math.min(pc / paEff, 1e12), g);
  const lip = ((e.nozzle_lip_angle_deg ?? 10) * Math.PI) / 180;
  const turn = prandtlMeyer(Mj, g) - prandtlMeyer(Me, g);
  const thetaB = Math.max(-0.35, Math.min(1.45, lip + turn));
  const de = 2 * Math.sqrt(e.ae_m2 / Math.PI);
  const dj = de * Math.sqrt(Math.min(areaRatio(Mj, g) / areaRatio(Me, g), 1e4));
  const cellLength = 1.306 * dj * Math.sqrt(Math.max(Mj * Mj - 1, 0));
  // diamonds are visible when the shock train is strong enough and the ambient gas dense enough
  const strength = Math.min(Math.abs(Math.log(ratio)) / Math.log(2.5), 1);
  const density = Math.min(pa / 25000, 1);
  const diamonds = pa < 500 ? 0 : Math.max(0.15, strength) * density;
  const machDisk = pa > 500 && (ratio < 0.45 || ratio > 2.2);
  const dThroat = de / Math.sqrt(areaRatio(Me, g));
  const machDiskX = ratio > 1 ? 0.67 * dThroat * Math.sqrt(pc / paEff) : 0.6 * cellLength;
  const regime = pa < 50 ? 'vacuum' : ratio < 0.9 ? 'over-expanded' : ratio > 1.1 ? 'under-expanded' : 'near-ideal';
  return { pe, pa, ratio, Me, Mj, thetaB, dj, cellLength, diamonds, machDisk, machDiskX, regime };
}
