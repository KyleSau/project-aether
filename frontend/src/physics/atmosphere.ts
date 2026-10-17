// U.S. Standard Atmosphere 1976 — mirror of physics/aether_physics/atmosphere.py
const G0 = 9.80665;
const R_AIR = 8314.32 / 28.9644;
const R0 = 6356766.0;
export const GAMMA_AIR = 1.4;

const LAYERS: [number, number, number][] = [
  [0.0, 288.15, -0.0065],
  [11000.0, 216.65, 0.0],
  [20000.0, 216.65, 0.001],
  [32000.0, 228.65, 0.0028],
  [47000.0, 270.65, 0.0],
  [51000.0, 270.65, -0.0028],
  [71000.0, 214.65, -0.002],
  [84852.0, 186.946, 0.0],
];

const PB: number[] = (() => {
  const p = [101325.0];
  for (let i = 0; i < LAYERS.length - 1; i++) {
    const [hb, tb, lb] = LAYERS[i];
    const h1 = LAYERS[i + 1][0];
    if (lb === 0) p.push(p[i] * Math.exp((-G0 * (h1 - hb)) / (R_AIR * tb)));
    else {
      const t1 = tb + lb * (h1 - hb);
      p.push(p[i] * Math.pow(t1 / tb, -G0 / (R_AIR * lb)));
    }
  }
  return p;
})();

// geometric altitude [m], T [K], P [Pa], rho [kg/m^3]
const UPPER: [number, number, number, number][] = [
  [86000.0, 186.87, 3.7338e-1, 6.958e-6],
  [90000.0, 186.87, 1.8359e-1, 3.416e-6],
  [100000.0, 195.08, 3.2011e-2, 5.604e-7],
  [110000.0, 240.0, 7.1042e-3, 9.708e-8],
  [120000.0, 360.0, 2.5382e-3, 2.222e-8],
  [150000.0, 634.39, 4.5422e-4, 2.076e-9],
  [200000.0, 854.56, 8.4736e-5, 2.541e-10],
  [300000.0, 976.01, 8.7704e-6, 1.916e-11],
  [400000.0, 995.83, 1.4518e-6, 2.803e-12],
  [500000.0, 999.24, 3.0236e-7, 5.215e-13],
  [600000.0, 999.85, 8.213e-8, 1.137e-13],
  [800000.0, 999.99, 1.7036e-8, 1.136e-14],
  [1000000.0, 1000.0, 7.5138e-9, 3.561e-15],
];

export interface AtmoOut {
  T: number;
  P: number;
  rho: number;
  a: number;
}

export function atmosphere(z: number, out: AtmoOut = { T: 0, P: 0, rho: 0, a: 0 }): AtmoOut {
  if (z < 0) z = 0;
  if (z <= 86000.0) {
    const h = (R0 * z) / (R0 + z);
    let i = LAYERS.length - 1;
    while (i > 0 && h < LAYERS[i][0]) i--;
    const [hb, tb, lb] = LAYERS[i];
    const pb = PB[i];
    let t: number, p: number;
    if (lb === 0) {
      t = tb;
      p = pb * Math.exp((-G0 * (h - hb)) / (R_AIR * tb));
    } else {
      t = tb + lb * (h - hb);
      p = pb * Math.pow(t / tb, -G0 / (R_AIR * lb));
    }
    out.T = t;
    out.P = p;
    out.rho = p / (R_AIR * t);
    out.a = Math.sqrt(GAMMA_AIR * R_AIR * t);
    return out;
  }
  let k: number;
  if (z >= UPPER[UPPER.length - 1][0]) k = UPPER.length - 2;
  else {
    k = 0;
    while (UPPER[k + 1][0] < z) k++;
  }
  const [z0, t0, p0, r0] = UPPER[k];
  const [z1, t1, p1, r1] = UPPER[k + 1];
  const f = (z - z0) / (z1 - z0);
  out.T = t0 + f * (t1 - t0);
  out.P = Math.exp(Math.log(p0) + f * (Math.log(p1) - Math.log(p0)));
  out.rho = Math.exp(Math.log(r0) + f * (Math.log(r1) - Math.log(r0)));
  out.a = Math.sqrt(GAMMA_AIR * R_AIR * out.T);
  return out;
}
