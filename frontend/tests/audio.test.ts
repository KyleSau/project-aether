import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  bandSpl, crackleWeight, iso9613Alpha, listenerImpedanceDb, molarH2oFromRh, padWeight, Ray, sourceState, type V3,
} from '../src/audio/acoustics';
import { designShelfEq, newShelfEq, shelfEqResponse } from '../src/audio/eqDesign';
import { listenerForCamera } from '../src/audio/listeners';
import { AcousticMixer, OneShotScheduler, SILENT_DB, warpPolicy, type OneShotPlan } from '../src/audio/mixModel';
import { NB } from '../src/audio/profile';
import { solveEmissionTime, TelemetryRing } from '../src/audio/telemetryRing';
import { cfg, loadReferenceTelemetry, prep, rawProfile } from './audioHelpers';

const ref = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'audio_model_ref.json'), 'utf8'));
const tel = loadReferenceTelemetry();
const LIS: Record<string, V3> = {
  pad: [310, -260, 1.7], tracker: [-1800, -2600, 2], aerial: [1400, 1700, 650], obs5km: [-3536, 3536, 1.7], obs20km: [-14142, 14142, 1.7],
};
const frameAt = (t: number) => tel.frames.reduce((a, f) => (Math.abs(f.t - t) < Math.abs(a.t - t) ? f : a));
const ground = (pos: V3, LpMax = 140) => ({ key: `g${pos.join(',')}`, mode: 'ground' as const, pos, LpMax });
// the mixer's ring holds 8192 frames = 409 s of the 0.05 s reference CSV (819 s of live 0.1 s history)
function mixerWithRing(l = ground(LIS.pad), upTo = 400) {
  const m = new AcousticMixer(prep, cfg, l);
  for (const f of tel.frames) if (f.t <= upTo) m.ring.push(f);
  return m;
}

describe('ISO 9613-1 absorption', () => {
  it('matches the standard table in the profile (20 C, 70 % RH, 101.325 kPa)', () => {
    const tb = rawProfile.propagation as unknown as { atmospheric_absorption: { table_standard_conditions: { octave_hz: number[]; alpha_dB_per_km: number[] } } };
    const T = tb.atmospheric_absorption.table_standard_conditions;
    const h = molarH2oFromRh(70, 293.15, 101325);
    T.octave_hz.forEach((f, i) => {
      const a = iso9613Alpha(f, 293.15, 101325, h) * 1000;
      expect(Math.abs(a - T.alpha_dB_per_km[i]) / T.alpha_dB_per_km[i]).toBeLessThan(0.013); // docs: within 1.3 % of the ISO table
    });
  });
  it('a horizontal ground ray integrates to alpha(z=0) x r', () => {
    const r = 2500;
    const ray = new Ray(24).set([0, 0, 1.7], [r, 0, 1.7], prep);
    const A = ray.absorption(prep.oct, new Float64Array(NB));
    const h = molarH2oFromRh(rawProfile.propagation.atmospheric_absorption.humidity_profile.surface_RH_percent, 288.15, 101325);
    for (let b = 0; b < NB; b++) expect(A[b]).toBeCloseTo(iso9613Alpha(prep.oct[b], ray.T[0], ray.P[0], h * Math.exp(-1.7 / 2500)) * r, 3);
  });
});

describe('source + propagation model vs the Python reference (octave bands)', () => {
  it('free-jet band SPL, delay, angle, crackle weight and pad term', () => {
    let maxDL = 0, maxDP = 0, maxDelay = 0, maxTh = 0, maxWc = 0, maxCrk = 0;
    for (const s of ref.samples) {
      const f = frameAt(s.t);
      const st = sourceState(prep, cfg, f)!;
      expect(st.Lw).toBeCloseTo(s.Lw, 6);
      expect(st.f_peak).toBeCloseTo(s.f_peak, 6);
      const L = new Float64Array(NB);
      const g = bandSpl(prep, st, f.noz, f.ax, LIS[s.listener], L, { ground: true, absorb: true });
      const wPad = f.phase <= 2 ? padWeight(Math.max(f.noz[2], 0), st.De) : 0;
      for (let b = 0; b < NB; b++) {
        const v = L[b] + 20 * Math.log10(Math.max(1 - 0.5 * wPad, 1e-3));
        if (s.L_jet[b] > -150) maxDL = Math.max(maxDL, Math.abs(v - s.L_jet[b]));
      }
      maxDelay = Math.max(maxDelay, Math.abs(g.delay - s.delay));
      maxTh = Math.max(maxTh, Math.abs(g.theta - s.theta));
      maxWc = Math.max(maxWc, Math.abs(crackleWeight(prep, st.w_crackle, g.theta, g.r, st.xt) - s.w_c));
      const ray = new Ray(24).set(g.src, LIS[s.listener], prep);
      const A = ray.absorption(prep.oct, new Float64Array(NB));
      for (let b = 0; b < NB; b++) if (s.A_iso[b] < 300) maxCrk = Math.max(maxCrk, Math.abs(A[b] - s.A_iso[b]) / Math.max(1, s.A_iso[b]));
      if (s.L_pad) {
        const m = new AcousticMixer(prep, cfg, ground(LIS[s.listener]));
        const Lp = new Float64Array(NB);
        (m as unknown as { padSpl: (st: unknown, ph: number, hz: number, lis: V3, o: Float64Array) => number }).padSpl(st, f.phase, Math.max(f.noz[2], 0), LIS[s.listener], Lp);
        for (let b = 0; b < NB; b++) maxDP = Math.max(maxDP, Math.abs(Lp[b] - s.L_pad[b]));
      }
    }
    expect(maxDL).toBeLessThan(0.02); // dB
    expect(maxDP).toBeLessThan(0.02);
    expect(maxDelay).toBeLessThan(1e-4); // s
    expect(maxTh).toBeLessThan(1e-6);
    expect(maxWc).toBeLessThan(1e-9);
    expect(maxCrk).toBeLessThan(1e-3);
  });

  it('spherical spreading with the near-field limit r_eff = sqrt(r^2 + x_t^2)', () => {
    const f = frameAt(0.5);
    const st = sourceState(prep, cfg, f)!;
    const L1 = new Float64Array(NB), L2 = new Float64Array(NB), L0 = new Float64Array(NB);
    const o = { ground: false, absorb: false, thetaOverride: 90 };
    bandSpl(prep, st, f.noz, f.ax, [1000, 0, 30], L1, { ...o, rOverride: 1000 });
    bandSpl(prep, st, f.noz, f.ax, [1000, 0, 30], L2, { ...o, rOverride: 4000 });
    bandSpl(prep, st, f.noz, f.ax, [1000, 0, 30], L0, { ...o, rOverride: 0 });
    const re = (r: number) => Math.sqrt(r * r + st.xt * st.xt);
    for (let b = 0; b < NB; b++) {
      expect(L1[b] - L2[b]).toBeCloseTo(20 * Math.log10(re(4000) / re(1000)), 9);
      expect(L0[b] - L1[b]).toBeCloseTo(20 * Math.log10(re(1000) / st.xt), 9); // finite at r = 0
    }
  });

  it('absorption EQ grows with distance and with frequency (5 km vs pad, same emission state)', () => {
    const f = frameAt(20);
    const st = sourceState(prep, cfg, f)!;
    const Lp = new Float64Array(NB), Lf = new Float64Array(NB), Lp0 = new Float64Array(NB), Lf0 = new Float64Array(NB);
    bandSpl(prep, st, f.noz, f.ax, LIS.pad, Lp, { ground: false, absorb: true });
    bandSpl(prep, st, f.noz, f.ax, LIS.obs5km, Lf, { ground: false, absorb: true });
    bandSpl(prep, st, f.noz, f.ax, LIS.pad, Lp0, { ground: false, absorb: false });
    bandSpl(prep, st, f.noz, f.ax, LIS.obs5km, Lf0, { ground: false, absorb: false });
    let prev = 0;
    for (let b = 0; b < NB; b++) {
      const extra = (Lp0[b] - Lp[b]) - (Lf0[b] - Lf[b]); // extra absorption at 5 km (negative)
      expect(extra).toBeLessThanOrEqual(prev + 1e-9);
      prev = extra;
    }
    expect(prev).toBeLessThan(-100); // 16 kHz is gone at 5 km
  });
});

describe('emission time and Doppler (delay line only)', () => {
  it('static source: t_e = t - r/c', () => {
    const te = solveEmissionTime(100, NaN, 340, () => 3400, -10);
    expect(te).toBeCloseTo(90, 6);
  });
  for (const v of [100, 340, 1000, 2500]) {
    it(`receding source at ${v} m/s: unique root, read rate c/(c+v)`, () => {
      const c = 340, r0 = 500;
      const rAt = (te: number) => r0 + v * Math.max(te, 0);
      let te = NaN;
      const ts: number[] = [];
      for (let t = 5; t <= 60; t += 1 / 30) { te = solveEmissionTime(t, te, c, rAt, -10); ts.push(te); expect(te + rAt(te) / c).toBeCloseTo(t, 4); }
      const rate = (ts[ts.length - 1] - ts[ts.length - 31]) / 1;
      expect(rate).toBeCloseTo(c / (c + v), 4);
    });
  }
  it('mixer reproduces the Python reception->emission map at the pad, 5 km and 20 km', () => {
    for (const name of ['pad', 'obs5km', 'obs20km']) {
      const mp = ref.mapping[name];
      const m = mixerWithRing(ground(LIS[name]));
      mp.t_r.forEach((tr: number, i: number) => {
        const o = m.compute(tr, 1);
        const want = mp.tau[i];
        if (want === null) expect(isNaN(o.dbg.te)).toBe(true);
        else expect(Math.abs(o.dbg.te - want)).toBeLessThan(0.02); // grid interpolation in Python (0.05 s samples)
      });
    }
  });
  it('Doppler rate at the pad listener is below 1 (receding) and equals 1/(1 + v_r/c)', () => {
    const m = mixerWithRing(ground(LIS.pad));
    const a = m.compute(40, 1).dbg.te;
    const b = m.compute(40 + 1 / 30, 1);
    const rate = (b.dbg.te - a) * 30;
    expect(b.jet.rate).toBeLessThan(0.95);
    expect(Math.abs(b.jet.rate - rate)).toBeLessThan(0.01);
  });
});

describe('ring buffer', () => {
  it('holds >= 60 s at the sim history rate and interpolates', () => {
    const r = new TelemetryRing(8192);
    for (const f of tel.frames) if (f.t < 300) r.push(f);
    expect(r.span).toBeGreaterThan(60);
    const out = { ...tel.frames[0], noz: [0, 0, 0] as V3, ax: [0, 0, 0] as V3 };
    const fa = frameAt(20), fb = tel.frames[tel.frames.indexOf(fa) + 1];
    expect(r.sample(0.5 * (fa.t + fb.t), out)).toBe(true);
    expect(out.h).toBeCloseTo(0.5 * (fa.h + fb.h), 6);
    const small = new TelemetryRing(100);
    for (const f of tel.frames) small.push(f);
    expect(small.size).toBe(100);
    expect(small.tLast).toBe(tel.frames[tel.frames.length - 1].t);
  });
});

describe('listeners, vacuum and onboard rules', () => {
  it('Orbital camera is a vacuum listener: silent', () => {
    const m = mixerWithRing(listenerForCamera(prep, 'orbital', [0, 0, 3e6]));
    const o = m.compute(60, 1);
    expect(o.jet.on || o.pad.on || o.crackle.on).toBe(false);
    expect(o.structDb).toBe(SILENT_DB);
    expect(m.planOneShots({ name: 'IGNITION', t: -3 })).toHaveLength(0);
  });
  it('listener rho*c -> 0 in vacuum (P < 1 Pa)', () => {
    expect(listenerImpedanceDb(120e3, prep)).toBe(-Infinity);
    expect(listenerImpedanceDb(0, prep)).toBeCloseTo(10 * Math.log10((1.225 * 340.294) / 413.3), 3);
  });
  it('Onboard: plume heard at liftoff, external killed above Mach ~1.2 / in vacuum, structure continues while burning', () => {
    const m = mixerWithRing(listenerForCamera(prep, 'onboard', [0, 0, 0]));
    const lo = m.compute(2, 1);
    expect(lo.jet.on).toBe(true);
    expect(lo.structDb).toBeGreaterThan(-40);
    const hi = m.compute(140, 1); // M ~ 8, ~65 km
    expect(hi.jet.on).toBe(false);
    expect(hi.dbg.wair).toBe(0);
    expect(hi.structDb).toBeGreaterThan(-40);
    const s2 = m.compute(300, 1); // stage 2 burning in near vacuum
    expect(s2.jet.on).toBe(false);
    expect(s2.structDb).toBeGreaterThan(-60);
    const coast = m.compute(153, 1); // MECO -> separation coast: silent structure
    expect(coast.structDb).toBe(SILENT_DB);
  });
  it('Free camera far up in vacuum hears nothing external', () => {
    const m = mixerWithRing(listenerForCamera(prep, 'free', [100000, 0, 150000]));
    const o = m.compute(300, 1);
    expect(o.jet.on).toBe(false);
  });
  it('profile presets drive the camera listeners', () => {
    const p = listenerForCamera(prep, 'pad', [0, 0, 0]);
    expect(p.mode).toBe('ground');
    expect(p.pos).toEqual([310, -260, 1.7]);
    expect(p.LpMax).toBe(150);
    expect(listenerForCamera(prep, 'aerial', [0, 0, 0]).pos[2]).toBe(650);
    const e = listenerForCamera(prep, 'engine', [0, 0, 0]);
    expect(e.mode).toBe('chase');
    expect(e.chaseR).toBe(30);
    expect(e.chaseTheta).toBe(100);
  });
});

describe('event delays and one-shot scheduling', () => {
  it('ignition arrives late by the pad->listener travel time (matches Python)', () => {
    for (const name of ['pad', 'obs5km', 'obs20km']) {
      const m = mixerWithRing(ground(LIS[name]), -2.9);
      const [p] = m.planOneShots({ name: 'IGNITION', t: -3 });
      expect(Math.abs(p.tArrive - ref.mapping[name].ignition_arrival)).toBeLessThan(2e-3);
    }
  });
  it('MECO tail is delayed by the vehicle->listener travel time (matches Python)', () => {
    const tMeco = tel.events.find((e) => e.name === 'MECO')!.t;
    for (const name of ['pad', 'obs5km']) {
      const m = mixerWithRing(ground(LIS[name]), tMeco);
      const [p] = m.planOneShots({ name: 'MECO', t: tMeco });
      // Python takes the delay two 0.05 s grid samples before MECO, the runtime 0.1 s before; the vehicle recedes
      // at ~7 c there, so the 441 s (pad) delay differs by ~0.3 s
      expect(Math.abs(p.tArrive - ref.mapping[name].meco_arrival)).toBeLessThan(0.5);
    }
  });
  it('onboard: MECO structural + stage-sep clunk at event time, no delay', () => {
    const tMeco = tel.events.find((e) => e.name === 'MECO')!.t;
    const m = mixerWithRing(listenerForCamera(prep, 'onboard', [0, 0, 0]), tMeco + 5);
    const pm = m.planOneShots({ name: 'MECO', t: tMeco });
    expect(pm.some((p) => p.key.includes(':struct') && p.tArrive === tMeco)).toBe(true);
    const ps = m.planOneShots({ name: 'STAGE_SEP', t: tMeco + 3 });
    expect(ps[0].asset).toBe('stage_sep_structural');
    // ground listeners never hear the structural clunk
    expect(mixerWithRing(ground(LIS.pad)).planOneShots({ name: 'STAGE_SEP', t: tMeco + 3 })).toHaveLength(0);
  });
  it('scheduler: fires once within the lookahead, late starts with an offset, respects the warp policy', () => {
    const pl: OneShotPlan = { key: 'k', asset: 'ignition', tArrive: 10, eqDb: new Float64Array(NB) };
    const dur = () => 6;
    const s = new OneShotScheduler();
    expect(s.update(9.5, [pl], warpPolicy(1), 0.12, dur, () => false)).toHaveLength(0);
    const f = s.update(9.9, [pl], warpPolicy(1), 0.12, dur, () => false);
    expect(f).toHaveLength(1);
    expect(f[0].delay).toBeCloseTo(0.1, 9);
    s.reset(); // seen one control tick late: start from the beginning, now
    const one = s.update(10.03, [pl], warpPolicy(1), 0.12, dur, () => false);
    expect(one[0].offset).toBe(0);
    expect(one[0].delay).toBe(0);
    expect(s.update(10.0, [pl], warpPolicy(1), 0.12, dur, () => false)).toHaveLength(0); // once
    s.reset(); // seek into the middle of the one-shot
    const late = s.update(12.5, [pl], warpPolicy(1), 0.12, dur, () => false);
    expect(late[0].offset).toBeCloseTo(2.5, 9);
    s.reset();
    expect(s.update(20, [pl], warpPolicy(1), 0.12, dur, () => false)).toHaveLength(0); // over
    s.reset();
    expect(s.update(9.9, [pl], warpPolicy(10), 1.2, dur, () => false)).toHaveLength(0); // > 4x: none
    s.reset();
    expect(s.update(9.9, [pl], warpPolicy(2), 0.24, dur, () => true)).toHaveLength(0); // 2x: no overlap
    s.reset();
    expect(s.update(9.9, [pl], warpPolicy(2), 0.24, dur, () => false)).toHaveLength(1);
  });
});

describe('time-warp policy', () => {
  it('normal up to 1x, -3 dB per doubling to 4x, time-lapse ambience beyond', () => {
    for (const w of [0.25, 0.5, 1]) {
      const p = warpPolicy(w);
      expect(p.airborneDb).toBe(0);
      expect(p.oneShots).toBe('all');
      expect(p.doppler).toBe(true);
    }
    expect(warpPolicy(2).airborneDb).toBeCloseTo(-3, 9);
    expect(warpPolicy(4).airborneDb).toBeCloseTo(-6, 9);
    expect(warpPolicy(2).oneShots).toBe('no-overlap');
    for (const w of [5, 10, 50]) {
      const p = warpPolicy(w);
      expect(p.airborneDb).toBe(-24);
      expect(p.fadeTau * 3).toBeCloseTo(1, 6); // reaches -24 dB in about 1 s
      expect(p.mask).toEqual({ rumble_lf: 1, jet_mix: 0, crackle: 0, pad_interaction: 0, onboard_structure: 1, aero_buffet: 0 });
      expect(p.oneShots).toBe('none');
      expect(p.doppler).toBe(false);
    }
  });
  it('warp never changes the playback rate beyond the Doppler ratio (no pitch-shift chaos)', () => {
    const m = mixerWithRing(ground(LIS.pad));
    expect(m.compute(60, 50).jet.rate).toBe(1);
    const r1 = m.compute(60, 1).jet.rate;
    const r2 = m.compute(60, 2).jet.rate;
    expect(r2).toBeCloseTo(r1, 9);
    expect(r1).toBeGreaterThanOrEqual(0.5);
    expect(r1).toBeLessThanOrEqual(1.5);
  });
});

describe('determinism', () => {
  it('a seek (fresh solver state) gives the same mix as continuous playback', () => {
    const a = mixerWithRing(ground(LIS.obs5km));
    let last = a.compute(0, 1);
    for (let t = 0; t <= 90; t += 1 / 30) last = a.compute(t, 1);
    const eqA = Float64Array.from(last.jet.eqDb), teA = last.dbg.te;
    const b = mixerWithRing(ground(LIS.obs5km));
    const o = b.compute(last.t, 1);
    expect(o.dbg.te).toBeCloseTo(teA, 4);
    for (let i = 0; i < NB; i++) expect(o.jet.eqDb[i]).toBeCloseTo(eqA[i], 2);
  });
});

describe('shelf-cascade EQ', () => {
  it('hits the octave targets of real mixes in the audible bands', () => {
    const eq = newShelfEq(prep.oct);
    const errs: number[] = [];
    for (const [name, t] of [['pad', 5], ['pad', 60], ['obs5km', 40], ['tracker', 30], ['aerial', 10]] as const) {
      const m = mixerWithRing(ground(LIS[name]));
      const o = m.compute(t, 1);
      for (const [g, A] of [[o.jet, prep.A_jet], [o.pad, prep.A_pad]] as const) {
        if (!g.on) continue;
        designShelfEq(g.eqDb, prep.oct, 48000, eq, 5, A);
        const out = Array.from(g.eqDb, (v, b) => v + A[b]);
        const peak = Math.max(...out);
        for (let b = 0; b < NB; b++) {
          const r = shelfEqResponse(eq, prep.oct[b], 48000);
          if (out[b] > peak - 40) errs.push(Math.abs(r - g.eqDb[b])); // clearly audible bands
          else expect(r + A[b]).toBeLessThan(peak - 30); // inaudible bands stay inaudible
        }
      }
    }
    errs.sort((a, b) => a - b);
    expect(errs[errs.length - 1]).toBeLessThan(2.0);
    expect(errs[Math.floor(errs.length * 0.9)]).toBeLessThan(0.5);
  });
  it('a flat target is exact', () => {
    const eq = newShelfEq(prep.oct);
    designShelfEq(new Float64Array(NB).fill(-12), prep.oct, 48000, eq);
    for (const f of prep.oct) expect(shelfEqResponse(eq, f, 48000)).toBeCloseTo(-12, 2);
    expect(eq.lpHz).toBe(0);
  });
});
