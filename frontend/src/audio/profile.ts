// Typed view of data/audio/acoustic_profile.json (schema "aether-acoustic-profile/1") plus the derived
// octave-band tables the runtime needs. Everything numeric comes from the profile; nothing is re-tuned here.

export interface AssetMeta {
  name: string;
  file: string; // WAV, relative to data/audio/
  file_web: string; // Ogg/Opus, relative to data/audio/
  loop: boolean;
  samples: number; // exact length at sample_rate_hz
  seconds: number;
  rms_dbfs: number;
  octave_hz: number[];
  octave_db: number[];
  group?: string;
  sync_group?: string;
  event?: string;
  emit_at?: string;
  /** one-shots: seconds after the event at which the file starts (ignition_overpressure) */
  offset_s?: number;
  /** ignition_overpressure: RMS of the main 0.5 s (the level rule's reference) */
  rms_main_dbfs?: number;
}

export interface ListenerPreset {
  mode: 'ground' | 'chase' | 'onboard' | 'cabin' | 'vacuum';
  pos_enu?: [number, number, number];
  Lp_max_db?: number;
  notes?: string;
}

/** Only the fields the runtime reads are typed; the file carries more (provenance, notes). */
export interface AcousticProfile {
  schema: string;
  sample_rate_hz: number;
  source: {
    acoustic_efficiency: { value: number };
    source_continuum_cutoff: { pa_lo: number; pa_hi: number };
    spectrum: { strouhal: number[]; level_db: number[] };
  };
  directivity: { angles_deg: number[]; strouhal: number[]; di_db: Record<string, number[]> };
  propagation: {
    geometric_spreading: { rho0c0: number };
    atmospheric_absorption: {
      humidity_profile: { surface_RH_percent: number; h2o_scale_height_m: number };
      crackle_band_policy: { evaluate_at_hz: number; exponent: number; nonlinear_range_m: number };
    };
    ground_reflection: { h_listener_m: number };
  };
  crackle: {
    reference_weight: number;
    max_weight: number;
    nonlinear_growth: { max_gain: number; log2_saturation: number };
    directivity: { peak_deg: number; width_deg: number };
  };
  pad_interaction: { level_db_re_jet: number; ignition_overpressure?: { level_db_re_full_thrust_jet: number } };
  onboard: {
    structure_borne: { thrust_ref_stage1_N: number };
    digital_reference_levels_dbfs: Record<'onboard' | 'cabin', { structure_at_stage1_full_thrust: number; aero_at_w_aero_1: number; stage_sep: number }>;
    cabin_transmission_loss: { m_s_kg_m2: number };
  };
  level_mapping: {
    target_peak_dbfs: number; k_default: number; k_game: number;
    limiter?: { ceiling_dbfs: number; lookahead_ms: number; release_ms: number };
  };
  anti_repetition: { voice_offset_fraction: number; crossfade_cycle_s: number };
  listeners: { aether: Record<string, ListenerPreset> };
  layers: AssetMeta[];
  oneshots: AssetMeta[];
}

export const NB = 11; // octave bands 16 Hz .. 16 kHz

export interface PreparedProfile {
  raw: AcousticProfile;
  oct: number[]; // band centres (profile octave_hz)
  octLo: number[];
  octHi: number[];
  /** jet group asset band level: rumble_lf + jet_mix power sum (the crackle layer shares the jet EQ) */
  A_jet: number[];
  A_pad: number[];
  /** pad asset spectral shape normalised to 0 dB overall */
  padShape: number[];
  /** overall (power-summed) octave level of an asset */
  oa: Record<string, number>;
  assets: Record<string, AssetMeta>;
  dirSt: number[];
  dirAng: number[];
  dirTab: number[][]; // [nSt][nAngle]
  specLogSt: number[];
  specDb: number[];
}

export const dbsum = (xs: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += Math.pow(10, xs[i] / 10);
  return 10 * Math.log10(s + 1e-30);
};

export function prepareProfile(raw: AcousticProfile): PreparedProfile {
  if (raw.schema !== 'aether-acoustic-profile/1') throw new Error(`unsupported acoustic profile schema ${raw.schema}`);
  const assets: Record<string, AssetMeta> = {};
  for (const a of [...raw.layers, ...raw.oneshots]) assets[a.name] = a;
  const oct = assets.jet_mix.octave_hz.slice();
  if (oct.length !== NB) throw new Error('profile octave grid must have 11 bands');
  const octLo = oct.map((f) => f / Math.SQRT2);
  const octHi = oct.map((f) => f * Math.SQRT2);
  const A_jet = oct.map((_, b) => 10 * Math.log10(Math.pow(10, assets.rumble_lf.octave_db[b] / 10) + Math.pow(10, assets.jet_mix.octave_db[b] / 10)));
  const A_pad = assets.pad_interaction.octave_db.slice();
  const oaPad = dbsum(A_pad);
  const padShape = A_pad.map((v) => v - oaPad);
  const oa: Record<string, number> = {};
  for (const a of Object.values(assets)) oa[a.name] = dbsum(a.octave_db);
  const d = raw.directivity;
  const dirTab = d.strouhal.map((k) => {
    const row = d.di_db[String(k)] ?? d.di_db[k.toString()];
    if (!row) throw new Error(`directivity row ${k} missing`);
    return row;
  });
  return {
    raw, oct, octLo, octHi, A_jet, A_pad, padShape, oa, assets,
    dirSt: d.strouhal, dirAng: d.angles_deg, dirTab,
    specLogSt: raw.source.spectrum.strouhal.map(Math.log), specDb: raw.source.spectrum.level_db,
  };
}
