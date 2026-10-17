// Vehicle configuration (mirror of data/vehicle.json). All values are labelled assumptions.
export interface EngineCfg {
  count: number;
  propellants: string;
  mdot0_kg_s: number;
  ve_m_s: number;
  pe_pa: number;
  ae_m2: number;
  throttle_min: number;
  gimbal_max_deg: number;
  x_gimbal: number;
  startup_s: number;
}

export interface VehicleConfig {
  name: string;
  disclaimer: string;
  site: {
    latitude_deg: number;
    longitude_deg: number;
    pad_altitude_m: number;
    launch_azimuth_deg: number;
    rotating_earth: boolean;
    j2: boolean;
    aero?: boolean;
  };
  geometry: {
    radius_m: number;
    reference_area_m2: number;
    length_m: number;
    stage1: { x0: number; x1: number; tank_x0: number; tank_x1: number };
    stage2: { x0: number; x1: number; tank_x0: number; tank_x1: number };
    payload: { x0: number; x1: number; radius_m: number };
    fairing: { x0: number; x1: number; radius_m: number };
  };
  mass: {
    stage1_dry_kg: number;
    stage1_prop_kg: number;
    stage2_dry_kg: number;
    stage2_prop_kg: number;
    payload_kg: number;
    fairing_kg: number;
  };
  engines: { stage1: EngineCfg; stage2: EngineCfg };
  aero: { ca_mach: number[]; ca_value: number[]; cn_alpha_per_rad: number; x_cp_m: number; _source?: string };
  wind: { alt_m: number[]; speed_m_s: number[] };
  guidance: {
    t_vertical_s: number;
    kick_duration_s: number;
    kick_angle_deg: number;
    blend_s: number;
    q_limit_pa: number;
    q_band_pa: number;
    g_limit: number;
    attitude_wn: number;
    attitude_zeta: number;
    rcs_alpha_max: number;
    sep_delay_s: number;
    ses_delay_s: number;
    fairing_alt_m: number;
    target_alt_m: number;
    s2_wn: number;
    s2_zeta: number;
    s2_sin_min: number;
    s2_sin_max: number;
  };
}
