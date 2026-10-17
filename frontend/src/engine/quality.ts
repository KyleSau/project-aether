// Device-tier detection and automatic quality scaling.
export type Tier = 'low' | 'medium' | 'high' | 'ultra';

export interface QualitySettings {
  tier: Tier;
  pixelRatio: number;
  maxParticles: number;
  shadowMapSize: number;
  shadows: boolean;
  engineLightShadows: boolean;
  bloom: boolean;
  msaa: number;
  skySamples: number;
  cloudCount: number;
  towerDetail: number;
  motionBlur: boolean;
}

export const TIERS: Record<Tier, Omit<QualitySettings, 'tier' | 'pixelRatio'> & { maxPR: number }> = {
  low: { maxPR: 1.0, maxParticles: 6000, shadowMapSize: 1024, shadows: true, engineLightShadows: false, bloom: true, msaa: 0, skySamples: 6, cloudCount: 24, towerDetail: 0, motionBlur: false },
  medium: { maxPR: 1.5, maxParticles: 16000, shadowMapSize: 2048, shadows: true, engineLightShadows: false, bloom: true, msaa: 0, skySamples: 10, cloudCount: 48, towerDetail: 1, motionBlur: false },
  high: { maxPR: 2.0, maxParticles: 48000, shadowMapSize: 4096, shadows: true, engineLightShadows: true, bloom: true, msaa: 4, skySamples: 16, cloudCount: 90, towerDetail: 2, motionBlur: true },
  ultra: { maxPR: 2.0, maxParticles: 120000, shadowMapSize: 4096, shadows: true, engineLightShadows: true, bloom: true, msaa: 4, skySamples: 24, cloudCount: 140, towerDetail: 2, motionBlur: true },
};

export function isMobile(): boolean {
  const ua = navigator.userAgent || '';
  const touch = 'ontouchstart' in window || navigator.maxTouchPoints > 1;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (touch && Math.min(screen.width, screen.height) < 820);
}

export function detectTier(isWebGPU: boolean): Tier {
  const q = new URLSearchParams(location.search).get('quality');
  if (q === 'low' || q === 'medium' || q === 'high' || q === 'ultra') return q;
  if (isMobile()) return 'low';
  const cores = navigator.hardwareConcurrency || 4;
  const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory || 8;
  if (isWebGPU && cores >= 8 && mem >= 8) return 'high';
  return cores >= 6 ? 'medium' : 'low';
}

export function makeSettings(tier: Tier): QualitySettings {
  const t = TIERS[tier];
  return { tier, pixelRatio: Math.min(window.devicePixelRatio || 1, t.maxPR), ...t };
}

/** Frame-time based resolution scaler (adjusts only pixel ratio, never physics). */
export class AutoScaler {
  private acc = 0;
  private n = 0;
  scale = 1;
  constructor(public enabled = true, public minScale = 0.5) {}
  /** returns true if scale changed */
  private warm = 0;
  sample(dtMs: number): boolean {
    if (!this.enabled) return false;
    // ignore start-up (shader compilation) and isolated stalls
    this.warm += dtMs;
    if (this.warm < 6000 || dtMs > 250) return false;
    this.acc += dtMs;
    this.n++;
    if (this.acc < 2000) return false;
    const avg = this.acc / this.n;
    this.acc = 0;
    this.n = 0;
    const old = this.scale;
    if (avg > 40) this.scale = Math.max(this.minScale, this.scale * 0.8);
    else if (avg < 14 && this.scale < 1) this.scale = Math.min(1, this.scale * 1.15);
    return old !== this.scale;
  }
}
