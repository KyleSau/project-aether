// Procedural launch audio (ARTISTIC): filtered noise "rumble" + "crackle" whose level follows the
// same acoustic-power model used for camera shake (delayed by distance / speed of sound).
export class LaunchAudio {
  ctx: AudioContext | null = null;
  private rumble!: GainNode;
  private crackle!: GainNode;
  private lp!: BiquadFilterNode;
  muted = false;

  start() {
    if (this.ctx) return;
    try {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctx();
    } catch { return; }
    const ctx = this.ctx;
    const len = ctx.sampleRate * 4;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) { const w = Math.random() * 2 - 1; last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; }
    const src = ctx.createBufferSource();
    src.buffer = buf; src.loop = true;
    this.lp = ctx.createBiquadFilter();
    this.lp.type = 'lowpass'; this.lp.frequency.value = 220;
    this.rumble = ctx.createGain(); this.rumble.gain.value = 0;
    src.connect(this.lp).connect(this.rumble).connect(ctx.destination);
    src.start();
    const wbuf = ctx.createBuffer(1, len, ctx.sampleRate);
    const w = wbuf.getChannelData(0);
    for (let i = 0; i < len; i++) w[i] = Math.random() < 0.004 ? (Math.random() * 2 - 1) : (Math.random() * 2 - 1) * 0.08;
    const src2 = ctx.createBufferSource();
    src2.buffer = wbuf; src2.loop = true;
    const hp = ctx.createBiquadFilter();
    hp.type = 'bandpass'; hp.frequency.value = 900; hp.Q.value = 0.6;
    this.crackle = ctx.createGain(); this.crackle.gain.value = 0;
    src2.connect(hp).connect(this.crackle).connect(ctx.destination);
    src2.start();
  }

  /** level 0..~1.5 from the acoustic model; nearField adds crackle; vacuum -> silent */
  update(level: number, nearField: boolean, paused: boolean) {
    if (!this.ctx) return;
    const g = this.muted || paused ? 0 : Math.min(1, level) * 0.6;
    const t = this.ctx.currentTime;
    this.rumble.gain.setTargetAtTime(g, t, 0.15);
    this.crackle.gain.setTargetAtTime(nearField ? g * 0.5 : g * 0.2, t, 0.15);
    this.lp.frequency.setTargetAtTime(120 + 300 * Math.min(1, level), t, 0.3);
  }
}
