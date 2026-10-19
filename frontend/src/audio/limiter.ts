// Look-ahead peak limiter (acoustic_profile.json level_mapping.limiter): the real-time form of
// tools/audio/render_launch.py lookahead_limit(). Gain reduction is computed on the input and applied to the
// input delayed by L = lookahead samples, so the output equals the offline limiter shifted by L samples.
//   1. GRreq[n] = max(0, 20 log10|x[n]| - ceiling)          (dB)
//   2. hold   = max(GRreq over the last L+1 inputs)          (= look-ahead window of the delayed sample)
//   3. GRr    = max(hold, GRr * exp(-1 / (release * fs)))    (exponential release of the dB reduction)
//   4. GR     = mean(GRr over the last L samples), >= GRreq of the delayed sample (linear attack ramp)
//   5. y      = x[n-L] * 10^(-GR/20), clipped to the ceiling
// It runs as an AudioWorklet. The processor source is kept here as a string so it can be loaded from a Blob URL
// (browsers) or a temporary file (node-web-audio-api in the offline validation render), with no bundler step.

export interface LimiterParams { ceilingDb: number; lookaheadMs: number; releaseMs: number }

export const LIMITER_PROCESSOR = 'aether-lookahead-limiter';

export const LIMITER_SOURCE = `
class AetherLookaheadLimiter extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const p = (options && options.processorOptions) || {};
    const L = Math.max(1, Math.round(((p.lookaheadMs ?? 5) * sampleRate) / 1000));
    this.L = L;
    this.ceil = p.ceilingDb ?? -1;
    this.lin = Math.pow(10, this.ceil / 20);
    this.dec = Math.exp(-1 / (((p.releaseMs ?? 150) * sampleRate) / 1000));
    this.xs = new Float32Array(L + 1);      // input ring (delay line)
    this.rq = new Float64Array(L + 1);      // GRreq ring
    this.dqI = new Float64Array(L + 2);     // monotonic deque for the sliding max: sample index
    this.dqV = new Float64Array(L + 2);     //   and value
    this.h = 0; this.t = 0;                 // deque head / tail (count)
    this.av = new Float64Array(L);          // last L release values (moving average)
    this.avI = 0; this.avS = 0;
    this.grr = 0;
    this.k = 0;                             // input sample counter
    this.gr = 0;                            // last gain reduction (dB), for metering
  }
  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0][0];
    const L = this.L, cap = L + 2;
    for (let i = 0; i < out.length; i++) {
      const x = inp ? inp[i] : 0;
      const a = Math.abs(x);
      const req = a > this.lin ? 20 * Math.log10(a) - this.ceil : 0;
      const k = this.k;
      const slot = k % (L + 1);
      this.xs[slot] = x;
      this.rq[slot] = req;
      // sliding max over the last L+1 requirements
      while (this.t > this.h && this.dqV[(this.t - 1) % cap] <= req) this.t--;
      this.dqI[this.t % cap] = k; this.dqV[this.t % cap] = req; this.t++;
      while (this.dqI[this.h % cap] <= k - (L + 1)) this.h++;
      const hold = this.dqV[this.h % cap];
      this.grr = Math.max(hold, this.grr * this.dec);
      this.avS += this.grr - this.av[this.avI];
      this.av[this.avI] = this.grr;
      this.avI = (this.avI + 1) % L;
      // delayed sample: x[k - L]
      const dslot = (k - L + (L + 1) * 4) % (L + 1);
      const xDel = k >= L ? this.xs[dslot] : 0;
      const reqDel = k >= L ? this.rq[dslot] : 0;
      const gr = Math.max(this.avS / L, reqDel);
      this.gr = gr;
      let y = xDel * Math.pow(10, -gr / 20);
      if (y > this.lin) y = this.lin; else if (y < -this.lin) y = -this.lin;
      out[i] = y;
      this.k = k + 1;
    }
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    return true;
  }
}
registerProcessor('${'aether-lookahead-limiter'}', AetherLookaheadLimiter);
`;

/** Pure-TS reference of the same algorithm (unit tests; also documents the exact recipe) */
export function lookaheadLimitOffline(x: Float32Array | Float64Array, fs: number, p: LimiterParams): Float64Array {
  const L = Math.max(1, Math.round((p.lookaheadMs * fs) / 1000));
  const lin = Math.pow(10, p.ceilingDb / 20), dec = Math.exp(-1 / ((p.releaseMs * fs) / 1000));
  const n = x.length;
  const req = new Float64Array(n);
  for (let i = 0; i < n; i++) { const a = Math.abs(x[i]); req[i] = a > lin ? 20 * Math.log10(a) - p.ceilingDb : 0; }
  const y = new Float64Array(n);
  const av = new Float64Array(L);
  let avS = 0, avI = 0, grr = 0;
  for (let i = 0; i < n; i++) {
    let hold = 0;
    for (let j = i; j <= Math.min(n - 1, i + L); j++) if (req[j] > hold) hold = req[j];
    grr = Math.max(hold, grr * dec);
    avS += grr - av[avI]; av[avI] = grr; avI = (avI + 1) % L;
    const gr = Math.max(avS / L, req[i]);
    y[i] = Math.max(-lin, Math.min(lin, x[i] * Math.pow(10, -gr / 20)));
  }
  return y;
}

const loaded = new WeakMap<BaseAudioContext, Promise<boolean>>();
const ready = new WeakMap<BaseAudioContext, boolean>();

/** true once loadLimiterWorklet() has registered the processor on this context */
export const limiterAvailable = (ctx: BaseAudioContext) => ready.get(ctx) === true;

/** register the processor on a context; resolves false when AudioWorklet is unavailable (fallback path) */
export function loadLimiterWorklet(ctx: BaseAudioContext, moduleUrl?: string): Promise<boolean> {
  const hit = loaded.get(ctx);
  if (hit) return hit;
  const p = (async () => {
    const aw = (ctx as BaseAudioContext & { audioWorklet?: AudioWorklet }).audioWorklet;
    if (!aw) return false;
    let url = moduleUrl;
    let revoke = false;
    if (!url) {
      if (typeof Blob === 'undefined' || typeof URL === 'undefined' || !URL.createObjectURL) return false;
      url = URL.createObjectURL(new Blob([LIMITER_SOURCE], { type: 'application/javascript' }));
      revoke = true;
    }
    try { await aw.addModule(url); ready.set(ctx, true); return true; } catch { return false; } finally { if (revoke) URL.revokeObjectURL(url); }
  })();
  loaded.set(ctx, p);
  return p;
}
