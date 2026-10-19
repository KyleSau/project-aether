// Asset loading: Ogg/Opus first (file_web), WAV fallback (file) when the browser cannot decode Opus
// (Safari < 17). Decoded buffers are trimmed to the manifest's exact sample count (Opus decoders add
// pre-skip/padding), and the two LF/structure loops are kept at half rate to save memory on phones.
import type { AssetMeta, PreparedProfile } from './profile';

export type FetchBytes = (path: string) => Promise<ArrayBuffer>;

/** loops with no content near Nyquist, stored at fs/2 to save memory: rumble_lf (<= 160 Hz) and
 *  onboard_structure (<= 1.2 kHz; its 8 kHz octave is at -107 dBFS). aero_buffet is NOT: despite its nominal
 *  4 Hz - 2 kHz band it carries real energy up to 16 kHz (octave levels -38/-40 dBFS at 8/16 kHz). */
export const HALF_RATE = new Set(['rumble_lf', 'onboard_structure']);

export interface LoadedAssets {
  buffers: Record<string, AudioBuffer>;
  format: Record<string, 'opus' | 'wav'>;
  bytes: number; // network bytes
  decodedBytes: number; // float32 PCM held
}

export function opusLikelySupported(): boolean {
  try {
    if (typeof document === 'undefined') return false;
    const a = document.createElement('audio');
    return a.canPlayType('audio/ogg; codecs="opus"') !== '';
  } catch { return false; }
}

/** copy into a buffer of exactly n frames (circular fill when the decode is short, for loops) */
function fitLength(ctx: BaseAudioContext, src: AudioBuffer, n: number, loop: boolean, half: boolean): AudioBuffer {
  const x = src.getChannelData(0);
  if (!half) {
    if (src.length === n) return src;
    const out = ctx.createBuffer(1, n, src.sampleRate);
    const y = out.getChannelData(0);
    for (let i = 0; i < n; i++) y[i] = i < x.length ? x[i] : loop ? x[i % x.length] : 0;
    return out;
  }
  // decimate by 2 with a [1 2 1]/4 kernel (circular for loops): -0.07 dB at 2 kHz, content above is absent
  const m = Math.floor(n / 2);
  const out = ctx.createBuffer(1, m, src.sampleRate / 2);
  const y = out.getChannelData(0);
  const at = (i: number) => (i >= 0 && i < n ? (i < x.length ? x[i] : loop ? x[i % x.length] : 0) : loop ? x[((i % n) + n) % n % x.length] : 0);
  for (let j = 0; j < m; j++) { const i = 2 * j; y[j] = 0.25 * at(i - 1) + 0.5 * at(i) + 0.25 * at(i + 1); }
  return out;
}

export async function loadAssets(ctx: BaseAudioContext, prep: PreparedProfile, fetchBytes: FetchBytes, base: string,
  opts: { preferOpus: boolean; halfRate?: boolean }): Promise<LoadedAssets> {
  const res: LoadedAssets = { buffers: {}, format: {}, bytes: 0, decodedBytes: 0 };
  const all: AssetMeta[] = [...prep.raw.layers, ...prep.raw.oneshots];
  const sr0 = prep.raw.sample_rate_hz;
  await Promise.all(all.map(async (a) => {
    const n = Math.round((a.samples * ctx.sampleRate) / sr0);
    let buf: AudioBuffer | null = null;
    if (opts.preferOpus) {
      try {
        const ab = await fetchBytes(base + a.file_web);
        res.bytes += ab.byteLength;
        buf = await ctx.decodeAudioData(ab);
        res.format[a.name] = 'opus';
      } catch { buf = null; }
    }
    if (!buf) {
      const ab = await fetchBytes(base + a.file);
      res.bytes += ab.byteLength;
      buf = await ctx.decodeAudioData(ab);
      res.format[a.name] = 'wav';
    }
    const half = (opts.halfRate ?? true) && HALF_RATE.has(a.name) && a.loop;
    const fit = fitLength(ctx, buf, n, a.loop, half);
    res.buffers[a.name] = fit;
    res.decodedBytes += fit.length * 4 * fit.numberOfChannels;
  }));
  return res;
}
