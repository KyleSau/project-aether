// Verify the static site: allowed file types only, total size, no absolute-URL fetches.
import { readdirSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, extname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'site');
const allowed = new Set(['.html', '.js', '.mjs', '.wasm', '.json', '.txt', '.css', '.png', '.jpg', '.webp', '.svg', '.glb', '.ogg', '.wav']);
let total = 0;
let wavFallback = 0; // WAV copies of the audio assets: fetched only by browsers that cannot decode Opus
const files = [];
const walk = (d) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    const s = statSync(p);
    if (s.isDirectory()) walk(p);
    else {
      const ext = extname(f).toLowerCase();
      if (!allowed.has(ext)) throw new Error(`disallowed file type in site/: ${relative(root, p)}`);
      if (ext === '.wav') wavFallback += s.size; else total += s.size;
      files.push([relative(root, p).split(String.fromCharCode(92)).join('/'), s.size]);
    }
  }
};
walk(root);
for (const [f] of files) {
  if (!f.endsWith('.js') && !f.endsWith('.html')) continue;
  const src = readFileSync(join(root, f), 'utf8');
  const urls = src.match(/https?:\/\/[^\s"'`)]+/g) || [];
  const bad = urls.filter((u) => /fetch|import/.test(u));
  if (bad.length) console.warn('check URLs in', f, bad.slice(0, 5));
}
files.sort((a, b) => b[1] - a[1]);
const manifest = files.map(([f, s]) => `${(s / 1024).toFixed(1).padStart(9)} KB  ${f}`).join('\n');
writeFileSync(join(root, 'MANIFEST.txt'), `Project Aether static site\nTotal: ${(total / 1048576).toFixed(2)} MB + ${(wavFallback / 1048576).toFixed(2)} MB WAV audio fallback in ${files.length} files\n\n${manifest}\n`);
console.log(`site OK: ${files.length} files, ${(total / 1048576).toFixed(2)} MB (+ ${(wavFallback / 1048576).toFixed(2)} MB WAV audio fallback)`);
if (total > 15 * 1048576) throw new Error('site exceeds 15 MB');
if (wavFallback > 20 * 1048576) throw new Error('WAV audio fallback exceeds 20 MB');
// GitHub Pages: serve files as-is (no Jekyll processing)
writeFileSync(join(root, '.nojekyll'), '');
