// Copy the canonical data files (aether/data) into public/data for the app.
import { cpSync, mkdirSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', '..', 'data');
const dst = join(here, '..', 'public', 'data');
mkdirSync(dst, { recursive: true });
for (const f of readdirSync(src)) {
  if (/\.(json|txt)$/.test(f)) cpSync(join(src, f), join(dst, f));
}
const wasmPkg = join(here, '..', '..', 'wasm', 'pkg');
if (existsSync(join(wasmPkg, 'aether_wasm_bg.wasm'))) {
  mkdirSync(join(here, '..', 'src', 'wasm'), { recursive: true });
  for (const f of ['aether_wasm.js', 'aether_wasm.d.ts', 'aether_wasm_bg.wasm', 'aether_wasm_bg.wasm.d.ts']) {
    if (existsSync(join(wasmPkg, f))) cpSync(join(wasmPkg, f), join(here, '..', 'src', 'wasm', f));
  }
}
console.log('data synced ->', dst);
