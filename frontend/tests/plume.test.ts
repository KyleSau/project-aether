import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { plumeState } from '../src/engine/plumeModel';
import { atmosphere } from '../src/physics/atmosphere';
import type { VehicleConfig } from '../src/physics/config';

const DATA = join(__dirname, '..', '..', 'data');
const cfg: VehicleConfig = JSON.parse(readFileSync(join(DATA, 'vehicle.json'), 'utf8'));
const table = JSON.parse(readFileSync(join(DATA, 'plume_table.json'), 'utf8'));

describe('TS plume ROM matches the Python reduced-order model table', () => {
  for (const st of ['stage1', 'stage2'] as const) {
    it(st, () => {
      let maxRel = 0;
      for (const row of table.stages[st]) {
        const ps = plumeState(cfg.engines[st], 1, atmosphere(row.h).P);
        for (const k of ['Me', 'thetaB', 'dj', 'cellLength', 'diamonds', 'machDiskX'] as const) {
          const a = ps[k], b = row[k];
          const rel = Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
          if (Math.abs(b) > 1e-9 && isFinite(b)) maxRel = Math.max(maxRel, rel);
        }
        expect(ps.regime).toBe(row.regime);
        expect(ps.machDisk).toBe(row.machDisk);
      }
      expect(maxRel).toBeLessThan(1e-6);
    });
  }
});
