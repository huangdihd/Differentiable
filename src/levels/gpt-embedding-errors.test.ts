// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { compileGraph, safeCompileGraph, type GraphJSON } from '../engine/graph';
import { BackendError, WebGpuBackend } from '../engine/webgpu';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { createInitialProgress } from '../storage';
import { gameLevels } from './catalog';
import { chapter6TrainLevels } from './chapter6';
import { buildPlayerRegistry } from './registry';

function unlockBefore(id: string) {
  let p = createInitialProgress();
  for (const l of gameLevels) {
    if (l.id === id) break;
    p = {
      ...p,
      passed: [...new Set([...p.passed, l.id])],
      unlocked: [...new Set([...p.unlocked, ...l.unlock, l.id])],
      unlockedComponents: [...new Set([...p.unlockedComponents, ...(l.unlockComponents ?? [])])],
    };
  }
  return p;
}

const registry = buildPlayerRegistry(gameLevels, unlockBefore('6-4'));
const level = chapter6TrainLevels[0];
function graph(maxLength: number): GraphJSON {
  return {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      {
        id: 'emb',
        component: 'embed_pos',
        attrs: {
          automatic_parameters: true,
          num_embeddings: 27,
          max_length: maxLength,
          embedding_dim: 24,
        },
      },
      { id: 'out', component: 'output', attrs: { name: 'probe' } },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'emb', port: 'idx' } },
      { from: { node: 'emb', port: 'out' }, to: { node: 'out', port: 'x' } },
    ],
  };
}
const compile = (g: GraphJSON) =>
  compileGraph(g, { registry, inputShapes: level.interface.inputs, random: new Random(1) });

it('7-4: E+P configured 27/16/24 embeds the level x [8,16]', async () => {
  const batch = level.dataset(new Random(level.seed));
  const plan = compile(graph(16));
  const r = plan.execute({ x: batch.inputs.x });
  expect(r.probe.shape).toEqual([8, 16, 24]);
});

it('7-4: a too-small 最大序列长度 is a shape error on the E+P node, not a runtime lookup', () => {
  expect(
    safeCompileGraph(graph(8), { registry, inputShapes: level.interface.inputs }),
  ).toMatchObject({
    ok: false,
    errors: [
      { key: 'embed_pos_length', node: 'emb', port: 'idx', params: { length: 16, max_length: 8 } },
    ],
  });
});

it('7-4: an out-of-range character id is reported on E+P (CPU step error and GPU read error)', () => {
  const g = graph(16);
  const plan = compile(g);
  const ids = Array.from({ length: 128 }, (_, i) => (i === 5 ? 30 : i % 27));
  let caught: unknown;
  try {
    plan.execute({ x: T.tensor(ids, [8, 16]) });
  } catch (error) {
    caught = error;
  }
  const expected = {
    key: 'embed_token_range',
    node: 'emb',
    port: 'idx',
    params: { rows: 27, last: 26 },
  };
  expect(plan.diagnose(caught, g)).toMatchObject(expected);
  // WebGPU raises a node-less BackendError at the next read.
  expect(plan.diagnose(new BackendError('engine.gather.index'), g)).toMatchObject(expected);
});

it('WebGPU clears its out-of-range flag after reporting it, so a corrected graph reads again', () => {
  const writes: number[][] = [];
  const fake = {
    createBuffer: () => ({ destroy() {} }),
    lost: new Promise(() => {}),
    addEventListener() {},
    queue: { writeBuffer: (_b: unknown, _o: number, d: Uint32Array) => writes.push([...d]) },
  };
  const Ctor = WebGpuBackend as unknown as new (device: unknown, info: unknown) => WebGpuBackend;
  const backend = new Ctor(fake, {});
  const fault = (backend as unknown as { gatherFault(): BackendError }).gatherFault();
  expect(fault.key).toBe('engine.gather.index');
  expect(writes).toEqual([[0]]);
});
