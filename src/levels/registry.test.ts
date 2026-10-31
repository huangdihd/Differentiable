// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { compileGraph, createDefaultRegistry, type GraphJSON } from '../engine/graph';
import {
  applyResult,
  evaluateLevel,
  initialProgress,
  type ForwardLevel,
  type LevelResult,
} from '../engine/levels';
import { availableComponents, buildPlayerRegistry } from './registry';
import { tensor } from '../engine/tensor';
import { referenceRegistry } from './level-helpers';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3Levels } from './chapter3';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'out', component: 'output', attrs: { name: 'y' } },
  ],
  edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'out', port: 'x' } }],
};
const level: ForwardLevel = {
  id: '0-1',
  titleKey: 'test.title',
  descriptionKey: 'test.description',
  kind: 'forward',
  available: ['add', 'mul'],
  unlock: [],
  unlockComponents: ['identity'],
  interface: { inputs: { x: [2] }, output: 'y', outputShape: [2] },
  fixed: [],
  reference: (inputs) => inputs.x,
  referenceGraph: graph,
};
const result: LevelResult = {
  levelId: level.id,
  passed: true,
  tests: [],
  metrics: { topNodes: 2, expandedPrimitives: 0, parameterCount: 0 },
};
describe('player registry', () => {
  it('clones passing solutions, never drafts or failed results', () => {
    const initial = initialProgress([level.id]);
    const next = applyResult(initial, level, result, graph);
    expect(next.passedSolutions[level.id]).toEqual(graph);
    expect(next.passedSolutions[level.id]).not.toBe(graph);
    expect(applyResult(next, level, { ...result, passed: false }, graph)).toBe(next);
    expect(applyResult(next, level, result).passedSolutions).toBe(next.passedSolutions);
    expect(initial.passedSolutions).toEqual({});
    const registry = buildPlayerRegistry([level], next);
    const definition = registry.get('identity')!;
    expect(definition.inputs).toEqual(['x']);
    expect(definition.outputs).toEqual(['out']);
    expect((definition.subgraph as GraphJSON).nodes[1].attrs?.name).toBe('out');
    expect(graph.nodes[1].attrs?.name).toBe('y');
    expect(buildPlayerRegistry([level], initial).get('identity')).toBeUndefined();
  });
  it('combines unlocked and preset components then removes bans and own products', () => {
    const progress = initialProgress([level.id], ['identity', 'exp', 'add']);
    expect(
      availableComponents({ ...level, presets: ['log'], forbidden: ['mul', 'log'] }, progress),
    ).toEqual(['add', 'exp']);
    expect(
      buildPlayerRegistry([{ ...level, presets: ['identity'] }], progress).get('identity'),
    ).toBeDefined();
  });
  it('executes a newly passed lower implementation through an unchanged nested solution', async () => {
    const lower: ForwardLevel = {
      ...level,
      id: 'fixture-lower',
      available: ['add', 'const'],
      unlockComponents: ['fixture_lower'],
      fixed: [{ id: 'identity', inputs: { x: [2, -3] } }],
      atol: 0.01,
      rtol: 0,
    };
    const upper: ForwardLevel = {
      ...lower,
      id: 'fixture-upper',
      available: ['fixture_lower'],
      unlockComponents: ['fixture_upper'],
    };
    const call = (component: string): GraphJSON => ({
      version: 1,
      nodes: [...graph.nodes, { id: 'call', component }],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'call', port: 'x' } },
        { from: { node: 'call', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    });
    const replacement: GraphJSON = {
      version: 1,
      nodes: [
        ...graph.nodes,
        { id: 'offset', component: 'const', attrs: { value: 0.001 } },
        { id: 'add', component: 'add' },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'add', port: 'a' } },
        { from: { node: 'offset', port: 'out' }, to: { node: 'add', port: 'b' } },
        { from: { node: 'add', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    const levels = [lower, upper];
    let progress = initialProgress(levels.map((entry) => entry.id));
    const first = await evaluateLevel(lower, graph);
    expect(first.passed).toBe(true);
    expect(first.tests).toHaveLength(1);
    progress = applyResult(progress, lower, first, graph);
    const nested = call('fixture_lower');
    const upperResult = await evaluateLevel(upper, nested, {
      registry: buildPlayerRegistry(levels, progress),
    });
    expect(upperResult.passed).toBe(true);
    progress = applyResult(progress, upper, upperResult, nested);
    const before = buildPlayerRegistry(levels, progress);
    const savedUpper = progress.passedSolutions[upper.id];
    const second = await evaluateLevel(lower, replacement);
    expect(second.passed).toBe(true);
    expect(second.tests).toHaveLength(1);
    progress = applyResult(progress, lower, second, replacement);
    expect(progress.passedSolutions[upper.id]).toBe(savedUpper);
    const after = buildPlayerRegistry(levels, progress);
    const execute = async (registry: ReturnType<typeof buildPlayerRegistry>) => {
      const plan = compileGraph(call('fixture_upper'), {
        registry,
        inputShapes: lower.interface.inputs,
      });
      const x = tensor([2, -3], [2]);
      try {
        const outputs = plan.execute({ x });
        try {
          return Array.from(await outputs.y.read());
        } finally {
          plan.releaseExecution(outputs, [x]);
        }
      } finally {
        plan.dispose();
        x.dispose();
      }
    };
    expect(await execute(before)).toEqual([2, -3]);
    const actual = await execute(after);
    expect(actual[0]).toBeCloseTo(2.001, 5);
    expect(actual[1]).toBeCloseTo(-2.999, 5);
    expect(actual).not.toEqual([2, -3]);
    // Rebuilding must not mutate an earlier registry snapshot.
    expect(await execute(before)).toEqual([2, -3]);
  });
  it('rejects recursive persisted player components during compilation', () => {
    const call = (component: string): GraphJSON => ({
      version: 1,
      nodes: [...graph.nodes, { id: 'call', component }],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'call', port: 'x' } },
        { from: { node: 'call', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    });
    const a = { ...level, id: 'fixture-a', unlockComponents: ['fixture_a'] };
    const b = { ...level, id: 'fixture-b', unlockComponents: ['fixture_b'] };
    // Corrupted persisted data can contain direct or mutual recursion despite evaluation guards.
    for (const target of ['fixture_a', 'fixture_b']) {
      const progress = {
        ...initialProgress([a.id, b.id], ['fixture_a', 'fixture_b']),
        passed: [a.id, b.id],
        passedSolutions: { [a.id]: call(target), [b.id]: call('fixture_a') },
      };
      const registry = buildPlayerRegistry([a, b], progress);
      expect(() =>
        compileGraph(call('fixture_a'), { registry, inputShapes: level.interface.inputs }),
      ).toThrow('recursive_component');
    }
  });
  it('keeps native gather intact when a lesson also unlocks a saved embedding', async () => {
    const producer = { ...level, unlockComponents: ['embedding', 'gather'] };
    const initial = initialProgress([producer.id]);
    expect(availableComponents(producer, initial)).not.toContain('gather');
    const progress = applyResult(initial, producer, result, graph);
    const consumer = { ...level, unlockComponents: [] };
    expect(availableComponents(consumer, initial)).not.toContain('gather');
    expect(availableComponents(consumer, progress)).toContain('gather');
    expect(availableComponents(producer, progress)).not.toContain('gather');
    const native = createDefaultRegistry().get('gather')!;
    const player = buildPlayerRegistry([producer], progress);
    const authoring = referenceRegistry([producer]);
    for (const registry of [player, authoring]) {
      const gather = registry.get('gather')!;
      expect(gather.inputs).toEqual(native.inputs);
      expect(gather.outputs).toEqual(native.outputs);
      expect(gather.primitive).toBeDefined();
      expect(gather.subgraph).toBeUndefined();
      const embedding = registry.get('embedding')!;
      expect(embedding.inputs).toEqual(['x']);
      expect(embedding.subgraph).not.toBe(graph);
      expect((embedding.subgraph as GraphJSON).nodes[1].attrs?.name).toBe('out');
      const plan = compileGraph(
        {
          version: 1,
          nodes: [
            { id: 'table', component: 'input', attrs: { name: 'table' } },
            { id: 'idx', component: 'input', attrs: { name: 'idx' } },
            { id: 'gather', component: 'gather' },
            { id: 'out', component: 'output', attrs: { name: 'y' } },
          ],
          edges: [
            { from: { node: 'table', port: 'out' }, to: { node: 'gather', port: 'table' } },
            { from: { node: 'idx', port: 'out' }, to: { node: 'gather', port: 'idx' } },
            { from: { node: 'gather', port: 'out' }, to: { node: 'out', port: 'x' } },
          ],
        },
        { registry, inputShapes: { table: [3, 2], idx: [2] }, preferNative: false },
      );
      const table = tensor([1, 2, 3, 4, 5, 6], [3, 2]);
      const idx = tensor([2, 0], [2]);
      try {
        const outputs = plan.execute({ table, idx });
        try {
          expect(Array.from(await outputs.y.read())).toEqual([5, 6, 1, 2]);
        } finally {
          plan.releaseExecution(outputs, [table, idx]);
        }
      } finally {
        plan.dispose();
        table.dispose();
        idx.dispose();
      }
    }
    expect(graph.nodes[1].attrs?.name).toBe('y');
  });
  it('does not expose gather anywhere in chapters 0–3', () => {
    const preceding = [...chapter0Levels, ...chapter1Levels, ...chapter2Levels, ...chapter3Levels];
    const progress = initialProgress(preceding.map((entry) => entry.id));
    for (const entry of preceding) {
      expect(availableComponents(entry, progress), entry.id).not.toContain('gather');
      expect(entry.unlockComponents ?? [], entry.id).not.toContain('gather');
      progress.unlockedComponents.push(...(entry.unlockComponents ?? []));
    }
  });
  it('retains native kernels while replacing their inspectable subgraph', () => {
    const native = createDefaultRegistry()
      .list()
      .find((definition) => definition.native)!;
    expect(native).toBeDefined();
    const producer = { ...level, unlockComponents: [native.name] };
    const progress = applyResult(initialProgress([level.id]), producer, result, graph);
    const definition = buildPlayerRegistry([producer], progress).get(native.name)!;
    expect(definition.native).toBe(native.native);
    expect(definition.subgraph).not.toBe(native.subgraph);
    expect((definition.subgraph as GraphJSON).nodes).toHaveLength(2);
  });
});
