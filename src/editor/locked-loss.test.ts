// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import type { TrainLevel } from '../engine/levels';
import { evaluateLevel } from '../engine/levels';
import { trainDependencyError, TRAIN_DEPENDENCY_ERROR } from '../engine/train-dependency';
import { gameLevels } from '../levels/catalog';
import { referenceRegistry } from '../levels/level-helpers';
import { createEditorState, editorReducer, initialGraph } from './state';
import { ensureLockedLoss, isLockedEdge } from './locked-loss';

const trainLevels = gameLevels.filter((l): l is TrainLevel => l.kind === 'train');
const delay = trainLevels.find((l) => l.id === 'time-rnn-delay')!;
const registry = referenceRegistry(gameLevels as never);
const lossNode = (g: GraphJSON) => g.nodes.find((n) => n.ui?.locked)!;

describe('locked loss', () => {
  it('is applied exactly to the MSE training levels', () => {
    expect(trainLevels.filter((l) => l.lockedLoss).map((l) => l.id)).toEqual([
      '3-4',
      'time-rnn-delay',
      'time-forecast',
    ]);
  });

  it('seeds new drafts with the loss wired to t and to the loss output', () => {
    const state = createEditorState(delay);
    const loss = lossNode(state.graph);
    expect(loss.component).toBe('mse');
    const ends = state.graph.edges.filter((e) => e.from.node === loss.id || e.to.node === loss.id);
    expect(ends).toHaveLength(2);
    expect(ends.every((e) => isLockedEdge(state.graph, e))).toBe(true);
  });

  it('cannot be deleted, reconfigured, unwired or rewired; only p can be connected', () => {
    let state = createEditorState(delay);
    const loss = lossNode(state.graph);
    const locked = state.graph.edges.filter(
      (e) => e.to.node === loss.id || e.from.node === loss.id,
    );
    const input = state.graph.nodes.find((n) => n.attrs?.name === 'x')!;
    const output = state.graph.nodes.find((n) => n.component === 'output')!;
    const r = registry;
    expect(
      editorReducer(state, { type: 'removeNodes', ids: [loss.id] }, r).graph.nodes,
    ).toContainEqual(loss);
    expect(editorReducer(state, { type: 'setAttrs', id: loss.id, attrs: { a: 1 } }, r).error).toBe(
      'locked_node',
    );
    for (const edge of locked) {
      expect(editorReducer(state, { type: 'disconnect', edge }, r).error).toBe('locked_node');
      expect(
        editorReducer(
          state,
          { type: 'rewire', edge, from: { node: input.id, port: 'out' }, to: edge.to },
          r,
        ).error,
      ).toBe('locked_node');
    }
    const x = { node: input.id, port: 'out' };
    expect(
      editorReducer(state, { type: 'connect', from: x, to: { node: loss.id, port: 't' } }, r).error,
    ).toBe('locked_node');
    expect(
      editorReducer(state, { type: 'connect', from: x, to: { node: output.id, port: 'x' } }, r)
        .error,
    ).toBe('locked_node');
    state = editorReducer(state, { type: 'connect', from: x, to: { node: loss.id, port: 'p' } }, r);
    expect(state.error).toBeNull();
    // Copies of the locked loss are ordinary nodes.
    state = editorReducer(
      editorReducer(state, { type: 'select', ids: [loss.id] }, r),
      { type: 'copy' },
      r,
    );
    state = editorReducer(state, { type: 'paste' }, r);
    expect(state.graph.nodes.filter((n) => n.ui?.locked)).toHaveLength(1);
  });

  it('migrates saved drafts: adopts a matching loss, keeps conflicting player nodes beside it', () => {
    const reference = structuredClone(delay.referenceGraph!);
    const adopted = ensureLockedLoss(delay, reference);
    expect(adopted.nodes).toHaveLength(reference.nodes.length);
    expect(lossNode(adopted).id).toBe('loss');
    expect(ensureLockedLoss(delay, adopted)).toEqual(adopted);
    expect(trainDependencyError(delay, adopted, registry)).toBeNull();

    // A draft whose output is fed by something else keeps that node and gains the locked loss.
    const draft = initialGraph(delay);
    draft.nodes.push({ id: 'mine', component: 'abs', ui: { x: 300, y: 100 } });
    draft.edges.push({ from: { node: 'input_1', port: 'out' }, to: { node: 'mine', port: 'x' } });
    draft.edges.push({ from: { node: 'mine', port: 'out' }, to: { node: 'output', port: 'x' } });
    const migrated = ensureLockedLoss(delay, draft);
    expect(migrated.nodes.map((n) => n.id)).toEqual(
      expect.arrayContaining(['mine', 'locked_loss']),
    );
    expect(migrated.edges.find((e) => e.to.node === 'output')?.from.node).toBe('locked_loss');
    expect(migrated.edges).toContainEqual(draft.edges[0]);
    // Levels without a lock are untouched.
    const forward = gameLevels.find((l) => l.kind === 'forward')!;
    const g = initialGraph(forward as never);
    expect(ensureLockedLoss(forward as never, g)).toBe(g);
  });
});

describe('training anti-cheat', () => {
  it('accepts every training reference', () => {
    for (const level of trainLevels)
      expect([level.id, trainDependencyError(level, level.referenceGraph!, registry)]).toEqual([
        level.id,
        null,
      ]);
  });

  it('rejects a constant, the raw input, or the target fed into the loss', () => {
    const base = createEditorState(delay).graph;
    const loss = lossNode(base);
    const withSource = (extra: GraphJSON['nodes'], from: string): GraphJSON => ({
      ...base,
      nodes: [...base.nodes, ...extra],
      edges: [
        ...base.edges,
        { from: { node: from, port: 'out' }, to: { node: loss.id, port: 'p' } },
      ],
    });
    const x = base.nodes.find((n) => n.attrs?.name === 'x')!.id;
    const t = base.nodes.find((n) => n.attrs?.name === 't')!.id;
    expect(
      trainDependencyError(
        delay,
        withSource([{ id: 'c', component: 'const', attrs: { value: 0, shape: [32] } }], 'c'),
        registry,
      ),
    ).toBe(TRAIN_DEPENDENCY_ERROR);
    expect(
      trainDependencyError(
        delay,
        withSource([{ id: 'w', component: 'param', attrs: { shape: [32] } }], 'w'),
        registry,
      ),
    ).toBe(TRAIN_DEPENDENCY_ERROR);
    expect(trainDependencyError(delay, withSource([], t), registry)).toBe(TRAIN_DEPENDENCY_ERROR);
    expect(trainDependencyError(delay, withSource([], x), registry)).toBe(TRAIN_DEPENDENCY_ERROR);
    expect(trainDependencyError(delay, base, registry)).toBe(TRAIN_DEPENDENCY_ERROR);
  });

  it('fails grading with the anti-cheat message', async () => {
    const level = trainLevels.find((l) => l.id === '3-4')!;
    const base = createEditorState(level).graph;
    const graph: GraphJSON = {
      ...base,
      nodes: [...base.nodes, { id: 'c', component: 'param', attrs: { shape: [16] } }],
      edges: [
        ...base.edges,
        { from: { node: 'c', port: 'out' }, to: { node: lossNode(base).id, port: 'p' } },
      ],
    };
    // Players reach 3-4 having earned MSE in 3-3.
    const result = await evaluateLevel(
      { ...level, available: [...level.available, 'mse'] },
      graph,
      {
        registry,
      },
    );
    expect(result.passed).toBe(false);
    expect(result.errorKey ?? result.error?.key).toBe(TRAIN_DEPENDENCY_ERROR);
  });
});
