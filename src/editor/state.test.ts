import { describe, expect, it } from 'vitest';
import { createExampleRegistry, sigmoidLevel } from '../levels/examples';
import { chapter1Levels } from '../levels/chapter1';
import { parseExpression } from '../engine/expression';
import {
  createEditorState,
  editorReducer,
  nodePosition,
  screenToWorld,
  worldToScreen,
  zoomAt,
  clampZoom,
  type EditorState,
  type EditorCommand,
} from './state';
const registry = createExampleRegistry();
const reduce = (s: EditorState, c: EditorCommand) => editorReducer(s, c, registry);
const initial = () => createEditorState(sigmoidLevel);
const add = (s: EditorState, component = 'add') =>
  reduce(s, { type: 'addNode', component, position: { x: 250, y: 200 } });
const edge = (from: string, to: string, port = 'a') => ({
  from: { node: from, port: 'out' },
  to: { node: to, port },
});
const wire = (s: EditorState, from: string, to: string, port = 'a') =>
  reduce(s, { type: 'connect', ...edge(from, to, port) });

describe('editor graph commands', () => {
  it('builds level interface and clones saved graphs', () => {
    const s = initial();
    expect(s.graph.nodes.map((n) => [n.component, n.attrs?.name])).toEqual([
      ['input', 'x'],
      ['output', 'y'],
    ]);
    const loaded = createEditorState(sigmoidLevel, s.graph);
    expect(loaded.graph).toEqual(s.graph);
    expect(loaded.graph).not.toBe(s.graph);
    expect(s.graph.nodes.every((n) => Number.isFinite(nodePosition(n).x))).toBe(true);
  });
  it('initializes every declared output with stable IDs, names, and positions', () => {
    const outputs = (id: string) =>
      createEditorState(chapter1Levels.find((level) => level.id === id)!)
        .graph.nodes.filter((node) => node.component === 'output')
        .map(({ id: nodeId, attrs, ui }) => ({ id: nodeId, name: attrs?.name, ui }));
    expect(outputs('1-10')).toEqual([
      { id: 'output', name: '∂a', ui: { x: 660, y: 120 } },
      { id: 'output_1', name: '∂b', ui: { x: 660, y: 320 } },
    ]);
    expect(outputs('1-11')).toEqual([
      { id: 'output', name: '∂f/∂w', ui: { x: 660, y: 120 } },
      { id: 'output_1', name: '∂f/∂x', ui: { x: 660, y: 320 } },
      { id: 'output_2', name: '∂f/∂b', ui: { x: 660, y: 520 } },
    ]);
  });
  it('accepts player components passed in beyond the level primitives', () => {
    expect(sigmoidLevel.available).not.toContain('matmul');
    expect(add(initial(), 'matmul').error).toBe('unavailable_component');
    const s = createEditorState(sigmoidLevel, undefined, [...sigmoidLevel.available, 'matmul']);
    const next = add(s, 'matmul');
    expect(next.error).toBeNull();
    expect(next.graph.nodes.some((n) => n.component === 'matmul')).toBe(true);
  });
  it('adds deterministically, undoes/redoes and clears the redo branch', () => {
    const s = initial(),
      a = add(s);
    expect(add(s)).toEqual(a);
    expect(s.graph.nodes).toHaveLength(2);
    expect(a.selection).toEqual(['node_1']);
    const u = reduce(a, { type: 'undo' });
    expect(u.graph).toEqual(s.graph);
    expect(reduce(u, { type: 'redo' }).graph).toEqual(a.graph);
    expect(add(u, 'mul').future).toEqual([]);
    expect(reduce(s, { type: 'undo' })).toBe(s);
    expect(add(s, 'unknown').error).toBe('unavailable_component');
    expect(add(s, 'input').graph).toBe(s.graph);
  });
  it('removes incident edges but preserves interface nodes and their attributes', () => {
    let s = wire(add(initial()), 'input_0', 'node_1');
    s = wire(s, 'node_1', 'output', 'x');
    const removed = reduce(s, { type: 'removeNodes', ids: s.graph.nodes.map((n) => n.id) });
    expect(removed.graph).toEqual(initial().graph);
    expect(reduce(removed, { type: 'undo' }).graph).toEqual(s.graph);
    expect(reduce(s, { type: 'setAttrs', id: 'input_0', attrs: { name: 'bad' } }).error).toBe(
      'protected_node',
    );
  });
  it('validates ports and nodes, replaces single inputs, rejects self and indirect cycles', () => {
    const s = wire(add(add(initial())), 'input_0', 'node_1');
    const replaced = wire(s, 'node_2', 'node_1');
    expect(replaced.graph.edges).toEqual([edge('node_2', 'node_1')]);
    expect(wire(replaced, 'node_1', 'node_2').error).toBe('cycle');
    expect(wire(s, 'node_1', 'node_1').graph).toBe(s.graph);
    expect(wire(s, 'absent', 'node_1').error).toBe('missing_node');
    expect(wire(s, 'input_0', 'node_1', 'wrong').error).toBe('invalid_port');
    const invalid = reduce(s, {
      type: 'connect',
      from: { node: 'output', port: 'out' },
      to: { node: 'node_1', port: 'a' },
    });
    expect(invalid.error).toBe('invalid_port');
  });
  it('rewires atomically, preserving original edge on rejection and undo', () => {
    const s = wire(add(add(initial())), 'input_0', 'node_1');
    const bad = reduce(s, { type: 'rewire', edge: s.graph.edges[0], ...edge('node_2', 'node_2') });
    expect(bad.graph).toBe(s.graph);
    expect(bad.past).toBe(s.past);
    const good = reduce(s, {
      type: 'rewire',
      edge: s.graph.edges[0],
      ...edge('input_0', 'node_2'),
    });
    expect(good.graph.edges).toEqual([edge('input_0', 'node_2')]);
    expect(reduce(good, { type: 'undo' }).graph).toEqual(s.graph);
    expect(reduce(good, { type: 'disconnect', edge: good.graph.edges[0] }).graph.edges).toEqual([]);
  });
  it('groups movement into one undo, cancels, and rejects nonfinite positions', () => {
    const s = add(initial());
    let drag = reduce(s, { type: 'beginMove' });
    for (let x = 1; x <= 10; x++)
      drag = reduce(drag, { type: 'moveNodes', positions: { node_1: { x, y: 50 } } });
    expect(drag.past).toHaveLength(s.past.length);
    expect(reduce(drag, { type: 'cancelMove' }).graph).toEqual(s.graph);
    const done = reduce(drag, { type: 'commitMove' });
    expect(done.past).toHaveLength(s.past.length + 1);
    expect(reduce(done, { type: 'undo' }).graph).toEqual(s.graph);
    expect(reduce(s, { type: 'moveNodes', positions: { node_1: { x: NaN, y: 0 } } }).graph).toBe(
      s.graph,
    );
    expect(reduce(reduce(s, { type: 'beginMove' }), { type: 'commitMove' }).past).toEqual(s.past);
  });
  it('supports shift toggle and additive reverse-direction box selection without history', () => {
    let s = add(initial());
    s = reduce(s, { type: 'select', ids: ['input_0', 'node_1'], shift: true });
    expect(s.selection).toEqual(['input_0']);
    s = reduce(s, {
      type: 'boxSelect',
      rect: { x: 450, y: 320, width: -210, height: -130 },
      shift: true,
    });
    expect(s.selection).toEqual(['input_0', 'node_1']);
    expect(s.past).toHaveLength(1);
    expect(reduce(s, { type: 'select', ids: ['absent'] }).selection).toEqual([]);
  });
  it('copies only internal non-interface graph, remaps IDs, offsets and undoes one paste', () => {
    let s = wire(wire(add(add(initial())), 'input_0', 'node_1'), 'node_1', 'node_2');
    s = reduce(s, { type: 'select', ids: s.graph.nodes.map((n) => n.id) });
    s = reduce(s, { type: 'copy' });
    expect(s.clipboard?.nodes).toHaveLength(2);
    expect(s.clipboard?.edges).toEqual([edge('node_1', 'node_2')]);
    const pasted = reduce(s, { type: 'paste' });
    expect(pasted.selection).toEqual(['node_3', 'node_4']);
    expect(pasted.graph.edges.at(-1)).toEqual(edge('node_3', 'node_4'));
    expect(nodePosition(pasted.graph.nodes.at(-1)!)).toEqual({ x: 282, y: 232 });
    expect(reduce(pasted, { type: 'undo' }).graph).toEqual(s.graph);
    expect(reduce(s, { type: 'paste' })).toEqual(pasted);
    expect(reduce(pasted, { type: 'paste' }).selection).toEqual(['node_5', 'node_6']);
  });
  it('merges attributes without mutating command payloads', () => {
    const s = add(initial(), 'const'),
      attrs = { value: [2], shape: [1] };
    const changed = reduce(s, { type: 'setAttrs', id: 'node_1', attrs });
    attrs.value[0] = 9;
    expect(changed.graph.nodes.at(-1)?.attrs).toEqual({ value: [2], shape: [1] });
    expect(reduce(changed, { type: 'undo' }).graph).toEqual(s.graph);
  });
  it('replaces property attributes so clearing optional fields survives undo/redo', () => {
    let s = add(initial(), 'const');
    s = reduce(s, { type: 'setAttrs', id: 'node_1', attrs: { value: 1, shape: [2] } });
    const changed = reduce(s, {
      type: 'setAttrs',
      id: 'node_1',
      attrs: { value: 1 },
      replace: true,
    });
    expect(changed.graph.nodes.at(-1)?.attrs).toEqual({ value: 1 });
    const undone = reduce(changed, { type: 'undo' });
    expect(undone.graph).toEqual(s.graph);
    expect(reduce(undone, { type: 'redo' }).graph).toEqual(changed.graph);
  });
  it('rejects cyclic paste atomically without consuming IDs or redo history', () => {
    const s = reduce(add(initial()), { type: 'undo' });
    const withClipboard: EditorState = {
      ...s,
      clipboard: {
        version: 1,
        nodes: [
          { id: 'a', component: 'add' },
          { id: 'b', component: 'add' },
        ],
        edges: [edge('a', 'b'), edge('b', 'a')],
      },
    };
    const rejected = reduce(withClipboard, { type: 'paste' });
    expect(rejected.error).toBe('cycle');
    expect(rejected.graph).toBe(s.graph);
    expect(rejected.past).toBe(s.past);
    expect(rejected.future).toBe(s.future);
    expect(rejected.nextId).toBe(s.nextId);
  });
  it('rejects edits during a drag and restores the redo branch when cancelled', () => {
    const s = reduce(add(add(initial())), { type: 'undo' });
    const moving = reduce(reduce(s, { type: 'beginMove' }), {
      type: 'moveNodes',
      positions: { node_1: { x: 9, y: 8 } },
    });
    expect(reduce(moving, { type: 'paste' }).error).toBe('movement_in_progress');
    const cancelled = reduce(moving, { type: 'cancelMove' });
    expect(cancelled.graph).toEqual(s.graph);
    expect(cancelled.future).toBe(s.future);
  });
  it('inserts an expression as one graph command with external variables and output replacement', () => {
    const s = add(initial());
    const expression = parseExpression('x + 2', {
      variables: { x: { node: 'input_0', port: 'out' } },
    });
    const command: EditorCommand = {
      type: 'insertExpression',
      expression,
      position: { x: 300, y: 400 },
      to: { node: 'output', port: 'x' },
    };
    const result = reduce(s, command);
    expect(result.error).toBeNull();
    expect(result.graph.nodes).toHaveLength(5);
    expect(result.graph.edges).toHaveLength(3);
    expect(result.past).toHaveLength(s.past.length + 1);
    expect(reduce(result, { type: 'undo' }).graph).toEqual(s.graph);
    const bad = reduce(s, { ...command, to: { node: 'input_0', port: 'x' } });
    expect(bad.graph).toBe(s.graph);
    expect(bad.nextId).toBe(s.nextId);
  });
});

describe('viewport helpers', () => {
  it('clamps zoom and preserves world position under cursor even at bounds', () => {
    const view = { x: 20, y: -50, zoom: 0.8 },
      cursor = { x: 240, y: 170 };
    for (const z of [0.001, 1.5, 100]) {
      const next = zoomAt(view, cursor, z);
      expect(screenToWorld(cursor, next)).toEqual(screenToWorld(cursor, view));
      expect(worldToScreen(screenToWorld(cursor, next), next)).toEqual(cursor);
    }
    expect(clampZoom(NaN)).toBe(1);
    const s = initial();
    expect(reduce(s, { type: 'setViewport', viewport: zoomAt(view, cursor, 100) }).past).toEqual(
      [],
    );
  });
});
