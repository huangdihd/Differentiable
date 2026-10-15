import { describe, expect, it } from 'vitest';
import { createEditorState, editorReducer } from './editor/state';
import { compileGraph, computationGraph, type GraphJSON, type GraphNode } from './engine/graph';
import { graphMetrics } from './engine/levels';
import { sigmoidLevel } from './levels/examples';
import { createInitialProgress, loadProgress, saveProgress } from './storage';
import { validateProperties } from './ui/Properties';
import { gradientParameterLabel, parameterName } from './ui/parameter-label';

function parameter(id: string, shape: number[] = [2, 2], label?: string): GraphNode {
  return {
    id,
    component: 'param',
    attrs: { shape, init: 'ones', ...(label === undefined ? {} : { label }) },
  };
}
function graph(label?: string): GraphJSON {
  return {
    version: 1,
    nodes: [
      parameter('weight', [2, 2], label),
      { id: 'out', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [{ from: { node: 'weight', port: 'out' }, to: { node: 'out', port: 'x' } }],
  };
}

describe('parameter labels', () => {
  it('sets and clears labels through validation and replacement edits, with undo and redo', () => {
    const initial = createEditorState(sigmoidLevel, graph());
    const base = initial.graph.nodes[0].attrs!;
    const named = validateProperties(
      'param',
      { shape: '2,2', init: 'ones', label: ' 权重_1 ' },
      base,
    );
    expect(named.errors).toEqual({});
    expect(named.attrs).toEqual({ ...base, label: '权重_1' });
    const set = editorReducer(initial, {
      type: 'setAttrs',
      id: 'weight',
      attrs: named.attrs,
      replace: true,
    });
    expect(set.error).toBeNull();
    expect(set.graph.nodes[0].attrs).toEqual(named.attrs);
    expect(editorReducer(set, { type: 'undo' }).graph).toEqual(initial.graph);
    expect(editorReducer(editorReducer(set, { type: 'undo' }), { type: 'redo' }).graph).toEqual(
      set.graph,
    );

    const blank = validateProperties(
      'param',
      { shape: '2,2', init: 'ones', label: ' \t ' },
      named.attrs,
    );
    expect(blank.errors).toEqual({});
    expect(blank.attrs).toEqual(base);
    expect(blank.attrs).not.toHaveProperty('label');
    const cleared = editorReducer(set, {
      type: 'setAttrs',
      id: 'weight',
      attrs: blank.attrs,
      replace: true,
    });
    expect(cleared.error).toBeNull();
    expect(cleared.graph.nodes[0].attrs).not.toHaveProperty('label');
    expect(editorReducer(cleared, { type: 'undo' }).graph).toEqual(set.graph);
    expect(editorReducer(editorReducer(cleared, { type: 'undo' }), { type: 'redo' }).graph).toEqual(
      cleared.graph,
    );
    expect(initial.graph.nodes[0].attrs).toEqual(base);
  });

  it('trims before truncating to 16 Unicode code points without splitting astral characters', () => {
    const name = '😀权重_1'.repeat(4);
    const result = validateProperties('param', { shape: '2', label: ` \t${name} \n` });
    expect(result.errors).toEqual({});
    expect(result.attrs.label).toBe(Array.from(name).slice(0, 16).join(''));
    expect(Array.from(result.attrs.label as string)).toHaveLength(16);
  });

  it.each(['权重😀', 'W_1', '_1'])(
    'preserves the literal custom name %s in both display helpers',
    (name) => {
      const g = graph(name);
      expect(parameterName(g.nodes[0])).toBe(name);
      expect(gradientParameterLabel(g.nodes[0], g)).toBe(name);
    },
  );

  it('allows duplicate custom names without adding subscripts', () => {
    const g: GraphJSON = {
      version: 1,
      nodes: [parameter('a', [2], 'W_1'), parameter('b', [2], 'W_1')],
      edges: [],
    };
    expect(g.nodes.map(parameterName)).toEqual(['W_1', 'W_1']);
    expect(g.nodes.map((node) => gradientParameterLabel(node, g))).toEqual(['W_1', 'W_1']);
  });

  it('retains default shape labels and same-shape subscripts in graph order', () => {
    const nodes = [
      parameter('a'),
      parameter('different', [3]),
      parameter('b'),
      parameter('scalar', []),
    ];
    const g: GraphJSON = { version: 1, nodes, edges: [] };
    expect(nodes.map(parameterName)).toEqual([undefined, undefined, undefined, undefined]);
    expect(nodes.map((node) => gradientParameterLabel(node, g))).toEqual([
      'θ₁ 2×2',
      'θ 3',
      'θ₂ 2×2',
      'θ',
    ]);
    const many: GraphJSON = {
      version: 1,
      nodes: Array.from({ length: 10 }, (_, i) => parameter(String(i), [])),
      edges: [],
    };
    expect(gradientParameterLabel(many.nodes[8], many)).toBe('θ₉');
    expect(gradientParameterLabel(many.nodes[9], many)).toBe('θ10');
  });

  it('compiles and executes identically with or without a label, with unchanged graphMetrics', async () => {
    const plain = graph();
    const named = graph('权重_1');
    expect(graphMetrics(named)).toEqual(graphMetrics(plain));
    expect(graphMetrics(named)).toEqual({ topNodes: 2, expandedPrimitives: 0, parameterCount: 4 });
    const a = compileGraph(plain);
    const b = compileGraph(named);
    try {
      expect(b.inputShapes).toEqual(a.inputShapes);
      expect(b.outputShapes).toEqual(a.outputShapes);
      const outA = a.execute({}).y;
      const outB = b.execute({}).y;
      expect(Array.from(await outA.read())).toEqual([1, 1, 1, 1]);
      expect(Array.from(await outB.read())).toEqual(Array.from(await outA.read()));
      expect([...b.parameters.keys()]).toEqual([...a.parameters.keys()]);
    } finally {
      a.dispose();
      b.dispose();
    }
  });

  it('roundtrips old unnamed and new named graphs through in-memory progress storage', () => {
    const data = new Map<string, string>();
    const source = () => ({
      getItem: (key: string) => data.get(key) ?? null,
      setItem: (key: string, value: string) => {
        data.set(key, value);
      },
    });
    for (const saved of [graph(), graph('权重😀_1')]) {
      const progress = {
        ...createInitialProgress(),
        passed: ['0-1'],
        savedGraphs: { '0-1': saved },
        passedSolutions: { '0-1': saved },
      };
      expect(saveProgress(progress, source)).toBe(true);
      const restored = loadProgress(source);
      expect(restored).toEqual(progress);
      expect(restored.savedGraphs['0-1']).not.toBe(saved);
      expect(parameterName(restored.savedGraphs['0-1'].nodes[0])).toBe(saved.nodes[0].attrs?.label);
    }
  });

  it('keeps computation cache JSON stable across naming, renaming and clearing', () => {
    const key = (g: GraphJSON) => JSON.stringify(computationGraph(g));
    expect(key(graph('W_1'))).toBe(key(graph()));
    expect(key(graph('新名字😀'))).toBe(key(graph('W_1')));
    for (const attrs of [
      { shape: [4], init: 'ones' },
      { shape: [2, 2], init: 'zeros' },
    ]) {
      const changed = graph('W_1');
      changed.nodes[0].attrs = { ...attrs, label: 'W_1' };
      expect(key(changed)).not.toBe(key(graph('W_1')));
    }
  });

  it('computationGraph removes only param attrs.label and never mutates the saved graph', () => {
    const original = graph('W_1');
    original.nodes[0].attrs!.custom = { label: 'nested' };
    original.nodes[0].ui = { x: 10, y: 20, label: 'UI metadata' };
    original.nodes[1].attrs!.label = 'output metadata';
    original.nodes.push({ id: 'bare', component: 'param' });
    const before = structuredClone(original);
    const expected = structuredClone(original);
    delete expected.nodes[0].attrs!.label;
    expect(computationGraph(original)).toEqual(expected);
    expect(original).toEqual(before);
  });
});
