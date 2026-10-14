import { describe, expect, it } from 'vitest';
import { evaluateLevel, type ForwardLevel } from './levels';
import type { GraphJSON } from './graph';
import * as T from './tensor';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'out', component: 'output', attrs: { name: 'out' } },
  ],
  edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'out', port: 'x' } }],
};
const level: ForwardLevel = {
  id: 'variants',
  titleKey: 'test.title',
  descriptionKey: 'test.description',
  kind: 'forward',
  available: ['reshape'],
  unlock: [],
  interface: { inputs: { x: [2] }, output: 'out', outputShape: [2] },
  fixed: [{ id: 'fixed-0', inputs: { x: [1, 2] } }],
  reference: ({ x }) => T.reshape(x, x.shape),
  shapeVariants: [
    {
      inputs: { x: [1, 3] },
      fixed: [{ id: 'fixed-0', inputs: { x: [1, 2, 3] } }],
      generated: {
        seed: 42,
        count: 2,
        generate: (r, i) => ({
          id: `random-${i}`,
          inputs: { x: [r.uniform(), r.uniform(), r.uniform()] },
        }),
      },
    },
  ],
};
describe('shape variants', () => {
  it('recompiles each shape and reproducibly runs fixed and seeded cases', async () => {
    const a = await evaluateLevel(level, graph);
    expect(a.passed).toBe(true);
    expect(a.tests.map((t) => t.id)).toEqual([
      'fixed-0',
      'shape-1/fixed-0',
      'shape-1/random-0',
      'shape-1/random-1',
    ]);
    expect(await evaluateLevel(level, graph)).toEqual(a);
  });
  it('reports variant compilation failures with input shape context', async () => {
    const hardcoded: GraphJSON = {
      ...graph,
      nodes: [...graph.nodes, { id: 'r', component: 'reshape', attrs: { shape: [2] } }],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'r', port: 'x' } },
        { from: { node: 'r', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    const result = await evaluateLevel(level, hardcoded);
    expect(result.tests[0].passed).toBe(true);
    expect(result.passed).toBe(false);
    expect(result.tests[1]).toMatchObject({
      id: 'shape-1/compile',
      errorKey: 'levels.shape_variant',
      errorParams: { shapes: 'x [1×3]' },
    });
  });
  it('compares reference shapes, not just flattened lengths, on variants', async () => {
    const result = await evaluateLevel(
      { ...level, reference: ({ x }) => T.reshape(x, [x.shape.reduce((a, b) => a * b, 1)]) },
      graph,
    );
    expect(result.tests[0].passed).toBe(true);
    expect(result.tests[1].errorKey).toBe('levels.shape_variant');
    expect(result.passed).toBe(false);
  });
  it('still enforces the declared output shape on the primary shape', async () => {
    const result = await evaluateLevel(
      { ...level, interface: { ...level.interface, outputShape: [1, 2] } },
      graph,
    );
    expect(result.errorKey).toBe('levels.output_shape');
    expect(result.passed).toBe(false);
  });
});
