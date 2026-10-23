import { describe, expect, it } from 'vitest';
import {
  evaluateLevel,
  levelDerivativeInputs,
  levelOutputShapes,
  type ForwardLevel,
} from './levels';
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

const gradientGraph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'y', component: 'input', attrs: { name: 'y' } },
    { id: 'dx', component: 'output', attrs: { name: 'dx' } },
    { id: 'dy', component: 'output', attrs: { name: 'dy' } },
  ],
  edges: [
    { from: { node: 'y', port: 'out' }, to: { node: 'dx', port: 'x' } },
    { from: { node: 'x', port: 'out' }, to: { node: 'dy', port: 'x' } },
  ],
};
const gradientLevel: ForwardLevel = {
  id: 'multi-variants',
  titleKey: 'test.title',
  descriptionKey: 'test.description',
  kind: 'derivative',
  available: ['reshape'],
  unlock: [],
  interface: {
    inputs: { x: [2], y: [2] },
    output: 'dx',
    outputShape: [2],
    outputs: { dx: [2], dy: [2] },
  },
  derivativeInputs: ['x', 'y'],
  fixed: [{ id: 'fixed', inputs: { x: [1, 2], y: [3, 4] } }],
  shapeVariants: [
    {
      inputs: { x: [1, 3], y: [1, 3] },
      fixed: [{ id: 'fixed', inputs: { x: [2, 3, 4], y: [6, 5, 4] } }],
    },
  ],
  reference: ({ x, y }) => T.mul(x, y),
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
      errorParams: { shapes: { x: [1, 3] } },
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

  it('evaluates the declared number of outputs for the primary and every variant shape', async () => {
    expect(levelOutputShapes(gradientLevel)).toEqual({ dx: [2], dy: [2] });
    expect(levelDerivativeInputs(gradientLevel)).toEqual(['x', 'y']);
    const result = await evaluateLevel(gradientLevel, gradientGraph);
    expect(result.passed).toBe(true);
    expect(result.tests).toEqual([
      {
        id: 'fixed',
        passed: true,
        actual: [3, 4, 1, 2],
        expected: [3, 4, 1, 2],
      },
      {
        id: 'shape-1/fixed',
        passed: true,
        actual: [6, 5, 4, 2, 3, 4],
        expected: [6, 5, 4, 2, 3, 4],
      },
    ]);
    expect(result.tests.every((test) => test.actual.length === test.expected.length)).toBe(true);
  });

  it('validates every required output on variant compilation, not only the primary output', async () => {
    const hardcodedSecondary: GraphJSON = {
      ...gradientGraph,
      nodes: [
        ...gradientGraph.nodes,
        { id: 'reshape-y', component: 'reshape', attrs: { shape: [2] } },
      ],
      edges: [
        gradientGraph.edges[0],
        { from: { node: 'x', port: 'out' }, to: { node: 'reshape-y', port: 'x' } },
        { from: { node: 'reshape-y', port: 'out' }, to: { node: 'dy', port: 'x' } },
      ],
    };
    const result = await evaluateLevel(gradientLevel, hardcodedSecondary);
    expect(result.tests[0].passed).toBe(true);
    expect(result.tests[0].actual).toEqual([3, 4, 1, 2]);
    expect(result.tests[1]).toMatchObject({
      id: 'shape-1/compile',
      passed: false,
      errorKey: 'levels.shape_variant',
      errorParams: { shapes: { x: [1, 3], y: [1, 3] } },
    });
    expect(result.passed).toBe(false);
  });

  it('returns structured failures for an inconsistent primary shape and a missing output', async () => {
    const badPrimary = await evaluateLevel(
      {
        ...gradientLevel,
        interface: { ...gradientLevel.interface, outputShape: [1] },
      },
      gradientGraph,
    );
    expect(badPrimary).toMatchObject({
      passed: false,
      tests: [],
      errorKey: 'levels.output_shape',
    });

    const missingOutput: GraphJSON = {
      ...gradientGraph,
      nodes: gradientGraph.nodes.filter((node) => node.id !== 'dy'),
      edges: gradientGraph.edges.filter((edge) => edge.to.node !== 'dy'),
    };
    const missing = await evaluateLevel(gradientLevel, missingOutput);
    expect(missing).toMatchObject({
      passed: false,
      tests: [],
      errorKey: 'levels.output_shape',
    });
  });

  it('returns a structured failure when derivative inputs cannot map one-to-one to outputs', async () => {
    for (const derivativeInputs of [['x'], ['x', 'x'], ['x', 'missing']] as const) {
      const malformed = { ...gradientLevel, derivativeInputs };
      expect(() => levelDerivativeInputs(malformed)).toThrowError('levels.output_shape');
      await expect(evaluateLevel(malformed, gradientGraph)).resolves.toMatchObject({
        passed: false,
        tests: [],
        errorKey: 'levels.output_shape',
      });
    }
  });
});
