import { expect, it } from 'vitest';
import { evaluateLevel } from '../engine/levels';
import type { GraphJSON } from '../engine/graph';
import {
  createExampleRegistry,
  derivativeLevel,
  derivativeSolution,
  sigmoidLevel,
  sigmoidSolution,
  xorLevel,
  xorSolution,
} from './examples';

const identity: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'y', component: 'output', attrs: { name: 'y' } },
  ],
  edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'y', port: 'x' } }],
};
it('derivative solution matches autodiff of x exp(x)', async () => {
  const result = await evaluateLevel(derivativeLevel, derivativeSolution, {
    registry: createExampleRegistry(),
  });
  expect(result.errorKey).toBeUndefined();
  expect(result.passed).toBe(true);
});
it('incorrect forward and derivative solutions fail actual numerical tests', async () => {
  for (const level of [sigmoidLevel, derivativeLevel]) {
    const result = await evaluateLevel(level, identity);
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(false);
    expect(result.tests.some((test) => !test.passed)).toBe(true);
  }
});
it('seeded XOR MLP trains below CE .05 at perfect accuracy', async () => {
  const result = await evaluateLevel(xorLevel, xorSolution, { registry: createExampleRegistry() });
  expect(result.errorKey).toBeUndefined();
  expect(result.training?.loss).toBeLessThan(0.05);
  expect(result.training?.accuracy).toBe(1);
  expect(result.passed).toBe(true);
}, 20000);
it('incorrect XOR constant classifier cannot pass', async () => {
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      {
        id: 'p',
        component: 'const',
        attrs: { shape: [4, 2], value: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] },
      },
      { id: 'y', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [{ from: { node: 'p', port: 'out' }, to: { node: 'y', port: 'x' } }],
  };
  const result = await evaluateLevel({ ...xorLevel, steps: 2 }, graph);
  expect(result.errorKey).toBeUndefined();
  expect(result.passed).toBe(false);
  expect(result.training?.accuracy).toBe(0.5);
});

it('correct sigmoid forward solution passes fixed and seeded cases', async () => {
  const result = await evaluateLevel(sigmoidLevel, sigmoidSolution, {
    registry: createExampleRegistry(),
  });
  expect(result.errorKey).toBeUndefined();
  expect(result.passed).toBe(true);
  expect(result.tests).toHaveLength(4);
  expect(result.tests.every((test) => test.passed)).toBe(true);
});
