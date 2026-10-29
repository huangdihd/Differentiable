import { expect, it } from 'vitest';
import { evaluateSuite } from './level-suite';
import { mul, tensor } from './tensor';
import type { CircuitLevel } from './levels';

it('combines forward and derivative checks without stopping after a failure', async () => {
  const base = {
    id: 'test',
    titleKey: 'test.title',
    descriptionKey: 'test.description',
    available: [],
    interface: { inputs: { x: [] }, output: 'y', outputShape: [] },
    unlock: [],
    fixed: [{ id: 'one', inputs: { x: [1] } }],
  };
  const tests: CircuitLevel[] = [
    { ...base, kind: 'forward', reference: ({ x }) => mul(x, tensor([1], [], x.backend)) },
    { ...base, kind: 'derivative', reference: ({ x }) => mul(x, x) },
  ];
  const result = await evaluateSuite(
    {
      id: 'mixed',
      chapter: 1,
      titleKey: 'mixed.title',
      descriptionKey: 'mixed.description',
      hintKeys: [],
      unlockComponents: [],
      tests,
    },
    {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'y', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'y', port: 'x' } }],
    },
  );
  expect(result.passed).toBe(false);
  expect(result.tests.map((t) => t.passed)).toEqual([true, false]);
});
