import { parseExpression } from '../engine/expression';
import { createDefaultRegistry, type GraphJSON } from '../engine/graph';
import { graphMetrics, type ForwardLevel } from '../engine/levels';

/** A small authoring helper, not a separate evaluator: answers are ordinary GraphJSON. */
export function formulaGraph(formula: string, inputs: readonly string[] = ['x']): GraphJSON {
  const parsed = parseExpression(formula, {
    unaryMinus: {
      component: 'mul',
      inputs: { a: 0, b: { component: 'const', inputs: {}, attrs: { value: -1 } } },
    },
    variables: Object.fromEntries(inputs.map((name) => [name, { node: name, port: 'out' }])),
    functions: Object.fromEntries(
      ['exp', 'log', 'neg', 'recip', 'sigmoid', 'tanh', 'relu', 'sqrt'].map((name) => [
        name,
        { component: name, inputs: ['x'] },
      ]),
    ),
  });
  return {
    version: 1,
    nodes: [
      ...inputs.map((name) => ({ id: name, component: 'input', attrs: { name } })),
      ...parsed.nodes,
      { id: 'output', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [...parsed.edges, { from: parsed.output, to: { node: 'output', port: 'x' } }],
  };
}

export const scalarPrimitives = ['const', 'add', 'mul', 'exp', 'log'] as const;

export function componentAnswer(graph: GraphJSON): GraphJSON {
  const copy = structuredClone(graph);
  for (const node of copy.nodes) if (node.component === 'output') node.attrs = { name: 'out' };
  return copy;
}

/** Authoring registry for scoring/reference tests; runtime uses the shared game registry. */
export function referenceRegistry(levels: readonly ForwardLevel[]) {
  const registry = createDefaultRegistry();
  for (const level of levels) {
    for (const name of level.unlockComponents ?? []) {
      registry.register({
        name,
        inputs: Object.keys(level.interface.inputs),
        outputs: ['out'],
        subgraph: componentAnswer(level.referenceGraph!),
      });
    }
  }
  return registry;
}

export function withPar(level: ForwardLevel, preceding: readonly ForwardLevel[]): ForwardLevel {
  return {
    atol: 1e-5,
    rtol: 1e-4,
    forbidden: [],
    ...level,
    par: graphMetrics(level.referenceGraph!, referenceRegistry(preceding)).expandedPrimitives,
  };
}

export function cases(
  index: number,
  positiveInputs: readonly string[] = [],
  inputs: readonly string[] = ['x'],
) {
  return {
    fixed: [
      {
        id: 'fixed',
        inputs: Object.fromEntries(
          inputs.map((name, i) => [
            name,
            positiveInputs.includes(name)
              ? [0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
              : i
                ? [2, -1, 0.5, 3, -2, 0, -0.5, 1]
                : [-3, -2, -1, -0.25, 0, 0.5, 1, 3],
          ]),
        ),
      },
    ],
    generated: {
      seed: 1000 + index,
      count: 5,
      generate: (random: import('../engine/random').Random, i: number) => ({
        id: `seeded-${i}`,
        inputs: Object.fromEntries(
          inputs.map((name) => [
            name,
            Array.from({ length: 8 }, () =>
              positiveInputs.includes(name) ? 0.1 + random.uniform() * 5 : random.uniform() * 6 - 3,
            ),
          ]),
        ),
      }),
    },
  };
}
