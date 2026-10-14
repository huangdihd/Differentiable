import { parseExpression } from '../engine/expression';
import {
  ComponentRegistry,
  createDefaultRegistry,
  type GraphJSON,
  type ShapePorts,
} from '../engine/graph';
import { graphMetrics, type ForwardLevel, type Level } from '../engine/levels';

/** A small authoring helper, not a separate evaluator: answers are ordinary GraphJSON. */
export function formulaGraph(formula: string, inputs: readonly string[] = ['x']): GraphJSON {
  // `(-1)` would otherwise lower to mul(1, -1); bind each negative literal to one const node.
  const literals: { name: string; value: number }[] = [];
  formula = formula.replace(/\(-(\d+(?:\.\d+)?)\)/g, (_, digits: string) => {
    const name = `neg_literal_${literals.length}`;
    literals.push({ name, value: -Number(digits) });
    return name;
  });
  const parsed = parseExpression(formula, {
    unaryMinus: {
      component: 'mul',
      inputs: { a: 0, b: { component: 'const', inputs: {}, attrs: { value: -1 } } },
    },
    variables: Object.fromEntries(
      [...inputs, ...literals.map((l) => l.name)].map((name) => [
        name,
        { node: name, port: 'out' },
      ]),
    ),
    functions: Object.fromEntries(
      ['exp', 'log', 'neg', 'recip', 'sigmoid', 'tanh', 'relu', 'sqrt', 'dsigmoid', 'dtanh'].map(
        (name) => [name, { component: name, inputs: ['x'] }],
      ),
    ),
  });
  // Share repeated subexpressions (sigmoid(x) used twice is one node), so par is what a player
  // wiring the formula by hand would reach. Parser output lists operands before their users.
  const alias = new Map<string, string>();
  const seen = new Map<string, string>();
  const nodes = parsed.nodes.filter((node) => {
    const incoming = parsed.edges
      .filter((e) => e.to.node === node.id)
      .map((e) => `${e.to.port}=${alias.get(e.from.node) ?? e.from.node}.${e.from.port}`)
      .sort();
    const key = JSON.stringify([node.component, node.attrs ?? null, incoming]);
    const existing = seen.get(key);
    if (existing) alias.set(node.id, existing);
    else seen.set(key, node.id);
    return !existing;
  });
  const remap = (end: { node: string; port: string }) => ({
    ...end,
    node: alias.get(end.node) ?? end.node,
  });
  return {
    version: 1,
    nodes: [
      ...inputs.map((name) => ({ id: name, component: 'input', attrs: { name } })),
      ...literals.map((l) => ({ id: l.name, component: 'const', attrs: { value: l.value } })),
      ...nodes,
      { id: 'output', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [
      ...parsed.edges
        .filter((e) => !alias.has(e.to.node))
        .map((e) => ({ from: remap(e.from), to: e.to })),
      { from: remap(parsed.output), to: { node: 'output', port: 'x' } },
    ],
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
  const definitions = new Map(
    createDefaultRegistry()
      .list()
      .map((item) => [item.name, item]),
  );
  for (const level of levels) {
    for (const name of level.unlockComponents ?? []) {
      const existing = definitions.get(name);
      // Native-only unlocks grant access; they are not implementations of the lesson.
      if (existing && !existing.subgraph) continue;
      definitions.set(name, {
        ...existing,
        name,
        inputs: Object.keys(level.interface.inputs),
        outputs: ['out'],
        subgraph: componentAnswer(level.referenceGraph!),
      });
    }
  }
  const registry = new ComponentRegistry();
  for (const definition of definitions.values()) registry.register(definition);
  return registry;
}

export function withPar<L extends Level>(level: L, preceding: readonly ForwardLevel[]): L {
  return {
    atol: 1e-5,
    rtol: 1e-4,
    forbidden: [],
    ...level,
    par: graphMetrics(level.referenceGraph!, referenceRegistry(preceding)).expandedPrimitives,
  };
}

/**
 * Scalar test cases. Chapters 0-1 work on single numbers, so every case feeds one value per input;
 * the editor steps through them one at a time when verifying.
 */
export function cases(
  index: number,
  positiveInputs: readonly string[] = [],
  inputs: readonly string[] = ['x'],
  thirdColumn?: readonly number[],
) {
  const positive = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5];
  const first = [-3, -2, -1, -0.25, 0, 0.5, 1, 3];
  const second = [2, -1, 0.5, 3, -2, 0, -0.5, 1];
  return {
    fixed: first.map((_, k) => ({
      id: `fixed-${k}`,
      inputs: Object.fromEntries(
        inputs.map((name, i) => [
          name,
          [
            positiveInputs.includes(name)
              ? positive[k]
              : i === 2 && thirdColumn
                ? thirdColumn[k]
                : i
                  ? second[k]
                  : first[k],
          ],
        ]),
      ),
    })),
    generated: {
      seed: 1000 + index,
      count: 4,
      generate: (random: import('../engine/random').Random, i: number) => ({
        id: `seeded-${i}`,
        inputs: Object.fromEntries(
          inputs.map((name) => [
            name,
            [positiveInputs.includes(name) ? 0.1 + random.uniform() * 5 : random.uniform() * 6 - 3],
          ]),
        ),
      }),
    },
  };
}

/** Shape-aware deterministic fixtures, including zeros, constants and mixed signs. */
export function tensorCases(index: number, inputs: ShapePorts) {
  return {
    fixed: Array.from({ length: 5 }, (_, k) => ({
      id: `tensor-${k}`,
      inputs: Object.fromEntries(
        Object.entries(inputs).map(([name, shape], port) => [
          name,
          Array.from({ length: shape.reduce((n, d) => n * d, 1) }, (_, i) =>
            k === 0
              ? 0
              : k === 1
                ? 2
                : k === 2
                  ? (((i * 3 + port * 2) % 11) - 5) / 2
                  : k === 3
                    ? 20 + ((i + port) % 5)
                    : -20 + ((i * 2 + port) % 7),
          ),
        ]),
      ),
    })),
    generated: {
      seed: 2000 + index,
      count: 4,
      generate: (random: import('../engine/random').Random, i: number) => ({
        id: `seeded-${i}`,
        inputs: Object.fromEntries(
          Object.entries(inputs).map(([name, shape]) => [
            name,
            Array.from(
              { length: shape.reduce((n, d) => n * d, 1) },
              () => random.uniform() * 8 - 4,
            ),
          ]),
        ),
      }),
    },
  };
}
