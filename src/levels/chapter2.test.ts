import { describe, expect, it } from 'vitest';
import { compileGraph, createDefaultRegistry, type GraphJSON } from '../engine/graph';
import { applyResult, evaluateLevel, graphMetrics, initialProgress } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { gameLevels } from './catalog';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { availableComponents, buildPlayerRegistry } from './registry';
import { referenceRegistry } from './level-helpers';

const preceding = [...chapter0Levels, ...chapter1Levels];
const registry = referenceRegistry([...preceding, ...chapter2Levels]);
const available = (index: number) =>
  [...preceding, ...chapter2Levels.slice(0, index)].flatMap(
    (level) => level.unlockComponents ?? [],
  );

describe('tensor curriculum', () => {
  it('preserves the six tensor interfaces and restricted primitive palette', () => {
    expect(chapter2Levels.map((level) => level.unlockComponents)).toEqual(
      ['dot', 'mean', 'outer', 'matmul', 'softmax', 'layernorm'].map((name) => [name]),
    );
    expect(
      chapter2Levels.map((level) => [
        level.interface.inputs,
        level.interface.outputShape,
        level.shapeVariants!.map((variant) => variant.inputs),
      ]),
    ).toEqual([
      [{ a: [6], b: [6] }, [], [{ a: [3], b: [3] }]],
      [{ x: [4, 6] }, [4, 1], [{ x: [3, 5] }, { x: [2, 3, 4] }]],
      [{ a: [4], b: [3] }, [4, 3], [{ a: [2], b: [5] }]],
      [
        { a: [4, 3], b: [3, 5] },
        [4, 5],
        [
          { a: [2, 6], b: [6, 3] },
          { a: [2, 4, 3], b: [3, 5] },
        ],
      ],
      [{ x: [4, 6] }, [4, 6], [{ x: [3, 5] }]],
      [{ x: [4, 8] }, [4, 8], [{ x: [3, 5] }]],
    ]);
    for (const level of chapter2Levels) {
      expect([...level.available].sort()).toEqual(
        ['const', 'add', 'mul', 'exp', 'log', 'max', 'sum', 'reduce_max', 'unsqueeze'].sort(),
      );
      expect(level.rtol).toBe(1e-4);
      expect(level.atol).toBe(level.id === '2-6' ? 1e-3 : 1e-4);
    }
    expect(
      chapter2Levels[2].referenceGraph!.nodes.find((node) => node.id === 'right')!.attrs,
    ).toEqual({ axis: 0 });
    expect(preceding.map((level) => level.par)).toEqual([
      1, 2, 3, 4, 6, 8, 1, 3, 1, 3, 3, 3, 9, 11, 7, 2, 4, 12,
    ]);
    expect(chapter2Levels.map((level) => level.par)).toEqual([2, 8, 3, 4, 9, 27]);
  });

  for (const [name, index] of [
    ['dot without reduction', 0],
    ['fixed mean', 1],
    ['outer without inserted axes', 2],
    ['layernorm without epsilon', 5],
    ['reversed matmul', 3],
    ['unstable softmax', 4],
  ] as const) {
    it(`rejects ${name}`, async () => {
      const level = chapter2Levels[index];
      const graph = structuredClone(level.referenceGraph!);
      if (index === 0) {
        graph.edges.find((edge) => edge.to.node === 'output')!.from.node = 'products';
      } else if (index === 2) {
        graph.edges.find((edge) => edge.to.node === 'result' && edge.to.port === 'a')!.from.node =
          'a';
        graph.edges.find((edge) => edge.to.node === 'result' && edge.to.port === 'b')!.from.node =
          'b';
      } else if (index === 5) {
        graph.nodes.find((node) => node.id === 'epsilon')!.attrs = { value: 0 };
      } else if (index === 1) {
        graph.nodes.push({ id: 'six', component: 'const', attrs: { value: 6 } });
        graph.edges.find((edge) => edge.to.node === 'result' && edge.to.port === 'b')!.from.node =
          'six';
      } else if (index === 3) {
        for (const edge of graph.edges) {
          if (edge.from.node === 'a') edge.from.node = 'b';
          else if (edge.from.node === 'b') edge.from.node = 'a';
        }
      } else {
        graph.edges.find((edge) => edge.to.node === 'weights')!.from.node = 'x';
      }
      const result = await evaluateLevel(
        { ...level, available: [...level.available, ...available(index)] },
        graph,
        { registry },
      );
      expect(result.passed).toBe(false);
      if (index === 1) {
        expect(
          result.tests.filter((test) => !test.id.startsWith('shape-')).every((test) => test.passed),
        ).toBe(true);
        const failures = result.tests.filter((test) => !test.passed);
        expect(failures.length).toBeGreaterThan(0);
        expect(
          failures.every(
            (test) => test.id.startsWith('shape-') && test.errorKey === 'levels.shape_variant',
          ),
        ).toBe(true);
        expect(failures[0].errorParams?.shapes).toContain('3×5');
      }
      if (index === 4) {
        expect(result.tests.some((test) => test.id.includes('large-logits') && !test.passed)).toBe(
          true,
        );
      }
    });
  }

  for (const [index, level] of chapter2Levels.entries()) {
    it(`${level.id}: reference passes all shapes with deterministic fixtures and correct par`, async () => {
      const result = await evaluateLevel(
        { ...level, available: [...level.available, ...available(index)] },
        level.referenceGraph!,
        { registry },
      );
      expect(result.errorKey).toBeUndefined();
      expect(result.tests.filter((test) => !test.passed)).toEqual([]);
      expect(result.passed).toBe(true);
      const suites = [level, ...level.shapeVariants!];
      expect(result.tests).toHaveLength(
        suites.reduce(
          (count, suite) => count + suite.fixed.length + (suite.generated?.count ?? 0),
          0,
        ),
      );
      expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
      for (const suite of suites) {
        const generator = suite.generated!;
        expect(generator.generate(new Random(generator.seed), 0)).toEqual(
          generator.generate(new Random(generator.seed), 0),
        );
      }
    });
    it(`${level.id}: rejects a shape-correct constant answer`, async () => {
      const graph: GraphJSON = {
        version: 1,
        nodes: [
          ...Object.keys(level.interface.inputs).map((name) => ({
            id: name,
            component: 'input',
            attrs: { name },
          })),
          {
            id: 'zero',
            component: 'const',
            attrs: { shape: [...level.interface.outputShape], value: 0 },
          },
          { id: 'out', component: 'output', attrs: { name: 'y' } },
        ],
        edges: [{ from: { node: 'zero', port: 'out' }, to: { node: 'out', port: 'x' } }],
      };
      const result = await evaluateLevel({ ...level, shapeVariants: [] }, graph, { registry });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(false);
      expect(result.tests.some((test) => !test.passed && !test.errorKey)).toBe(true);
    });
  }

  it('unlocks the linear route, uses passed graphs, and preserves XOR before and after earned components', async () => {
    let progress = initialProgress(['0-1']);
    const xor = gameLevels.at(-1)!;
    expect(buildPlayerRegistry(gameLevels, progress).get('matmul')?.native).toBe(
      createDefaultRegistry().get('matmul')!.native,
    );
    for (const level of gameLevels.filter((entry) => entry.kind !== 'train')) {
      if (level.id === '2-1') {
        const before = await evaluateLevel(
          { ...xor, available: availableComponents(xor, progress) },
          xor.referenceGraph!,
          { registry: buildPlayerRegistry(gameLevels, progress) },
        );
        expect(before.passed).toBe(true);
        expect(before.training?.loss).toBeLessThan(0.05);
        expect(before.training?.accuracy).toBe(1);
      }
      expect(progress.unlocked).toContain(level.id);
      expect(availableComponents(level, progress)).not.toContain(level.unlockComponents?.[0]);
      const result = await evaluateLevel(
        { ...level, available: availableComponents(level, progress) },
        level.referenceGraph!,
        { registry: buildPlayerRegistry(gameLevels, progress) },
      );
      expect(result.passed, level.id).toBe(true);
      progress = applyResult(progress, level, result, level.referenceGraph!);
    }
    expect(progress.unlocked).toContain('xor');
    expect(availableComponents(gameLevels.at(-1)!, progress)).toEqual(
      expect.arrayContaining(['matmul', 'softmax']),
    );
    const player = buildPlayerRegistry(gameLevels, progress);
    const after = await evaluateLevel(
      { ...xor, available: availableComponents(xor, progress) },
      xor.referenceGraph!,
      { registry: player },
    );
    expect(after.passed).toBe(true);
    expect(after.training?.loss).toBeLessThan(0.05);
    expect(after.training?.accuracy).toBe(1);
    for (const name of ['matmul', 'softmax', 'layernorm']) {
      const level = chapter2Levels.find((entry) => entry.unlockComponents?.includes(name))!;
      const definition = player.get(name)!;
      expect(definition.native).toBe(createDefaultRegistry().get(name)!.native);
      expect(typeof definition.subgraph).toBe('object');
      const graph: GraphJSON = {
        version: 1,
        nodes: [
          ...Object.keys(level.interface.inputs).map((input) => ({
            id: input,
            component: 'input',
            attrs: { name: input },
          })),
          { id: 'call', component: name },
          { id: 'out', component: 'output', attrs: { name: 'y' } },
        ],
        edges: [
          ...Object.keys(level.interface.inputs).map((input) => ({
            from: { node: input, port: 'out' },
            to: { node: 'call', port: input },
          })),
          { from: { node: 'call', port: 'out' }, to: { node: 'out', port: 'x' } },
        ],
      };
      const inputs = Object.fromEntries(
        Object.entries(level.fixed[2].inputs).map(([key, data]) => [
          key,
          T.tensor(data, [...level.interface.inputs[key]]),
        ]),
      );
      const outputs: number[][] = [];
      try {
        for (const preferNative of [true, false]) {
          const plan = compileGraph(graph, {
            registry: player,
            inputShapes: level.interface.inputs,
            preferNative,
          });
          try {
            const result = plan.execute(inputs);
            try {
              outputs.push(Array.from(await result.y.read()));
            } finally {
              plan.releaseExecution(result);
            }
          } finally {
            plan.dispose();
          }
        }
        outputs[0].forEach((value, i) => expect(outputs[1][i]).toBeCloseTo(value, 4));
      } finally {
        Object.values(inputs).forEach((input) => input.dispose());
      }
    }
  });
});
