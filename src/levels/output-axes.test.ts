import { describe, expect, it } from 'vitest';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import en from '../locales/en.json';
import zh from '../locales/zh-CN.json';
import { catalog } from './catalog';
import { matrixAxes } from './output-axes';

describe('matrix output labels', () => {
  for (const { level, code } of catalog) {
    if (level.kind === 'observation' || level.kind === 'train' || !level.outputAxes) continue;
    it(`${code}: labels match every fixed and generated case at every shape`, () => {
      expect(level.outputNoteKey).toBe(`levels.${level.id}.expectedNote`);
      for (const locale of [en, zh]) {
        expect(Object.hasOwn(locale, level.outputNoteKey!)).toBe(true);
        const note = (locale as Record<string, unknown>)[level.outputNoteKey!];
        expect(typeof note).toBe('string');
        expect(String(note).trim()).not.toBe('');
      }
      const suites = [
        { inputs: level.interface.inputs, fixed: level.fixed, generated: level.generated },
        ...(level.shapeVariants ?? []),
      ];
      for (const suite of suites) {
        const cases = [...suite.fixed];
        if (suite.generated) {
          const random = new Random(suite.generated.seed);
          for (let i = 0; i < suite.generated.count; i++)
            cases.push(suite.generated.generate(random, i));
        }
        for (const test of cases) {
          const inputs = Object.fromEntries(
            Object.entries(suite.inputs).map(([name, shape]) => [
              name,
              T.tensor(test.inputs[name], shape),
            ]),
          );
          const output = level.reference(inputs);
          try {
            const axes = level.outputAxes!(test.inputs);
            for (const [axis, labels] of [axes.rows, axes.cols].entries()) {
              if (labels === undefined) continue;
              expect(output.shape).toHaveLength(2);
              expect(labels).toHaveLength(output.shape[axis]);
            }
            if (output.shape.length === 2)
              expect((axes.rows?.length ?? 0) + (axes.cols?.length ?? 0)).toBeGreaterThan(0);
          } finally {
            T.disposeGraph([output, ...Object.values(inputs)]);
          }
        }
      }
    });
  }

  it('omits labels for unknown or ambiguous flat input shapes', () => {
    const labels = () => ({ rows: ['0', '1'] });
    expect(matrixAxes([{ x: [2, 3] }], labels)({ x: [1] })).toEqual({});
    expect(matrixAxes([{ x: [2, 3] }, { x: [3, 2] }], labels)({ x: [1, 2, 3, 4, 5, 6] })).toEqual(
      {},
    );
  });
});
