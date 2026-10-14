import { describe, expect, it } from 'vitest';
import { caseInputs, fullValues, tensorBlocks, tensorRows } from './tensor-values';
import en from '../locales/en.json';
import zh from '../locales/zh-CN.json';

describe('read-only tensor values', () => {
  it('formats matrices in complete last-axis rows', () => {
    expect(tensorRows([2, 3], [1, 2, 3, 4, 5, 6])).toBe('1, 2, 3\n4, 5, 6');
    expect(tensorRows([3], [1, 2, 3])).toBe('1, 2, 3');
    expect(tensorRows([0], [])).toBe('');
  });
  it('stacks higher axes as separate matrix blocks', () => {
    const v = Array.from({ length: 12 }, (_, i) => i);
    expect(tensorBlocks([4], v.slice(0, 4))).toEqual([[[0, 1, 2, 3]]]);
    expect(tensorBlocks([2, 2], v.slice(0, 4))).toEqual([
      [
        [0, 1],
        [2, 3],
      ],
    ]);
    expect(tensorBlocks([3, 2, 2], v)).toHaveLength(3);
    expect(tensorBlocks([3, 2, 2], v)[2]).toEqual([
      [8, 9],
      [10, 11],
    ]);
  });
  it('never caps full values at 99 or rounds their precision', () => {
    const values = Array.from({ length: 128 }, (_, i) => i + 0.125);
    expect(fullValues(values).split(', ')).toHaveLength(128);
    expect(fullValues(values)).toContain('127.125');
  });
  it('switches all inputs without changing the source case', () => {
    const inputs = { a: [1, 2], b: [3, 4] };
    expect(caseInputs(inputs)).toEqual({ a: '1, 2', b: '3, 4' });
    expect(inputs).toEqual({ a: [1, 2], b: [3, 4] });
  });
  it('defines exactly six localized chapter-two lessons, each with two hints', () => {
    for (const locale of [en, zh]) {
      const keys = Object.keys(locale).filter((key) => key.startsWith('levels.2-'));
      expect(keys).toHaveLength(24);
      for (let i = 1; i <= 6; i++)
        for (const field of ['title', 'description', 'hint1', 'hint2'])
          expect(keys).toContain(`levels.2-${i}.${field}`);
      expect(locale).not.toHaveProperty('components.broadcast');
    }
  });
});
