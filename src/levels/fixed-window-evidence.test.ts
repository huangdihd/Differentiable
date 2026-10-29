import { describe, expect, it } from 'vitest';
import { corpus, vocabulary } from './corpus';
import {
  fixedWindowContext,
  fixedWindowExamples,
  hasFixedWindowEvidence,
} from './fixed-window-evidence';

describe('fixed-window information conflict', () => {
  it('uses two real corpus continuations with the same last two characters and different targets', () => {
    expect(fixedWindowExamples).toHaveLength(2);
    expect(fixedWindowContext).toBe('th');
    for (const { history, target } of fixedWindowExamples) {
      expect(corpus).toContain(history + target);
      expect(history.slice(-2)).toBe(fixedWindowContext);
      expect(target).toHaveLength(1);
    }
    expect(fixedWindowExamples[0].history).not.toBe(fixedWindowExamples[1].history);
    expect(fixedWindowExamples.map(({ target }) => target)).toEqual([' ', 'e']);
    expect(hasFixedWindowEvidence()).toBe(true);
  });

  it('no shared vocabulary prediction can match both corpus continuations', () => {
    for (const prediction of vocabulary) {
      const matches = fixedWindowExamples.filter(({ target }) => target === prediction);
      expect(matches.length, JSON.stringify(prediction)).toBeLessThanOrEqual(1);
    }
  });
});
