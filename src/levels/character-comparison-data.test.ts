// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { corpus, decode } from './corpus';
import {
  characterComparisonLines,
  characterComparisonPositions,
} from './character-comparison-data';
import { fixedWindowExamples } from './fixed-window-evidence';

it('covers every eligible within-line target once, including newlines, without crossing lines', () => {
  const expected: number[] = [];
  let lineStart = 0;
  for (let position = 0; position < corpus.length; position++) {
    if (position - lineStart >= 2) expected.push(position);
    if (corpus[position] === '\n') lineStart = position + 1;
  }
  const actual = characterComparisonPositions.map(({ targetPosition }) => targetPosition);
  expect([...actual].sort((left, right) => left - right)).toEqual(expected);
  expect(new Set(actual).size).toBe(actual.length);
  for (const { tokens, start } of characterComparisonLines) {
    expect(decode(tokens)).toBe(corpus.slice(start, start + tokens.length));
    expect(decode(tokens).slice(0, -1)).not.toContain('\n');
  }
});

it('aligns fixed and recurrent histories with the exact same next character, never its input', () => {
  let previousOffset = 0;
  for (const { lineIndex, offset, targetPosition } of characterComparisonPositions) {
    const { start, tokens } = characterComparisonLines[lineIndex];
    const history = decode(tokens.slice(0, offset));
    const fixedInput = decode(tokens.slice(offset - 2, offset));
    expect(offset).toBeGreaterThanOrEqual(previousOffset);
    previousOffset = offset;
    expect(history).toBe(corpus.slice(start, targetPosition));
    expect(history.slice(-2)).toBe(fixedInput);
    expect(decode([tokens[offset]])).toBe(corpus[targetPosition]);
    expect(history).not.toContain('\n');
  }
});

it('includes both displayed positions without oversampling or special casing either example', () => {
  for (const { history, target } of fixedWindowExamples) {
    const position = corpus.indexOf(history + target) + history.length;
    expect(position).toBeGreaterThanOrEqual(history.length);
    const matches = characterComparisonPositions.filter(
      ({ targetPosition }) => targetPosition === position,
    );
    expect(matches).toHaveLength(1);
    const { lineIndex, offset } = matches[0];
    expect(decode(characterComparisonLines[lineIndex].tokens.slice(0, offset))).toBe(history);
  }
});
