import { expect, it } from 'vitest';
import { Random } from '../engine/random';
import {
  bigramEntropy,
  conditionalEntropy,
  contextSequences,
  contextBigramEntropy,
  contextEntropy,
  VOCAB,
  corpus,
  corpusBatch,
  corpusStats,
  corpusTokens,
  decode,
  encode,
  sampleToken,
  vocabulary,
  vocabularySize,
} from './corpus';
it('original corpus has a stable sorted vocabulary and lossless encoding', () => {
  expect(decode(encode(corpus))).toBe(corpus);
  expect(vocabulary).toEqual([...new Set(corpus)].sort());
  expect(corpusStats.characters).toBeGreaterThanOrEqual(1500);
  expect(corpusStats.characters).toBeLessThanOrEqual(3000);
  expect(corpus).toMatch(/^[a-z .\n]+$/);
  expect(VOCAB).toEqual(vocabulary);
  expect(vocabularySize).toBeLessThanOrEqual(30);
  expect(encode(decode([...Array(vocabularySize).keys()]))).toEqual([
    ...Array(vocabularySize).keys(),
  ]);
  expect(() => encode('☃')).toThrow('corpus.unknown_character');
  expect(() => decode([-1])).toThrow('corpus.invalid_token');
});
it('seeded batches contain adjacent contexts and one-hot next characters', async () => {
  for (const context of [1, 2] as const) {
    const a = corpusBatch(new Random(42), 24, context),
      b = corpusBatch(new Random(42), 24, context);
    try {
      const x = [...(await a.inputs.x.read())],
        targets = [...(await a.targets.read())];
      expect(x).toEqual([...(await b.inputs.x.read())]);
      expect(targets).toEqual([...(await b.targets.read())]);
      expect(a.inputs.x.shape).toEqual(context === 1 ? [24] : [24, 2]);
      for (let row = 0; row < 24; row++) {
        const target = targets.slice(row * vocabularySize, (row + 1) * vocabularySize);
        expect(target.reduce((sum, value) => sum + value, 0)).toBe(1);
        const sequence = [...x.slice(row * context, (row + 1) * context), target.indexOf(1)];
        expect(
          corpusTokens.some((_, start) =>
            sequence.every((token, i) => corpusTokens[start + i] === token),
          ),
        ).toBe(true);
      }
    } finally {
      for (const batch of [a, b]) {
        batch.inputs.x.dispose();
        batch.targets.dispose();
      }
    }
  }
});
it('sampling is seeded and validates weights', () => {
  const weights = vocabulary.map((_, i) => i + 1);
  const sample = () => {
    const random = new Random(12);
    return Array.from({ length: 100 }, () => sampleToken(weights, random));
  };
  expect(sample()).toEqual(sample());
  expect(() =>
    sampleToken(
      weights.map(() => 0),
      new Random(),
    ),
  ).toThrow();
  expect(() =>
    sampleToken(
      weights.map(() => NaN),
      new Random(),
    ),
  ).toThrow();
});

it('conditional entropy counts the exact corpus and the shared context distribution', () => {
  expect(
    conditionalEntropy(
      [
        [0, 0],
        [0, 1],
      ],
      1,
    ),
  ).toBeCloseTo(Math.log(2));
  expect(
    conditionalEntropy(
      [
        [0, 0],
        [1, 1],
      ],
      1,
    ),
  ).toBe(0);
  expect(bigramEntropy).toBeCloseTo(
    conditionalEntropy(
      corpusTokens.slice(1).map((t, i) => [corpusTokens[i], t]),
      1,
    ),
  );
  expect(contextSequences).toHaveLength(512);
  expect(contextBigramEntropy).toBeCloseTo(conditionalEntropy(contextSequences, 1));
  expect(contextEntropy).toBeLessThan(contextBigramEntropy - 0.15);
});
