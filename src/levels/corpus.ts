// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { shuffledOrder } from '../datasets';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';

/** Original nursery rhyme written for this project. Training data, not UI copy.
 * Keep spelling stable: token IDs and calibration depend on this text.
 */
export const corpus = `little moth with silver feet.
tap a tune along the street.
little mouse with velvet toes.
ring the bell where clover grows.

round the pond and round the tree.
sing a tiny song with me.
moon above and moss below.
rock the seeds and watch them grow.

penny wren has made a nest.
soft as clouds for birds to rest.
benny bear has sewn a sail.
small enough to fit a snail.

sail away on drops of dew.
bring a buttercup or two.
bring a leaf and bring a light.
keep the sleepy garden bright.

little fox in socks of green.
sweeps the stepping stones all clean.
little frog in boots of red.
tucks the beetles into bed.

hush the hill and hush the lane.
hear the gentle tapping rain.
drip a drop and hum a hum.
tell the morning it may come.

when the sun begins to peep.
moths and mice fall fast asleep.
when the stars begin to shine.
all the garden joins the rhyme.

penny wren will count to three.
one for you and one for me.
one to hide beneath a rose.
where the drowsy cricket goes.

benny bear will stir the tea.
sweet with petals from the tree.
fox will pour a cup of rain.
frog will sing the tune again.

round the pond and round the tree.
sing a tiny song with me.
moon above and moss below.
rock the seeds and watch them grow.

little snail will mend the sail.
with a thread as soft as a tail.
little bee will bring a drum.
beat it gently tum tum tum.

on a leaf the friends will ride.
past the stones and down the tide.
back before the morning light.
home to kiss the flowers good night.

hush the hill and hush the lane.
hear the gentle tapping rain.
drip a drop and hum a hum.
tell the morning it may come.
`;

export const CORPUS = corpus;

/** Sorted Unicode characters give stable IDs without a hand-maintained alphabet. */
export const vocabulary = [...new Set(corpus)].sort();
export const VOCAB = vocabulary;
export const vocabularySize = vocabulary.length;
const tokenIds = new Map(vocabulary.map((character, index) => [character, index]));
export function encode(text: string): number[] {
  return [...text].map((character) => {
    const id = tokenIds.get(character);
    if (id === undefined) throw new Error('corpus.unknown_character');
    return id;
  });
}
export function decode(tokens: readonly number[]): string {
  return tokens
    .map((id) => {
      if (!Number.isInteger(id) || id < 0 || id >= vocabularySize)
        throw new Error('corpus.invalid_token');
      return vocabulary[id];
    })
    .join('');
}
export const corpusTokens = encode(corpus);
export const corpusStats = {
  characters: corpusTokens.length,
  vocabularySize,
  bigrams: corpusTokens.length - 1,
  trigrams: corpusTokens.length - 2,
};

/** Sample contiguous contexts; targets are the immediately following character.
 * Every call returns fresh owned tensors and uses only the supplied seeded PRNG.
 */
export function corpusBatch(random: Random, count: number, context: 1 | 2) {
  const x: number[] = [];
  const targets = new Float32Array(count * vocabularySize);
  for (let row = 0; row < count; row++) {
    const start = Math.floor(random.uniform() * (corpusTokens.length - context));
    x.push(...corpusTokens.slice(start, start + context));
    targets[row * vocabularySize + corpusTokens[start + context]] = 1;
  }
  return {
    inputs: { x: T.tensor(x, context === 1 ? [count] : [count, context]) },
    targets: T.tensor(targets, [count, vocabularySize]),
  };
}

/** Categorical sampling with explicit seed ownership; accepts unnormalized weights. */
export function sampleToken(weights: ArrayLike<number>, random: Random): number {
  if (weights.length !== vocabularySize) throw new Error('corpus.invalid_probabilities');
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    if (!Number.isFinite(weights[i]) || weights[i] < 0)
      throw new Error('corpus.invalid_probabilities');
    total += weights[i];
  }
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('corpus.invalid_probabilities');
  let draw = random.uniform() * total;
  for (let i = 0; i < weights.length; i++) {
    draw -= weights[i];
    if (draw < 0) return i;
  }
  return weights.length - 1;
}

/** Exact empirical conditional entropy in nats, counting every supplied transition. */
export function conditionalEntropy(
  sequences: readonly (readonly number[])[],
  context: number,
): number {
  const counts = new Map<string, Map<number, number>>();
  for (const sequence of sequences) {
    const key = sequence.slice(-context - 1, -1).join(',');
    const next = sequence[sequence.length - 1];
    const row = counts.get(key) ?? new Map<number, number>();
    row.set(next, (row.get(next) ?? 0) + 1);
    counts.set(key, row);
  }
  let entropy = 0;
  for (const row of counts.values()) {
    const total = [...row.values()].reduce((a, b) => a + b, 0);
    for (const count of row.values()) entropy -= count * Math.log(count / total);
  }
  return entropy / sequences.length;
}
export const bigramEntropy = conditionalEntropy(
  corpusTokens.slice(1).map((next, i) => [corpusTokens[i], next]),
  1,
);
/** All corpus pairs, with no sampling error in the bigram calibration.
 * With `order`, the same pairs come back shuffled: grading uses them so a [N×V] Parameter
 * table fitted to training rows cannot pass without reading x.
 */
export function corpusPairs(order?: Random) {
  const count = corpusTokens.length - 1;
  const rows = order ? shuffledOrder(order, count) : Array.from({ length: count }, (_, i) => i);
  const targets = new Float32Array(count * vocabularySize);
  rows.forEach((i, row) => {
    targets[row * vocabularySize + corpusTokens[i + 1]] = 1;
  });
  return {
    inputs: {
      x: T.tensor(
        rows.map((i) => corpusTokens[i]),
        [count],
      ),
    },
    targets: T.tensor(targets, [count, vocabularySize]),
  };
}
/** A fixed seeded empirical distribution shared by training and evaluation in 4-5. */
export const contextSequences = (() => {
  const random = new Random(4505);
  return Array.from({ length: 512 }, () => {
    const start = Math.floor(random.uniform() * (corpusTokens.length - 2));
    return corpusTokens.slice(start, start + 3);
  });
})();
export const contextBigramEntropy = conditionalEntropy(contextSequences, 1);
export const contextEntropy = conditionalEntropy(contextSequences, 2);
/** With `order`, the same 512 samples shuffled (grading order; see corpusPairs). */
export function contextBatch(order?: Random) {
  const sequences = order
    ? shuffledOrder(order, contextSequences.length).map((i) => contextSequences[i])
    : contextSequences;
  const targets = new Float32Array(sequences.length * vocabularySize);
  sequences.forEach((sequence, i) => {
    targets[i * vocabularySize + sequence[2]] = 1;
  });
  return {
    inputs: {
      x: T.tensor(
        sequences.flatMap((s) => s.slice(0, 2)),
        [sequences.length, 2],
      ),
    },
    targets: T.tensor(targets, [sequences.length, vocabularySize]),
  };
}
