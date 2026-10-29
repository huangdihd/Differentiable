import { corpus, encode } from './corpus';

/** Keep the newline as a target, but never carry state across lines. */
export const characterComparisonLines = [...corpus.matchAll(/[^\n]*\n|[^\n]+$/g)]
  .filter(([line]) => line.length >= 3)
  .map((match) => ({ start: match.index!, tokens: encode(match[0]) }));

/** Time-major order lets the RNN share prefix computation without changing target weights.
 * Both models see every within-line target with at least two preceding characters, once.
 */
export const characterComparisonPositions = Array.from(
  { length: Math.max(...characterComparisonLines.map(({ tokens }) => tokens.length)) - 2 },
  (_, index) => index + 2,
).flatMap((offset) =>
  characterComparisonLines.flatMap(({ start, tokens }, lineIndex) =>
    offset < tokens.length ? [{ lineIndex, offset, targetPosition: start + offset }] : [],
  ),
);
