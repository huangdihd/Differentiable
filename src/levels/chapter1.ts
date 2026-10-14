import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { cases, formulaGraph, scalarPrimitives, withPar } from './level-helpers';

const c = (x: T.Tensor, n: number) => T.tensor([n], [], x.backend);
const sigmoid = (x: T.Tensor) =>
  T.exp(T.mul(T.log(T.add(c(x, 1), T.exp(T.mul(c(x, -1), x)))), c(x, -1)));
const tanh = (x: T.Tensor) => T.add(T.mul(c(x, 2), sigmoid(T.mul(c(x, 2), x))), c(x, -1));

// reference is f, never f': the engine derives the numerical oracle by autodiff.
const definitions: {
  formula: string;
  positiveInputs?: string[];
  reference: ForwardLevel['reference'];
}[] = [
  { formula: '2 * x', reference: ({ x }) => T.mul(x, x) },
  { formula: 'exp(x) * (1 + x)', reference: ({ x }) => T.mul(x, T.exp(x)) },
  { formula: '3 * exp(3 * x)', reference: ({ x }) => T.exp(T.mul(c(x, 3), x)) },
  { formula: 'recip(x)', positiveInputs: ['x'], reference: ({ x }) => T.log(x) },
  { formula: 'sigmoid(x) + neg(sigmoid(x) * sigmoid(x))', reference: ({ x }) => sigmoid(x) },
  { formula: '1 + neg(tanh(x) * tanh(x))', reference: ({ x }) => tanh(x) },
  { formula: '2 * x * recip(1 + x * x)', reference: ({ x }) => T.log(T.add(c(x, 1), T.mul(x, x))) },
];

export const chapter1Levels: ForwardLevel[] = definitions.map((definition, index) => {
  const id = `1-${index + 1}`;
  return withPar(
    {
      id,
      chapter: '1',
      kind: 'derivative',
      titleKey: `levels.${id}.title`,
      descriptionKey: `levels.${id}.description`,
      hintsKeys: [`levels.${id}.hint1`, `levels.${id}.hint2`],
      available: [...scalarPrimitives, 'max'],
      interface: { inputs: { x: [] }, output: 'y', outputShape: [] },
      unlock: [index === 6 ? 'xor' : `1-${index + 2}`],
      unlockFeatures: index === 6 ? ['levels.features.autodiff', 'levels.features.expression'] : [],
      derivativeInput: 'x',
      referenceGraph: formulaGraph(definition.formula),
      ...cases(8 + index, definition.positiveInputs),
      reference: definition.reference,
    },
    chapter0Levels,
  );
});
