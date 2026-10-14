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
  inputs?: string[];
  derivativeInput?: string;
  thirdColumn?: readonly number[];
  reference: ForwardLevel['reference'];
}[] = [
  { formula: '2 * x', reference: ({ x }) => T.mul(x, x) },
  { formula: 'exp(x) * (1 + x)', reference: ({ x }) => T.mul(x, T.exp(x)) },
  { formula: '3 * exp(3 * x)', reference: ({ x }) => T.exp(T.mul(c(x, 3), x)) },
  { formula: 'recip(x)', positiveInputs: ['x'], reference: ({ x }) => T.log(x) },
  { formula: 'sigmoid(x) + neg(sigmoid(x) * sigmoid(x))', reference: ({ x }) => sigmoid(x) },
  { formula: '1 + neg(tanh(x) * tanh(x))', reference: ({ x }) => tanh(x) },
  { formula: '2 * x * recip(1 + x * x)', reference: ({ x }) => T.log(T.add(c(x, 1), T.mul(x, x))) },
  {
    formula: 'b + exp(a)',
    inputs: ['a', 'b'],
    derivativeInput: 'a',
    reference: ({ a, b }) => T.add(T.mul(a, b), T.exp(a)),
  },
  {
    formula: 'a * a * exp(a * b)',
    inputs: ['a', 'b'],
    derivativeInput: 'b',
    reference: ({ a, b }) => T.mul(a, T.exp(T.mul(a, b))),
  },
  {
    formula: 'sigmoid(w * x + b) * (1 + neg(sigmoid(w * x + b))) * x',
    inputs: ['w', 'x', 'b'],
    derivativeInput: 'w',
    thirdColumn: [3, -1, 1, -0.5, 0.25, -1.5, 2, -2],
    reference: ({ w, x, b }) => sigmoid(T.add(T.mul(w, x), b)),
  },
];

export const chapter1Levels: ForwardLevel[] = definitions.map((definition, index) => {
  const id = `1-${index + 1}`;
  const inputs = definition.inputs ?? ['x'];
  return withPar(
    {
      id,
      chapter: '1',
      kind: 'derivative',
      titleKey: `levels.${id}.title`,
      descriptionKey: `levels.${id}.description`,
      hintsKeys: [`levels.${id}.hint1`, `levels.${id}.hint2`],
      available: [...scalarPrimitives, 'max'],
      interface: {
        inputs: Object.fromEntries(inputs.map((name) => [name, []])),
        output: 'y',
        outputShape: [],
      },
      unlock: [index === definitions.length - 1 ? '2-1' : `1-${index + 2}`],
      unlockFeatures:
        index === definitions.length - 1
          ? ['levels.features.autodiff', 'levels.features.expression']
          : [],
      derivativeInput: definition.derivativeInput ?? 'x',
      referenceGraph: formulaGraph(definition.formula, inputs),
      ...cases(8 + index, definition.positiveInputs, inputs, definition.thirdColumn),
      reference: definition.reference,
    },
    chapter0Levels,
  );
});
