import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { cases, formulaGraph, scalarPrimitives, withPar } from './level-helpers';

const constant = (x: T.Tensor, value: number) => T.tensor([value], [], x.backend);
const neg = (x: T.Tensor) => T.mul(x, constant(x, -1));
const recip = (x: T.Tensor) => T.mul(x, T.exp(neg(T.log(T.mul(x, x)))));
const sigmoid = (x: T.Tensor) => recip(T.add(constant(x, 1), T.exp(neg(x))));

const definitions: {
  name: string;
  formula: string;
  inputs?: string[];
  positiveInputs?: string[];
  nonzeroInputs?: string[];
  reference: ForwardLevel['reference'];
}[] = [
  { name: 'neg', formula: 'x * (-1)', reference: ({ x }) => neg(x) },
  {
    name: 'sub',
    formula: 'a + neg(b)',
    inputs: ['a', 'b'],
    reference: ({ a, b }) => T.add(a, neg(b)),
  },
  {
    name: 'recip',
    formula: 'x * exp(neg(log(x * x)))',
    nonzeroInputs: ['x'],
    reference: ({ x }) => recip(x),
  },
  {
    name: 'div',
    formula: 'a * recip(b)',
    inputs: ['a', 'b'],
    nonzeroInputs: ['b'],
    reference: ({ a, b }) => T.mul(a, recip(b)),
  },
  { name: 'sigmoid', formula: 'recip(1 + exp(neg(x)))', reference: ({ x }) => sigmoid(x) },
  {
    name: 'tanh',
    formula: '(exp(2 * x) + (-1)) * recip(exp(2 * x) + 1)',
    reference: ({ x }) =>
      T.add(T.mul(constant(x, 2), sigmoid(T.mul(constant(x, 2), x))), constant(x, -1)),
  },
  { name: 'relu', formula: '', reference: ({ x }) => T.max(x, constant(x, 0)) },
  {
    name: 'sqrt',
    formula: 'exp(0.5 * log(x))',
    positiveInputs: ['x'],
    reference: ({ x }) => T.exp(T.mul(constant(x, 0.5), T.log(x))),
  },
];

export const chapter0Levels: ForwardLevel[] = [];
for (const [index, definition] of definitions.entries()) {
  const id = `0-${index + 1}`;
  const inputs = definition.inputs ?? ['x'];
  const graph =
    definition.name === 'relu'
      ? {
          version: 1 as const,
          nodes: [
            { id: 'x', component: 'input', attrs: { name: 'x' } },
            { id: 'zero', component: 'const', attrs: { value: 0 } },
            { id: 'relu', component: 'max' },
            { id: 'output', component: 'output', attrs: { name: 'y' } },
          ],
          edges: [
            { from: { node: 'x', port: 'out' }, to: { node: 'relu', port: 'a' } },
            { from: { node: 'zero', port: 'out' }, to: { node: 'relu', port: 'b' } },
            { from: { node: 'relu', port: 'out' }, to: { node: 'output', port: 'x' } },
          ],
        }
      : formulaGraph(definition.formula, inputs);
  chapter0Levels.push(
    withPar(
      {
        id,
        chapter: '0',
        kind: 'forward',
        titleKey: `levels.${id}.title`,
        descriptionKey: `levels.${id}.description`,
        hintsKeys: [`levels.${id}.hint1`, `levels.${id}.hint2`],
        available: index < 6 ? scalarPrimitives : [...scalarPrimitives, 'max'],
        forbidden: [definition.name],
        interface: {
          inputs: Object.fromEntries(inputs.map((name) => [name, []])),
          output: 'y',
          outputShape: [],
        },
        unlock: [index === 7 ? '1-1' : `0-${index + 2}`],
        unlockComponents: [definition.name],
        referenceGraph: graph,
        ...(definition.nonzeroInputs ? { revalidateOnLoad: true } : {}),
        ...cases(index, definition.positiveInputs, inputs, undefined, definition.nonzeroInputs),
        reference: definition.reference,
      },
      chapter0Levels,
    ),
  );
}
