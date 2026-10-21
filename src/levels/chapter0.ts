import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { cases, formulaGraph, scalarPrimitives, withPar } from './level-helpers';

const constant = (x: T.Tensor, value: number) => T.tensor([value], [], x.backend);
const neg = (x: T.Tensor) => T.mul(x, constant(x, -1));
const sigmoid = (x: T.Tensor) => T.recip(T.add(constant(x, 1), T.exp(neg(x))));

const definitions: {
  /** Lessons inserted after release keep a stable named ID so saved progress still lines up. */
  id?: string;
  /** Seed slot for generated cases; numbered lessons use their position among numbered lessons. */
  caseIndex?: number;
  /** Retired lessons keep their slot so later numbered IDs stay stable; they build no level. */
  retired?: boolean;
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
    id: 'square',
    caseIndex: 8,
    name: 'square',
    formula: 'x * x',
    reference: ({ x }) => T.mul(x, x),
  },
  // Reciprocal is now a basic primitive, so its lesson (0-3) is retired.
  { name: 'recip', retired: true, formula: '', reference: ({ x }) => T.recip(x) },
  {
    name: 'div',
    formula: 'a * recip(b)',
    inputs: ['a', 'b'],
    nonzeroInputs: ['b'],
    reference: ({ a, b }) => T.mul(a, T.recip(b)),
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
const numbered = definitions.filter((definition) => !definition.id);
const idOf = (definition: (typeof definitions)[number]) =>
  definition.id ?? `0-${numbered.indexOf(definition) + 1}`;
const live = definitions.filter((definition) => !definition.retired);
for (const [position, definition] of live.entries()) {
  const id = idOf(definition);
  const index = definition.caseIndex ?? numbered.indexOf(definition);
  const next = live[position + 1];
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
        available: ['relu', 'sqrt'].includes(definition.name)
          ? [...scalarPrimitives, 'max']
          : scalarPrimitives,
        forbidden: [definition.name],
        interface: {
          inputs: Object.fromEntries(inputs.map((name) => [name, []])),
          output: 'y',
          outputShape: [],
        },
        unlock: [next ? idOf(next) : '1-1'],
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
