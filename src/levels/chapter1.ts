import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import {
  cases,
  formulaGraph,
  formulaOutputsGraph,
  scalarPrimitives,
  withPar,
} from './level-helpers';

const c = (x: T.Tensor, n: number) => T.tensor([n], [], x.backend);
const sigmoid = (x: T.Tensor) =>
  T.exp(T.mul(T.log(T.add(c(x, 1), T.exp(T.mul(c(x, -1), x)))), c(x, -1)));
const tanh = (x: T.Tensor) => T.add(T.mul(c(x, 2), sigmoid(T.mul(c(x, 2), x))), c(x, -1));

// reference is f, never f': the engine derives the numerical oracle by autodiff.
const definitions: {
  formula: string;
  /** The derivative the player builds here becomes a part for later levels. */
  component?: string;
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
  {
    formula: 'sub(sigmoid(x), sigmoid(x) * sigmoid(x))',
    component: 'dsigmoid',
    reference: ({ x }) => sigmoid(x),
  },
  { formula: 'sub(1, tanh(x) * tanh(x))', component: 'dtanh', reference: ({ x }) => tanh(x) },
  { formula: 'div(2 * x, 1 + x * x)', reference: ({ x }) => T.log(T.add(c(x, 1), T.mul(x, x))) },
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
    // σ′ was built in 1-5; here it is reused through the chain rule, not derived again.
    formula: 'dsigmoid(w * x + b) * x',
    inputs: ['w', 'x', 'b'],
    derivativeInput: 'w',
    thirdColumn: [3, -1, 1, -0.5, 0.25, -1.5, 2, -2],
    reference: ({ w, x, b }) => sigmoid(T.add(T.mul(w, x), b)),
  },
];

export const chapter1Levels: ForwardLevel[] = [];
const derivativeOutputs = ['f′', 'f′', 'f′', 'f′', 'σ′', 'tanh′', 'f′', '∂f/∂a', '∂f/∂b'];
for (const [index, definition] of definitions.entries()) {
  const id = `1-${index + 1}`;
  const inputs = definition.inputs ?? ['x'];
  const output = derivativeOutputs[index];
  const level = withPar<ForwardLevel>(
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
        output,
        outputShape: [],
      },
      unlock: [index === definitions.length - 1 ? '1-10' : `1-${index + 2}`],
      unlockFeatures:
        index === definitions.length - 1
          ? ['levels.features.autodiff', 'levels.features.expression']
          : [],
      derivativeInput: definition.derivativeInput ?? 'x',
      ...(definition.component ? { unlockComponents: [definition.component] } : {}),
      referenceGraph: formulaGraph(definition.formula, inputs, output),
      ...cases(8 + index, definition.positiveInputs, inputs, definition.thirdColumn),
      reference: definition.reference,
    },
    [...chapter0Levels, ...chapter1Levels],
  );
  chapter1Levels.push(level);
}

// Assemble both partial derivatives into one gradient, then use that vector idea as the bridge
// into a local reverse pass that carries one upstream gradient back through a Linear node.
const backpropLevel = chapter1Levels.pop()!;
chapter1Levels.at(-1)!.unlock = ['1-10'];

const inputs = ['a', 'b'];
const gradientFormulas = { '∂a': 'b + exp(a)', '∂b': 'a' };
chapter1Levels.push(
  withPar<ForwardLevel>(
    {
      id: '1-10',
      chapter: '1',
      kind: 'derivative',
      titleKey: 'levels.1-10.title',
      descriptionKey: 'levels.1-10.description',
      hintsKeys: ['levels.1-10.hint1', 'levels.1-10.hint2'],
      available: [...scalarPrimitives, 'max'],
      interface: {
        inputs: { a: [], b: [] },
        output: '∂a',
        outputShape: [],
        outputs: { '∂a': [], '∂b': [] },
      },
      unlock: ['1-11'],
      derivativeInput: 'a',
      derivativeInputs: inputs,
      referenceGraph: formulaOutputsGraph(gradientFormulas, inputs),
      ...cases(18, [], inputs),
      reference: ({ a, b }) => T.add(T.mul(a, b), T.exp(a)),
    },
    [...chapter0Levels, ...chapter1Levels],
  ),
);
const upstream = '∂f/∂z';
const backpropGraph: NonNullable<ForwardLevel['referenceGraph']> = {
  version: 1,
  nodes: [
    { id: 'w', component: 'input', attrs: { name: 'w' } },
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'g', component: 'input', attrs: { name: upstream } },
    { id: 'dw', component: 'mul' },
    { id: 'dx', component: 'mul' },
    { id: 'out_w', component: 'output', attrs: { name: '∂f/∂w' } },
    { id: 'out_x', component: 'output', attrs: { name: '∂f/∂x' } },
    { id: 'out_b', component: 'output', attrs: { name: '∂f/∂b' } },
  ],
  edges: [
    { from: { node: 'g', port: 'out' }, to: { node: 'dw', port: 'a' } },
    { from: { node: 'x', port: 'out' }, to: { node: 'dw', port: 'b' } },
    { from: { node: 'g', port: 'out' }, to: { node: 'dx', port: 'a' } },
    { from: { node: 'w', port: 'out' }, to: { node: 'dx', port: 'b' } },
    { from: { node: 'dw', port: 'out' }, to: { node: 'out_w', port: 'x' } },
    { from: { node: 'dx', port: 'out' }, to: { node: 'out_x', port: 'x' } },
    { from: { node: 'g', port: 'out' }, to: { node: 'out_b', port: 'x' } },
  ],
};
const revealGraph: NonNullable<ForwardLevel['revealGraph']> = {
  graph: {
    version: 1,
    nodes: [
      { id: 'w', component: 'input', attrs: { name: 'w' }, ui: { y: 0 } },
      { id: 'x', component: 'input', attrs: { name: 'x' }, ui: { y: 80 } },
      { id: 'b', component: 'input', attrs: { name: 'b' }, ui: { y: 160 } },
      { id: 'product', component: 'mul', attrs: { labelKey: 'gradView.backprop.product' } },
      { id: 'z', component: 'add', attrs: { labelKey: 'gradView.backprop.z' } },
      { id: 'f', component: 'sigmoid', attrs: { labelKey: 'gradView.backprop.sigmoid' } },
      { id: 'output', component: 'output', attrs: { name: 'f' } },
    ],
    edges: [
      { from: { node: 'w', port: 'out' }, to: { node: 'product', port: 'a' } },
      { from: { node: 'x', port: 'out' }, to: { node: 'product', port: 'b' } },
      { from: { node: 'product', port: 'out' }, to: { node: 'z', port: 'a' } },
      { from: { node: 'b', port: 'out' }, to: { node: 'z', port: 'b' } },
      { from: { node: 'z', port: 'out' }, to: { node: 'f', port: 'x' } },
      { from: { node: 'f', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  },
  inputShapes: { w: [], x: [], b: [] },
  output: 'f',
  wrt: ['w', 'x', 'b'],
  inputs: { w: [1], x: [0.5], b: [0.25] },
  introKey: 'gradView.backprop.intro',
  playerBackwardStart: 'z',
  playerBackwardLabelKey: 'gradView.backprop.playerRegion',
  engineBackwardLabelKey: 'gradView.backprop.engineRegion',
};
Object.assign(backpropLevel, {
  id: '1-11',
  titleKey: 'levels.1-11.title',
  descriptionKey: 'levels.1-11.description',
  hintsKeys: ['levels.1-11.hint1', 'levels.1-11.hint2'],
  interface: {
    inputs: { w: [], x: [], [upstream]: [] },
    output: '∂f/∂w',
    outputShape: [],
    outputs: { '∂f/∂w': [], '∂f/∂x': [], '∂f/∂b': [] },
  },
  unlock: ['axes'],
  referenceGraph: backpropGraph,
  referenceOutputs: (values: Record<string, T.Tensor>) => ({
    '∂f/∂w': T.mul(values[upstream], values.x),
    '∂f/∂x': T.mul(values[upstream], values.w),
    '∂f/∂b': values[upstream],
  }),
  revalidateOnLoad: true,
  ...cases(19, [], ['w', 'x', upstream], [1, -1, 0.5, 2, -0.25, 0, 1.5, -2]),
  reference: ({ w, x }: Record<string, T.Tensor>) => T.mul(w, x),
  revealGraph,
});
delete backpropLevel.derivativeInput;
chapter1Levels.push(withPar(backpropLevel, [...chapter0Levels, ...chapter1Levels]));
