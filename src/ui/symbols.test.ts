import { expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import { ComponentRegistry } from '../engine/graph';
import { categoryOf, nodePorts, portRows, SYMBOLS } from './graph-geometry';

it('reshape dimension ports and node height follow the target rank', () => {
  const registry = new ComponentRegistry().register({
    name: 'reshape',
    inputs: ['x', 'd0', 'd1', 'd2', 'd3'],
    outputs: ['out'],
    optionalInputs: ['d0', 'd1', 'd2', 'd3'],
  });
  for (const rank of [0, 1, 2, 3, 4, 5]) {
    const node = { id: 'r', component: 'reshape', attrs: { shape: Array(rank).fill(1) } };
    expect(nodePorts(node, registry, 'inputs')).toEqual([
      'x',
      ...Array.from({ length: Math.min(rank, 4) }, (_, i) => `d${i}`),
    ]);
    expect(portRows(node, registry)).toBe(1 + Math.min(rank, 4));
    expect(nodePorts(node, registry, 'outputs')).toEqual(['out']);
  }
  expect(nodePorts({ id: 'r', component: 'reshape' }, registry, 'inputs')).toEqual(['x', 'd0']);
});

it('hides generated parameter ports only for fresh automatic module instances', () => {
  const registry = new ComponentRegistry().register({
    name: 'linear',
    inputs: ['x', 'w', 'b'],
    outputs: ['out'],
    generatedInputs: ['w', 'b'],
  });
  expect(
    nodePorts(
      { id: 'fresh', component: 'linear', attrs: { automatic_parameters: true } },
      registry,
      'inputs',
    ),
  ).toEqual(['x']);
  for (const attrs of [undefined, { automatic_parameters: false }])
    expect(nodePorts({ id: 'legacy', component: 'linear', attrs }, registry, 'inputs')).toEqual([
      'x',
      'w',
      'b',
    ]);
});

it('uses consistent binary-expression symbols for four arithmetic operations', () => {
  expect({ add: SYMBOLS.add, sub: SYMBOLS.sub, mul: SYMBOLS.mul, div: SYMBOLS.div }).toEqual({
    add: 'a+b',
    sub: 'a−b',
    mul: 'a×b',
    div: 'a÷b',
  });
});

it('size is available wherever reshape is available', () => {
  for (const level of gameLevels
    .filter((level) => level.kind !== 'observation')
    .filter((level) => level.available.includes('reshape')))
    expect(level.available).toContain('size');
  expect(SYMBOLS.size).toBe('|x|ₖ');
});

it('classifies the RNN window with the learning components', () => {
  expect(SYMBOLS.rnn_window).toBe('RNN×T');
  expect(SYMBOLS.transformer_block).toBe('Transformer');
  expect(categoryOf('rnn_cell')).toBe('learn');
  expect(categoryOf('rnn_window')).toBe('learn');
  expect(categoryOf('temporal_attention')).toBe('attn');
});

it('every part a player can place has a node symbol', () => {
  const names = new Set(
    gameLevels
      .filter((level) => level.kind !== 'observation')
      .flatMap((level) => [...(level.unlockComponents ?? []), ...level.available]),
  );
  expect([...names].filter((name) => !SYMBOLS[name])).toEqual([]);
});
