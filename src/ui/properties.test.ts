// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import type { ComponentDefinition } from '../engine/graph';
import { createTranslator } from '../i18n';
import Properties, { defaults, validateProperties } from './Properties';

const configurableLinear: ComponentDefinition = {
  name: 'linear',
  inputs: ['x', 'w', 'b'],
  outputs: ['out'],
  attributes: [
    { name: 'automatic_parameters', kind: 'boolean', default: false, hidden: true },
    { name: 'in_features', kind: 'integer', default: 2, min: 1 },
    { name: 'out_features', kind: 'integer', default: 2, min: 1 },
    { name: 'bias', kind: 'boolean', default: true },
  ],
  defaultAttributes: {
    automatic_parameters: true,
    in_features: 2,
    out_features: 2,
    bias: true,
  },
};
const configurableTransformer: ComponentDefinition = {
  name: 'transformer_block',
  inputs: ['x', 'wq', 'wk', 'wv', 'wo', 'w1', 'b1', 'w2', 'b2'],
  outputs: ['out'],
  attributes: [
    { name: 'automatic_parameters', kind: 'boolean', default: false, hidden: true },
    { name: 'features', kind: 'integer', default: 8, min: 1, multipleOf: 2 },
    { name: 'hidden_features', kind: 'integer', default: 16, min: 1 },
  ],
  defaultAttributes: { automatic_parameters: true, features: 8, hidden_features: 16 },
};

describe('property field validation', () => {
  it('preserves component defaults', () => {
    expect(defaults('sum')).toEqual({ axes: [-1], keepdims: true });
    expect(defaults('reduce_max')).toEqual({ axes: [-1], keepdims: true });
    expect(defaults('unsqueeze')).toEqual({ axis: -1 });
    expect(defaults('size')).toEqual({ axis: -1 });
    expect(defaults('transpose')).toEqual({ axes: [-2, -1] });
    expect(defaults('param')).toEqual({ shape: [2, 2], init: 'normal', scale: 1 });
    expect(defaults('linear', configurableLinear)).toEqual({
      automatic_parameters: true,
      in_features: 2,
      out_features: 2,
      bias: true,
    });
    expect(defaults('linear', configurableLinear, { in_features: 1, out_features: 1 })).toEqual({
      automatic_parameters: true,
      in_features: 1,
      out_features: 1,
      bias: true,
    });
  });
  it('validates reusable component configuration fields', () => {
    expect(
      validateProperties(
        'linear',
        {
          automatic_parameters: 'true',
          in_features: '3',
          out_features: '5',
          bias: 'false',
        },
        defaults('linear', configurableLinear),
        'scalar',
        configurableLinear,
      ),
    ).toEqual({
      attrs: { automatic_parameters: true, in_features: 3, out_features: 5, bias: false },
      errors: {},
    });
    for (const value of ['0', '-1', '1.5', ''])
      expect(
        validateProperties(
          'linear',
          { in_features: value, out_features: '2', bias: 'true' },
          {},
          'scalar',
          configurableLinear,
        ).errors.in_features,
      ).toBe('ui.invalidNumber');
    const html = renderToStaticMarkup(
      createElement(Properties, {
        node: { id: 'linear', component: 'linear' },
        definition: configurableLinear,
        t: createTranslator('en'),
        onChange: vi.fn(),
      }),
    );
    expect(html).toContain('Input features');
    expect(html).toContain('Output features');
    expect(html).toContain('Use bias');
    expect(html).not.toContain('automatic_parameters');
  });
  it('enforces reusable integer divisibility constraints', () => {
    expect(
      validateProperties(
        'transformer_block',
        { features: '8', hidden_features: '16' },
        {},
        'scalar',
        configurableTransformer,
      ).errors,
    ).toEqual({});
    expect(
      validateProperties(
        'transformer_block',
        { features: '7', hidden_features: '16' },
        {},
        'scalar',
        configurableTransformer,
      ).errors.features,
    ).toBe('ui.invalidNumber');
  });
  it('accepts negative integer unsqueeze axes but rejects incomplete or fractional axes', () => {
    for (const axis of ['-1', '-3', '0', '2']) {
      const result = validateProperties('unsqueeze', { axis });
      expect(result.errors).toEqual({});
      expect(result.attrs.axis).toBe(Number(axis));
    }
    for (const axis of ['', '-', '1.5', 'null', '1e999', '9007199254740992'])
      expect(validateProperties('unsqueeze', { axis }).errors.axis).toBe('ui.invalidNumber');
  });
  it('accepts integer size axes and rejects incomplete or fractional axes', () => {
    for (const axis of ['-1', '-3', '0', '2']) {
      expect(validateProperties('size', { axis })).toEqual({
        attrs: { axis: Number(axis) },
        errors: {},
      });
    }
    for (const axis of ['', '-', '1.5', 'null', '9007199254740992'])
      expect(validateProperties('size', { axis }).errors.axis).toBe('ui.invalidNumber');
  });
  it('supports scalar, vector and broadcast constants', () => {
    expect(validateProperties('const', { value: '2' }).errors).toEqual({});
    expect(validateProperties('const', { value: '[1,2]', shape: '2' }).errors).toEqual({});
    expect(validateProperties('const', { value: '2', shape: '2,3' }).errors).toEqual({});
    expect(validateProperties('const', { value: '[1,2]', shape: '3' }).errors.shape).toBe(
      'ui.invalidShape',
    );
  });
  it('maps localized scalar shape names to the rank-zero storage shape', () => {
    expect(validateProperties('param', { shape: 'scalar' }).attrs.shape).toEqual([]);
    expect(validateProperties('param', { shape: '标量' }, {}, '标量').attrs.shape).toEqual([]);
    expect(validateProperties('sum', { axes: 'scalar', keepdims: 'false' }).errors.axes).toBe(
      'ui.invalidShape',
    );
  });
  it('renders rank-zero shapes as localized words instead of empty brackets', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const t = createTranslator(locale);
      const html = renderToStaticMarkup(
        createElement(Properties, {
          node: { id: 'constant', component: 'const', attrs: { value: 1, shape: [] } },
          t,
          onChange: vi.fn(),
        }),
      );
      expect(html).toContain(`value="${t('ui.scalar')}"`);
      expect(html).not.toContain('value="[]"');
    }
  });
  it('requires finite numbers and complete input', () => {
    for (const value of ['', '-', '1e', '1e999', 'null'])
      expect(validateProperties('const', { value }).errors.value).toBe('ui.invalidNumber');
    for (const n of ['-1', '1.5', ''])
      expect(validateProperties('arange', { n }).errors.n).toBe('ui.invalidNumber');
  });
  it('matches initializer and scale constraints', () => {
    for (const init of ['zeros', 'ones', 'normal', 'uniform', 'constant'])
      expect(
        validateProperties('param', { shape: '2,2', init, scale: '0', value: '3' }).errors,
      ).toEqual({});
    expect(validateProperties('param', { shape: '[]', init: 'constant' }).errors.value).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('param', { shape: '[]', scale: '-1' }).errors.scale).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('param', { shape: '[]', init: 'unknown' }).errors.init).toBe(
      'ui.invalidInit',
    );
  });
  it('validates reshape, axes and permutations without knowing input rank', () => {
    expect(validateProperties('reshape', { shape: '-1,2' }).errors).toEqual({});
    for (const shape of ['-1,-1', '-2', '-1,0', '2,', '9007199254740991,2'])
      expect(validateProperties('reshape', { shape }).errors.shape).toBe('ui.invalidShape');
    expect(validateProperties('sum', { axes: '-1', keepdims: 'true' }).errors).toEqual({});
    expect(validateProperties('sum', { axes: '1,1' }).errors.axes).toBe('ui.invalidShape');
    expect(validateProperties('transpose', { perm: '1,0' }).errors).toEqual({});
    expect(validateProperties('transpose', { perm: '2,0' }).errors.perm).toBe('ui.invalidShape');
  });
  it('renders reshape shape as one row per dimension', () => {
    const t = createTranslator('en');
    const html = renderToStaticMarkup(
      createElement(Properties, {
        node: { id: 'flatten', component: 'reshape', attrs: { shape: [1, -1] } },
        connectedInputs: ['d0'],
        t,
        onChange: vi.fn(),
      }),
    );
    expect(html).toContain('Dimension 0');
    expect(html).toContain('Set by wire');
    expect(html).toContain('Target dimension 0');
    expect(html).toContain('Dimension 1');
    expect(html).toContain('value="-1"');
    expect(html).not.toContain('value="[1,-1]"');
    expect(html).toContain('Add dimension');
    expect(html).toContain('Remove last dimension');
    const zh = renderToStaticMarkup(
      createElement(Properties, {
        node: { id: 'flatten', component: 'reshape', attrs: { shape: [1, -1] } },
        connectedInputs: ['d0'],
        t: createTranslator('zh-CN'),
        onChange: vi.fn(),
      }),
    );
    expect(zh).toContain('由接线决定');
    expect(zh).toContain('维度 0');
    expect(zh).toContain('维度 1');
  });
  it('validates transpose pairs and preserves legacy permutation edits', () => {
    expect(validateProperties('transpose', { axes: '-2,-1' })).toEqual({
      attrs: { axes: [-2, -1] },
      errors: {},
    });
    for (const axes of ['[]', '1', '0,1,2', '1,1', '0,1.5', 'null'])
      expect(validateProperties('transpose', { axes }).errors.axes).toBe('ui.invalidShape');
    expect(
      validateProperties('transpose', { perm: '1,2,0' }, { perm: [2, 0, 1], custom: true }),
    ).toEqual({ attrs: { perm: [1, 2, 0], custom: true }, errors: {} });
    expect(validateProperties('transpose', { axes: '-2,-1', perm: '1,2,0' }).attrs).toEqual({
      axes: [-2, -1],
      perm: [1, 2, 0],
    });
  });
  it('removes blank optional attributes and preserves unrelated attributes', () => {
    expect(
      validateProperties('const', { value: '1', shape: '' }, { shape: [2], custom: true }).attrs,
    ).toEqual({ value: 1, custom: true });
    expect(validateProperties('sum', {}).attrs).toEqual({ keepdims: false });
  });
});
