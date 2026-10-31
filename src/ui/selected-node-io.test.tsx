// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { createDefaultRegistry, type GraphJSON } from '../engine/graph';
import { createTranslator } from '../i18n';
import SelectedNodeIO, { selectedNodePorts } from './SelectedNodeIO';

const registry = createDefaultRegistry()
  .register({
    name: 'multi_test',
    inputs: ['x', 'optional'],
    optionalInputs: ['optional'],
    outputs: ['left', 'right'],
  })
  .register({ name: 'div', inputs: ['a', 'b'], outputs: ['out'] });

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 'multi', component: 'multi_test' },
    { id: 'y', component: 'output', attrs: { name: 'prediction' } },
  ],
  edges: [
    { from: { node: 'x', port: 'out' }, to: { node: 'multi', port: 'x' } },
    { from: { node: 'multi', port: 'right' }, to: { node: 'y', port: 'x' } },
  ],
};

describe('selected node I/O', () => {
  it('lists each real input and output endpoint, including optional and multiple ports', () => {
    expect(selectedNodePorts(graph, graph.nodes[1], registry)).toEqual([
      {
        key: 'input:x',
        direction: 'input',
        port: 'x',
        endpoint: { node: 'x', port: 'out' },
        optional: false,
      },
      {
        key: 'input:optional',
        direction: 'input',
        port: 'optional',
        endpoint: undefined,
        optional: true,
      },
      {
        key: 'output:left',
        direction: 'output',
        port: 'left',
        endpoint: { node: 'multi', port: 'left' },
      },
      {
        key: 'output:right',
        direction: 'output',
        port: 'right',
        endpoint: { node: 'multi', port: 'right' },
      },
    ]);
  });

  it('represents external Input and Output nodes without inventing duplicate values', () => {
    expect(selectedNodePorts(graph, graph.nodes[0], registry)).toEqual([
      {
        key: 'output:out',
        direction: 'output',
        port: 'out',
        endpoint: { node: 'x', port: 'out' },
      },
    ]);
    expect(selectedNodePorts(graph, graph.nodes[2], registry)).toEqual([
      {
        key: 'input:x',
        direction: 'input',
        port: 'x',
        endpoint: { node: 'multi', port: 'right' },
        optional: false,
      },
      {
        key: 'output:prediction',
        direction: 'output',
        port: 'prediction',
        endpoint: { node: 'multi', port: 'right' },
      },
    ]);
  });

  it('renders semantic external output names in the selected locale', () => {
    const semanticGraph = structuredClone(graph);
    semanticGraph.nodes[2].attrs = { name: 'probabilities' };
    const node = semanticGraph.nodes[2];
    const ports = selectedNodePorts(semanticGraph, node, registry);
    for (const [locale, expected] of [
      ['en', 'Probabilities'],
      ['zh-CN', '概率'],
    ] as const) {
      const html = renderToStaticMarkup(
        <SelectedNodeIO
          t={createTranslator(locale)}
          locale={locale}
          node={node}
          ports={ports}
          probes={{
            'output:probabilities': {
              status: 'value',
              value: { text: '0.2, 0.8', title: '[2]', ok: true, shape: [2], values: [0.2, 0.8] },
            },
          }}
          errors={[]}
        />,
      );
      expect(html).toContain(expected);
      expect(html).not.toContain('>probabilities<');
    }
  });

  it('renders scalar, tensor, disconnected and error states with shapes', () => {
    const node = graph.nodes[1];
    const ports = selectedNodePorts(graph, node, registry);
    const html = renderToStaticMarkup(
      <SelectedNodeIO
        t={createTranslator('en')}
        locale="en"
        node={node}
        ports={ports}
        probes={{
          'input:x': {
            status: 'value',
            value: {
              text: '1, 2',
              title: '[2] 1, 2',
              ok: true,
              shape: [2],
              values: [1, 2],
            },
          },
          'output:left': {
            status: 'value',
            value: { text: '3', title: '[·] 3', ok: true, shape: [], values: [3] },
          },
          'output:right': { status: 'error', message: 'Bad output' },
        }}
        errors={[]}
      />,
    );

    expect(html).toContain('Selected component');
    expect(html).toContain('[2]');
    expect(html).toContain('Optional · not connected');
    expect(html).toContain('>3<');
    expect(html).toContain('Bad output');
    expect(html).toContain('tensor-grid');
  });

  it('explains division broadcasting with input and output shapes', () => {
    const division: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'a', component: 'input', attrs: { name: 'a' } },
        { id: 'b', component: 'input', attrs: { name: 'b' } },
        { id: 'division', component: 'div' },
      ],
      edges: [
        { from: { node: 'a', port: 'out' }, to: { node: 'division', port: 'a' } },
        { from: { node: 'b', port: 'out' }, to: { node: 'division', port: 'b' } },
      ],
    };
    const node = division.nodes[2];
    const ports = selectedNodePorts(division, node, registry);
    const html = renderToStaticMarkup(
      <SelectedNodeIO
        t={createTranslator('zh-CN')}
        locale="zh-CN"
        node={node}
        ports={ports}
        probes={{
          'input:a': {
            status: 'value',
            value: { text: '…', title: '[4×1]', ok: true, shape: [4, 1], values: [1, 2, 3, 4] },
          },
          'input:b': {
            status: 'value',
            value: { text: '…', title: '[3]', ok: true, shape: [3], values: [1, 2, 3] },
          },
          'output:out': {
            status: 'value',
            value: {
              text: '…',
              title: '[4×3]',
              ok: true,
              shape: [4, 3],
              values: Array(12).fill(1),
            },
          },
        }}
        errors={[]}
      />,
    );

    expect(html).toContain('selected-io-broadcast');
    expect(html).toContain('广播');
    expect(html).toContain('[4×1] · [3] → [4×3]');
  });

  it('keeps soft missing-wire notes quiet and marks nonfinite values as errors', () => {
    const node = graph.nodes[1];
    const html = renderToStaticMarkup(
      <SelectedNodeIO
        t={createTranslator('en')}
        locale="en"
        node={node}
        ports={[
          { key: 'input:x', direction: 'input', port: 'x' },
          {
            key: 'output:left',
            direction: 'output',
            port: 'left',
            endpoint: { node: 'multi', port: 'left' },
          },
        ]}
        probes={{
          'output:left': {
            status: 'value',
            value: { text: '∞', title: '[·] Infinity', ok: false, shape: [], values: [Infinity] },
          },
        }}
        errors={[{ node: 'multi', port: 'x', message: 'Missing input', soft: true }]}
      />,
    );

    expect(html).toContain('class="selected-io-port is-disconnected"');
    expect(html).not.toContain('Missing input');
    expect(html).toContain('Not connected');
    expect(html).toContain('class="selected-io-port is-error"');
  });
});
