// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ComponentRegistry, GraphEndpoint, GraphJSON, GraphNode } from '../engine/graph';
import { translatePort } from '../i18n';
import { broadcastDisplayShapes, shapeText, type NodeValue } from './Canvas';
import { formatValue } from './format-value';
import { nodePorts } from './graph-geometry';
import TensorGrid from './TensorGrid';

export type SelectedPortProbe =
  | { status: 'loading' | 'unavailable' }
  | { status: 'value'; value: NodeValue }
  | { status: 'error'; message: string };

export interface SelectedNodePort {
  key: string;
  direction: 'input' | 'output';
  port: string;
  endpoint?: GraphEndpoint;
  optional?: boolean;
}

/** Visible physical ports, plus the named external value represented by an Output node. */
export function selectedNodePorts(
  graph: GraphJSON,
  node: GraphNode,
  registry: ComponentRegistry,
): SelectedNodePort[] {
  const incoming = new Map(
    graph.edges.filter((edge) => edge.to.node === node.id).map((edge) => [edge.to.port, edge.from]),
  );
  const optional = new Set(registry.get(node.component)?.optionalInputs ?? []);
  const inputs = nodePorts(node, registry, 'inputs').map((port) => ({
    key: `input:${port}`,
    direction: 'input' as const,
    port,
    endpoint: incoming.get(port),
    optional: optional.has(port),
  }));
  const outputs: SelectedNodePort[] = nodePorts(node, registry, 'outputs').map((port) => ({
    key: `output:${port}`,
    direction: 'output' as const,
    port,
    endpoint: { node: node.id, port },
  }));
  if (node.component === 'output') {
    const port = String(node.attrs?.name ?? 'out');
    outputs.push({
      key: `output:${port}`,
      direction: 'output',
      port,
      endpoint: incoming.get('x'),
    });
  }
  return [...inputs, ...outputs];
}

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export default function SelectedNodeIO({
  t,
  locale,
  node,
  ports,
  probes,
  errors,
}: {
  t: T;
  locale: 'en' | 'zh-CN';
  node: GraphNode;
  ports: readonly SelectedNodePort[];
  probes: Readonly<Record<string, SelectedPortProbe>>;
  errors: readonly { node?: string; port?: string; message: string; soft?: boolean }[];
}) {
  const groups = (['input', 'output'] as const).map((direction) => ({
    direction,
    ports: ports.filter((port) => port.direction === direction),
  }));
  const shape = (key: string) => {
    const probe = probes[key];
    return probe?.status === 'value' ? probe.value.shape : undefined;
  };
  const output = groups.find((group) => group.direction === 'output')?.ports[0];
  const broadcast = broadcastDisplayShapes(
    node.component,
    shape('input:a'),
    shape('input:b'),
    output ? shape(output.key) : undefined,
  );
  return (
    <section className="section selected-io">
      <h3 className="label">{t('ui.nodeIO')}</h3>
      {broadcast && (
        <p className="selected-io-broadcast" title={t('ui.broadcastHelp')}>
          <span>{t('ui.broadcast')}</span>
          {shapeText(broadcast.a, t)} · {shapeText(broadcast.b, t)} →{' '}
          {shapeText(broadcast.output, t)}
        </p>
      )}
      {groups.map(({ direction, ports: group }) => (
        <div className="selected-io-group" key={direction}>
          <h4>{t(direction === 'input' ? 'ui.nodeInputs' : 'ui.nodeOutputs')}</h4>
          {group.length ? (
            group.map((entry) => {
              const diagnostic = errors.find(
                (error) =>
                  error.node === node.id &&
                  (error.port === entry.port || (!error.port && direction === 'output')),
              );
              const probe = probes[entry.key];
              const disconnected = !entry.endpoint;
              const value = probe?.status === 'value' ? probe.value : undefined;
              const message =
                (!disconnected || !diagnostic?.soft) && diagnostic?.message
                  ? diagnostic.message
                  : probe?.status === 'error'
                    ? probe.message
                    : '';
              const failed = !!message || value?.ok === false;
              return (
                <div
                  className={`selected-io-port${failed ? ' is-error' : ''}${disconnected ? ' is-disconnected' : ''}`}
                  key={entry.key}
                >
                  <div className="selected-io-head">
                    <span className="selected-io-name">
                      {translatePort(t, node.component, entry.port)}
                    </span>
                    <span className="selected-io-shape">
                      {value?.shape ? shapeText(value.shape, t) : '—'}
                    </span>
                  </div>
                  {message ? (
                    <p className="selected-io-state">{message}</p>
                  ) : disconnected ? (
                    <p className="selected-io-state">
                      {t(entry.optional ? 'ui.optionalDisconnected' : 'ui.disconnected')}
                    </p>
                  ) : probe?.status === 'loading' || !probe ? (
                    <p className="selected-io-state">{t('ui.loadingValue')}</p>
                  ) : probe.status === 'unavailable' ? (
                    <p className="selected-io-state">{t('ui.valueUnavailable')}</p>
                  ) : value?.shape?.length && value.values ? (
                    <TensorGrid
                      label={translatePort(t, node.component, entry.port)}
                      shape={value.shape}
                      values={value.values}
                      format={(number) => formatValue(locale, number)}
                    />
                  ) : (
                    <span className="selected-io-scalar mono" title={value?.title}>
                      {value?.values ? formatValue(locale, value.values[0]) : (value?.text ?? '—')}
                    </span>
                  )}
                </div>
              );
            })
          ) : (
            <p className="selected-io-empty">{t('ui.noPorts')}</p>
          )}
        </div>
      ))}
    </section>
  );
}
