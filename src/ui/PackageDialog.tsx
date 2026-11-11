// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Name a packaged component and its ports before it replaces the selected parts. */
import { useState } from 'react';
import type { GraphEndpoint, GraphJSON } from '../engine/graph';
import {
  MAX_COMPONENT_NAME,
  MAX_PORT_NAME,
  portNameError,
  type PackagePlan,
} from '../sandbox/components';
import { parameterName } from './parameter-label';
import { Dialog } from './Sandbox';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export interface PackageChoice {
  name: string;
  inputs: string[];
  outputs: string[];
}

export function PackageDialog({
  t,
  graph,
  plan,
  defaultName,
  shapeOf,
  onCancel,
  onConfirm,
}: {
  t: T;
  graph: GraphJSON;
  plan: PackagePlan;
  defaultName: string;
  /** Shape currently flowing out of an endpoint, when the canvas knows it. */
  shapeOf: (endpoint: GraphEndpoint) => readonly number[] | undefined;
  onCancel: () => void;
  onConfirm: (choice: PackageChoice) => void;
}) {
  const [name, setName] = useState(defaultName);
  const [inputs, setInputs] = useState(() => plan.inputs.map((p) => p.name));
  const [outputs, setOutputs] = useState(() => plan.outputs.map((p) => p.name));
  const label = (endpoint: GraphEndpoint) => {
    const node = graph.nodes.find((n) => n.id === endpoint.node);
    if (!node) return endpoint.node;
    const base =
      node.component === 'input' || node.component === 'output'
        ? String(node.attrs?.name ?? '')
        : (parameterName(node) ?? t(`components.${node.component}`));
    return endpoint.port === 'out' ? base : `${base}.${endpoint.port}`;
  };
  const shape = (endpoint: GraphEndpoint) => {
    const s = shapeOf(endpoint);
    return s ? (s.length ? `[${s.join('×')}]` : t('ui.scalar')) : undefined;
  };
  const errors = (names: string[], before: string[] = []) =>
    names.map((n, i) =>
      portNameError(
        n.trim(),
        [...before, ...names.slice(0, i)].map((x) => x.trim()),
      ),
    );
  // Inputs and outputs share one namespace so a port name never means two things.
  const inputErrors = errors(inputs),
    outputErrors = errors(outputs, inputs);
  const valid =
    name.trim().length > 0 && [...inputErrors, ...outputErrors].every((e) => e === undefined);
  const confirm = () =>
    valid &&
    onConfirm({
      name: name.trim(),
      inputs: inputs.map((n) => n.trim()),
      outputs: outputs.map((n) => n.trim()),
    });
  const rows = (
    side: 'inputs' | 'outputs',
    names: string[],
    set: (names: string[]) => void,
    problems: (string | undefined)[],
  ) => (
    <section className="pkg-side">
      <h3 className="sb-block-title">
        {t(`sandbox.components.${side}`)} <span className="pkg-count">{names.length}</span>
      </h3>
      {names.length === 0 ? (
        <p className="hint">{t('sandbox.components.noInputs')}</p>
      ) : (
        <ul className="pkg-ports">
          {plan[side].map((port, i) => {
            const shown = shape(port.source);
            return (
              <li key={i} className="pkg-port">
                <input
                  className="field pkg-port-name"
                  value={names[i]}
                  maxLength={MAX_PORT_NAME}
                  spellCheck={false}
                  aria-label={t('sandbox.components.portName', { n: i + 1 })}
                  aria-invalid={problems[i] !== undefined}
                  onChange={(e) => set(names.map((n, k) => (k === i ? e.target.value : n)))}
                  onKeyDown={(e) => e.key === 'Enter' && confirm()}
                />
                {shown && <span className="node-shape-tag pkg-shape">{shown}</span>}
                <span className="pkg-port-where">
                  {side === 'inputs'
                    ? t('sandbox.components.from', { source: label(port.source) })
                    : t('sandbox.components.to', {
                        source: label(port.source),
                        targets: port.targets.map(label).join(t('sandbox.components.listSep')),
                      })}
                </span>
                {problems[i] && <span className="pkg-port-error">{t(problems[i]!)}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
  return (
    <Dialog
      t={t}
      title={t('sandbox.components.packageTitle')}
      onClose={onCancel}
      wide
      footer={
        <>
          <button className="btn" onClick={onCancel}>
            {t('sandbox.cancel')}
          </button>
          <button className="btn btn-primary" disabled={!valid} onClick={confirm}>
            ⊞ {t('sandbox.components.packageConfirm')}
          </button>
        </>
      }
    >
      <p className="muted sb-dialog-lead">
        {t('sandbox.components.packageLead', { count: plan.nodes.length, n: plan.nodes.length })}
      </p>
      <label className="pkg-name">
        <span className="sb-block-title">{t('sandbox.components.name')}</span>
        <input
          className="field"
          autoFocus
          value={name}
          maxLength={MAX_COMPONENT_NAME}
          aria-invalid={!name.trim()}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && confirm()}
        />
      </label>
      <div className="pkg-io">
        {rows('inputs', inputs, setInputs, inputErrors)}
        <span className="pkg-arrow" aria-hidden="true">
          → ⊞ →
        </span>
        {rows('outputs', outputs, setOutputs, outputErrors)}
      </div>
      <p className="hint">{t('sandbox.components.packageHint')}</p>
    </Dialog>
  );
}
