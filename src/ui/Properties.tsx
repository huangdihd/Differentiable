// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useId, useRef, useState } from 'react';
import type {
  GraphNode,
  Attributes,
  ComponentDefinition,
  ComponentAttributeDefinition,
} from '../engine/graph';
import { translatePort, type TranslationParams } from '../i18n';
import { shapeText } from './Canvas';
import { parseRealNumber, parseRealValue } from './real-number';

type Props = {
  node: GraphNode;
  connectedInputs?: readonly string[];
  definition?: ComponentDefinition;
  t: (key: string, params?: TranslationParams) => string;
  onChange: (attrs: Attributes) => void;
};

/** Reshape exposes d0–d3; rank is attrs.shape.length (see graph-geometry nodePorts). */
const RESHAPE_MAX_RANK = 4;

function reshapeShapeOf(attrs: Attributes): number[] {
  const shape = attrs.shape;
  if (Array.isArray(shape) && shape.length >= 1 && shape.every((n) => Number.isSafeInteger(n)))
    return shape as number[];
  return [-1];
}
export function defaults(
  component: string,
  definition?: ComponentDefinition,
  overrides: Attributes = {},
): Attributes {
  let attrs: Attributes;
  switch (component) {
    case 'const':
      attrs = { value: 1 };
      break;
    case 'param':
      attrs = { shape: [2, 2], init: 'normal', scale: 1 };
      break;
    case 'sum':
    case 'reduce_max':
      attrs = { axes: [-1], keepdims: true };
      break;
    case 'size':
    case 'unsqueeze':
      attrs = { axis: -1 };
      break;
    case 'reshape':
      attrs = { shape: [-1] };
      break;
    case 'transpose':
      attrs = { axes: [-2, -1] };
      break;
    case 'arange':
      attrs = { n: 5 };
      break;
    default:
      attrs = {};
  }
  return { ...attrs, ...definition?.defaultAttributes, ...overrides };
}
const fields: Record<string, string[]> = {
  const: ['value', 'shape'],
  param: ['label', 'shape', 'init', 'scale', 'value'],
  sum: ['axes', 'keepdims'],
  reduce_max: ['axes', 'keepdims'],
  reshape: ['shape'],
  transpose: ['axes', 'perm'],
  unsqueeze: ['axis'],
  size: ['axis'],
  arange: ['n'],
};
const initializers = ['zeros', 'ones', 'normal', 'uniform', 'constant'];
const componentFields = (component: string, definition?: ComponentDefinition) => [
  ...(fields[component] ?? []),
  ...(definition?.attributes ?? [])
    .filter((attribute) => !attribute.hidden && !(fields[component] ?? []).includes(attribute.name))
    .map((attribute) => attribute.name),
];
const componentAttribute = (
  definition: ComponentDefinition | undefined,
  key: string,
): ComponentAttributeDefinition | undefined =>
  definition?.attributes?.find((attribute) => attribute.name === key);
type Draft = Record<string, string>;
const isScalarShape = (text: string, scalarLabel: string) =>
  /^\[\s*\]$/.test(text) ||
  text.toLocaleLowerCase() === 'scalar' ||
  text === '标量' ||
  text === scalarLabel;

function draftFrom(attrs: Attributes, scalarLabel: string): Draft {
  return Object.fromEntries(
    Object.entries(attrs).map(([key, value]) => [
      key,
      key === 'shape' && Array.isArray(value) && value.length === 0
        ? scalarLabel
        : typeof value === 'string'
          ? value
          : (JSON.stringify(value) ?? ''),
    ]),
  );
}

/** Blank optional fields remove attributes; real values accept fractions, integer lists stay JSON. */
export function validateProperties(
  component: string,
  draft: Draft,
  base: Attributes = {},
  scalarLabel = 'scalar',
  definition?: ComponentDefinition,
) {
  const attrs = { ...base };
  const errors: Record<string, string> = {};
  for (const key of componentFields(component, definition)) {
    const text = (draft[key] ?? (key === 'keepdims' ? 'false' : '')).trim();
    const scalarShape = key === 'shape' && isScalarShape(text, scalarLabel);
    if (key === 'label') {
      const label = [...text].slice(0, 16).join('');
      if (label) attrs.label = label;
      else delete attrs.label;
      continue;
    }
    const attribute = componentAttribute(definition, key);
    const optional =
      (component === 'const' && key === 'shape') ||
      (component === 'param' && ['init', 'scale', 'value'].includes(key)) ||
      key === 'axes' ||
      key === 'perm';
    if (!text && optional) {
      delete attrs[key];
      continue;
    }
    const list = ['shape', 'axes', 'perm'].includes(key);
    const error = list ? 'ui.invalidShape' : key === 'init' ? 'ui.invalidInit' : 'ui.invalidNumber';
    try {
      if (!text) throw Error();
      if (key === 'init') {
        if (!initializers.includes(text)) throw Error();
        attrs[key] = text;
        continue;
      }
      const value: unknown =
        key === 'value' && component === 'const'
          ? parseRealValue(text)
          : key === 'value' || key === 'scale'
            ? parseRealNumber(text)
            : JSON.parse(scalarShape ? '[]' : list && !text.startsWith('[') ? `[${text}]` : text);
      const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
      if (list) {
        if (!Array.isArray(value) || !value.every(Number.isSafeInteger)) throw Error();
        const numbers = value as number[];
        if (
          key === 'shape' &&
          (numbers.some((n) => n < (component === 'reshape' ? -1 : 0)) ||
            numbers.filter((n) => n === -1).length > 1 ||
            (numbers.includes(-1) && numbers.includes(0)) ||
            !Number.isSafeInteger(numbers.reduce((size, n) => size * Math.max(1, n), 1)))
        )
          throw Error();
        if (
          key === 'axes' &&
          (new Set(numbers).size !== numbers.length ||
            (component === 'transpose' && numbers.length !== 2))
        )
          throw Error();
        if (
          key === 'perm' &&
          (new Set(numbers).size !== numbers.length ||
            numbers.some((n) => n < 0 || n >= numbers.length))
        )
          throw Error();
      } else if (key === 'keepdims' || attribute?.kind === 'boolean') {
        if (typeof value !== 'boolean') throw Error();
      } else if (attribute?.kind === 'integer') {
        if (
          !finite(value) ||
          !Number.isSafeInteger(value) ||
          (attribute.min !== undefined && value < attribute.min) ||
          (attribute.max !== undefined && value > attribute.max) ||
          (attribute.multipleOf !== undefined && value % attribute.multipleOf !== 0)
        )
          throw Error();
      } else if (key === 'value' && component === 'const') {
        if (!finite(value) && !(Array.isArray(value) && value.every(finite))) throw Error();
      } else if (
        !finite(value) ||
        (key === 'n' && (!Number.isSafeInteger(value) || value < 0)) ||
        (key === 'axis' && !Number.isSafeInteger(value)) ||
        (key === 'scale' && value < 0)
      )
        throw Error();
      attrs[key] = value;
    } catch {
      errors[key] = error;
    }
  }
  if (component === 'param' && attrs.init === 'constant' && attrs.value === undefined)
    errors.value = 'ui.invalidNumber';
  if (
    component === 'const' &&
    Array.isArray(attrs.value) &&
    !errors.shape &&
    (!Array.isArray(attrs.shape) ||
      attrs.value.length !== attrs.shape.reduce((a: number, b: number) => a * b, 1))
  )
    errors.shape = 'ui.invalidShape';
  return { attrs, errors };
}

export default function Properties({ node, connectedInputs = [], definition, t, onChange }: Props) {
  const declaredDefaults = Object.fromEntries(
    (definition?.attributes ?? []).map((attribute) => [attribute.name, attribute.default]),
  );
  // Fresh nodes already persist defaultAttributes. A legacy node without attributes keeps the
  // compiler's conservative defaults instead of silently replacing its explicit parameter ports.
  const attrs = node.attrs
    ? { ...declaredDefaults, ...node.attrs }
    : definition?.attributes?.length
      ? declaredDefaults
      : defaults(node.component, definition);
  const source = JSON.stringify(attrs);
  const identity = `${node.id}:${node.component}`;
  const scalarLabel = t('ui.scalar');
  const [draft, setDraft] = useState(() => draftFrom(attrs, scalarLabel));
  const [dimTexts, setDimTexts] = useState(() => reshapeShapeOf(attrs).map(String));
  const published = useRef<string | null>(null);
  const previousIdentity = useRef(identity);
  const previousScalarLabel = useRef(scalarLabel);
  const id = useId();
  useEffect(() => {
    // Acknowledging our own edit must not normalize text or discard an unfinished field.
    if (
      previousIdentity.current !== identity ||
      previousScalarLabel.current !== scalarLabel ||
      published.current !== source
    ) {
      const nextAttrs = JSON.parse(source) as Attributes;
      setDraft(draftFrom(nextAttrs, scalarLabel));
      setDimTexts(reshapeShapeOf(nextAttrs).map(String));
    }
    previousIdentity.current = identity;
    previousScalarLabel.current = scalarLabel;
    published.current = null;
  }, [source, identity, scalarLabel]);
  const keys = componentFields(node.component, definition);
  if (!keys.length) return null;
  const { errors } = validateProperties(node.component, draft, attrs, scalarLabel, definition);
  const change = (key: string, value: string) => {
    const shown = key === 'shape' && isScalarShape(value.trim(), scalarLabel) ? scalarLabel : value;
    const next = { ...draft, [key]: shown };
    setDraft(next);
    const result = validateProperties(node.component, next, attrs, scalarLabel, definition);
    if (!Object.keys(result.errors).length) {
      published.current = JSON.stringify(result.attrs);
      onChange(result.attrs);
    }
  };
  const reshapeShape = reshapeShapeOf(attrs);
  const publishReshapeShape = (next: number[]) => {
    setDimTexts(next.map(String));
    change('shape', JSON.stringify(next));
  };
  const reshapeDimValues = (texts: string[]): number[] | null => {
    const built: number[] = [];
    for (let i = 0; i < texts.length; i++) {
      if (connectedInputs.includes(`d${i}`)) {
        built.push(reshapeShape[i] ?? 1);
        continue;
      }
      const parsed = Number(texts[i].trim());
      if (!Number.isSafeInteger(parsed)) return null;
      built.push(parsed);
    }
    return built;
  };
  return (
    <section className="section">
      <h3 className="label">{t('ui.properties')}</h3>
      {keys.map((key) => {
        const attribute = componentAttribute(definition, key);
        const error = errors[key];
        const inputId = `${id}-${key}`;
        if (node.component === 'reshape' && key === 'shape') {
          const rank = dimTexts.length;
          const lastWired = connectedInputs.includes(`d${rank - 1}`);
          return (
            <div key={key} className="reshape-dims">
              <div className="label">{t('ui.shape')}</div>
              {dimTexts.map((text, index) => {
                const port = `d${index}`;
                const wired = connectedInputs.includes(port);
                const dimId = `${inputId}-${index}`;
                return (
                  <div key={port} className="reshape-dim-row">
                    <label htmlFor={dimId}>
                      {t('ui.shapeDim', { n: index })}
                      {wired ? ` ← ${translatePort(t, node.component, port)}` : ''}
                    </label>
                    <input
                      id={dimId}
                      className="field"
                      type="text"
                      disabled={wired}
                      aria-invalid={!!error}
                      aria-describedby={error ? `${inputId}-error` : undefined}
                      value={wired ? t('ui.shapeByWire') : text}
                      onChange={(e) => {
                        const nextTexts = dimTexts.map((value, i) =>
                          i === index ? e.target.value : value,
                        );
                        setDimTexts(nextTexts);
                        const built = reshapeDimValues(nextTexts);
                        change(
                          'shape',
                          built
                            ? JSON.stringify(built)
                            : `[${nextTexts.map((v) => v.trim()).join(',')}]`,
                        );
                      }}
                    />
                  </div>
                );
              })}
              <div className="reshape-dim-actions btn-row">
                <button
                  type="button"
                  className="btn"
                  disabled={rank >= RESHAPE_MAX_RANK}
                  onClick={() => {
                    if (rank >= RESHAPE_MAX_RANK) return;
                    const current = reshapeDimValues(dimTexts) ?? reshapeShape.slice(0, rank);
                    const next = [...current, current.includes(-1) ? 1 : -1];
                    publishReshapeShape(next);
                  }}
                >
                  {t('ui.shapeAddDim')}
                </button>
                <button
                  type="button"
                  className="btn"
                  disabled={rank <= 1 || lastWired}
                  onClick={() => {
                    if (rank <= 1 || lastWired) return;
                    const current = reshapeDimValues(dimTexts) ?? reshapeShape.slice(0, rank);
                    publishReshapeShape(current.slice(0, -1));
                  }}
                >
                  {t('ui.shapeRemoveDim')}
                </button>
              </div>
              {error && (
                <p id={`${inputId}-error`} className="err">
                  {t(error)}
                </p>
              )}
            </div>
          );
        }
        const wired = connectedInputs.filter(
          (port) => node.component === 'arange' && key === 'n' && port === 'n',
        );
        const accessibility = {
          id: inputId,
          disabled: wired.length > 0,
          'aria-invalid': !!error,
          'aria-describedby': error ? `${inputId}-error` : undefined,
        };
        return (
          <div key={key}>
            <label htmlFor={inputId}>
              {t(key === 'label' ? 'ui.name' : `ui.${key}`)}
              {wired.map((port) => ` ← ${translatePort(t, node.component, port)}`).join(', ')}
            </label>
            {key === 'automatic_parameters' && (
              <p className="properties-help">{t('ui.automatic_parameters_help')}</p>
            )}
            {key === 'keepdims' || attribute?.kind === 'boolean' ? (
              <input
                {...accessibility}
                type="checkbox"
                checked={draft[key] === 'true'}
                onChange={(e) => change(key, String(e.target.checked))}
              />
            ) : key === 'init' ? (
              <select
                {...accessibility}
                className="field"
                value={draft[key] ?? 'zeros'}
                onChange={(e) => change(key, e.target.value)}
              >
                {initializers.map((init) => (
                  <option key={init} value={init}>
                    {t(`ui.${init}`)}
                  </option>
                ))}
              </select>
            ) : (
              <input
                {...accessibility}
                className="field"
                type="text"
                placeholder={
                  key === 'label' ? `θ ${shapeText((attrs.shape as number[]) ?? [], t)}` : undefined
                }
                title={key === 'label' ? t('ui.parameterNameHelp') : undefined}
                value={draft[key] ?? ''}
                onChange={(e) =>
                  change(
                    key,
                    key === 'label'
                      ? [...e.target.value.trimStart()].slice(0, 16).join('')
                      : e.target.value,
                  )
                }
              />
            )}
            {error && (
              <p id={`${inputId}-error`} className="err">
                {t(error)}
              </p>
            )}
          </div>
        );
      })}
    </section>
  );
}
