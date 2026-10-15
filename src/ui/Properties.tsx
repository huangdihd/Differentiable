import { useEffect, useId, useRef, useState } from 'react';
import type { GraphNode, Attributes } from '../engine/graph';
import { translatePort } from '../i18n';
import { shapeText } from './Canvas';

type Props = {
  node: GraphNode;
  connectedInputs?: readonly string[];
  t: (key: string) => string;
  onChange: (attrs: Attributes) => void;
};
export function defaults(component: string): Attributes {
  switch (component) {
    case 'const':
      return { value: 1 };
    case 'param':
      return { shape: [2, 2], init: 'normal', scale: 1 };
    case 'sum':
    case 'reduce_max':
      return { axes: [-1], keepdims: true };
    case 'size':
    case 'unsqueeze':
      return { axis: -1 };
    case 'reshape':
      return { shape: [-1] };
    case 'transpose':
      return { axes: [-2, -1] };
    case 'arange':
      return { n: 5 };
    default:
      return {};
  }
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
type Draft = Record<string, string>;
function draftFrom(attrs: Attributes): Draft {
  return Object.fromEntries(
    Object.entries(attrs).map(([key, value]) => [
      key,
      typeof value === 'string' ? value : (JSON.stringify(value) ?? ''),
    ]),
  );
}

/** Blank optional fields remove the attribute; lists accept JSON or comma-separated numbers. */
export function validateProperties(component: string, draft: Draft, base: Attributes = {}) {
  const attrs = { ...base };
  const errors: Record<string, string> = {};
  for (const key of fields[component] ?? []) {
    const text = (draft[key] ?? (key === 'keepdims' ? 'false' : '')).trim();
    if (key === 'label') {
      const label = [...text].slice(0, 16).join('');
      if (label) attrs.label = label;
      else delete attrs.label;
      continue;
    }
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
      const value: unknown = JSON.parse(list && !text.startsWith('[') ? `[${text}]` : text);
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
      } else if (key === 'keepdims') {
        if (typeof value !== 'boolean') throw Error();
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

export default function Properties({ node, connectedInputs = [], t, onChange }: Props) {
  const attrs = node.attrs ?? defaults(node.component);
  const source = JSON.stringify(attrs);
  const identity = `${node.id}:${node.component}`;
  const [draft, setDraft] = useState(() => draftFrom(attrs));
  const published = useRef<string | null>(null);
  const previousIdentity = useRef(identity);
  const id = useId();
  useEffect(() => {
    // Acknowledging our own edit must not normalize text or discard an unfinished field.
    if (previousIdentity.current !== identity || published.current !== source)
      setDraft(draftFrom(JSON.parse(source) as Attributes));
    previousIdentity.current = identity;
    published.current = null;
  }, [source, identity]);
  const keys = fields[node.component];
  if (!keys) return null;
  const { errors } = validateProperties(node.component, draft, attrs);
  const change = (key: string, value: string) => {
    const next = { ...draft, [key]: value };
    setDraft(next);
    const result = validateProperties(node.component, next, attrs);
    if (!Object.keys(result.errors).length) {
      published.current = JSON.stringify(result.attrs);
      onChange(result.attrs);
    }
  };
  return (
    <section className="section">
      <h3 className="label">{t('ui.properties')}</h3>
      {keys.map((key) => {
        const wired = connectedInputs.filter((port) =>
          node.component === 'reshape' && key === 'shape'
            ? /^d[0-3]$/.test(port)
            : node.component === 'arange' && key === 'n' && port === 'n',
        );
        const error = errors[key];
        const inputId = `${id}-${key}`;
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
            {key === 'keepdims' ? (
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
