// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useState, type CSSProperties } from 'react';

export type TensorTransformKind = 'softmax' | 'layernorm' | 'mse' | 'cross_entropy';
type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export interface TransformStage {
  key: string;
  values: number[];
}

export interface TransformExplanation {
  stages: TransformStage[];
  stats: { key: string; value: number }[];
}

/** The exact last-axis arithmetic taught by the Softmax, LayerNorm, MSE, and CE lessons. */
export function explainTensorTransform(
  kind: TensorTransformKind,
  row: readonly number[],
  target: readonly number[] = [],
): TransformExplanation {
  if (!row.length) return { stages: [], stats: [] };
  if (kind === 'mse') {
    if (target.length !== row.length) return { stages: [], stats: [] };
    const errors = row.map((value, index) => value - target[index]);
    const squares = errors.map((value) => value * value);
    const mean = squares.reduce((sum, value) => sum + value, 0) / squares.length;
    return {
      stages: [
        { key: 'prediction', values: [...row] },
        { key: 'target', values: [...target] },
        { key: 'errors', values: errors },
        { key: 'squares', values: squares },
        { key: 'output', values: [mean] },
      ],
      stats: [{ key: 'mean', value: mean }],
    };
  }
  if (kind === 'cross_entropy') {
    if (target.length !== row.length) return { stages: [], stats: [] };
    const logs = row.map(Math.log);
    const selected = logs.map((value, index) => value * target[index]);
    const rowLoss = -selected.reduce((sum, value) => sum + value, 0);
    return {
      stages: [
        { key: 'probabilities', values: [...row] },
        { key: 'logs', values: logs },
        { key: 'target', values: [...target] },
        { key: 'selected', values: selected },
        { key: 'rowLoss', values: [rowLoss] },
      ],
      stats: [{ key: 'rowLoss', value: rowLoss }],
    };
  }
  if (kind === 'softmax') {
    const maximum = Math.max(...row);
    const shifted = row.map((value) => value - maximum);
    const weights = shifted.map(Math.exp);
    const total = weights.reduce((sum, value) => sum + value, 0);
    return {
      stages: [
        { key: 'input', values: [...row] },
        { key: 'shifted', values: shifted },
        { key: 'weights', values: weights },
        { key: 'output', values: weights.map((value) => value / total) },
      ],
      stats: [
        { key: 'maximum', value: maximum },
        { key: 'total', value: total },
      ],
    };
  }
  const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
  const centered = row.map((value) => value - mean);
  const squares = centered.map((value) => value * value);
  const variance = squares.reduce((sum, value) => sum + value, 0) / row.length;
  const scale = Math.sqrt(variance + 1e-5);
  return {
    stages: [
      { key: 'input', values: [...row] },
      { key: 'centered', values: centered },
      { key: 'squares', values: squares },
      { key: 'output', values: centered.map((value) => value / scale) },
    ],
    stats: [
      { key: 'mean', value: mean },
      { key: 'variance', value: variance },
      { key: 'scale', value: scale },
    ],
  };
}

export function tensorTransformKind(levelId: string): TensorTransformKind | undefined {
  if (levelId === '2-7') return 'softmax';
  if (levelId === '2-8') return 'layernorm';
  if (levelId === '3-3') return 'mse';
  if (levelId === '3-5') return 'cross_entropy';
  return undefined;
}

/** A row-by-row visual trace; changing the current case or shape immediately changes every stage. */
export default function TensorTransformView({
  kind,
  values,
  targets,
  shape,
  t,
  format,
}: {
  kind: TensorTransformKind;
  values: readonly number[];
  targets?: readonly number[];
  shape: readonly number[];
  t: T;
  format: (value: number) => string;
}) {
  const width = shape.at(-1) ?? 0;
  const rows = width > 0 ? Math.floor(values.length / width) : 0;
  const [row, setRow] = useState(0);
  const shapeKey = shape.join('×');
  useEffect(() => setRow(0), [kind, shapeKey]);
  if (!width || !rows || values.length !== rows * width) return null;
  const selected = Math.min(row, rows - 1);
  const explanation = explainTensorTransform(
    kind,
    values.slice(selected * width, (selected + 1) * width),
    targets?.slice(selected * width, (selected + 1) * width),
  );
  if (!explanation.stages.length) return null;
  const stats = [...explanation.stats];
  if (kind === 'cross_entropy' && targets?.length === values.length) {
    const losses = Array.from({ length: rows }, (_, index) =>
      explainTensorTransform(
        kind,
        values.slice(index * width, (index + 1) * width),
        targets.slice(index * width, (index + 1) * width),
      ).stats.find((stat) => stat.key === 'rowLoss'),
    );
    if (losses.every((loss) => loss !== undefined))
      stats.push({
        key: 'batchMean',
        value: losses.reduce((sum, loss) => sum + loss!.value, 0) / rows,
      });
  }
  return (
    <section className="tensor-transform" aria-label={t(`visual.${kind}.title`)}>
      <div className="tensor-transform-head">
        <div>
          <span className="label">{t(`visual.${kind}.title`)}</span>
          <p>{t(`visual.${kind}.help`)}</p>
        </div>
        <span className="shape">[{shapeKey}]</span>
      </div>
      {rows > 1 && (
        <div className="tensor-transform-picker">
          <button
            type="button"
            aria-label={t('visual.previousRow')}
            disabled={selected === 0}
            onClick={() => setRow((value) => Math.max(0, value - 1))}
          >
            ←
          </button>
          <span>{t('visual.row', { current: selected + 1, total: rows })}</span>
          <button
            type="button"
            aria-label={t('visual.nextRow')}
            disabled={selected === rows - 1}
            onClick={() => setRow((value) => Math.min(rows - 1, value + 1))}
          >
            →
          </button>
        </div>
      )}
      <div className="tensor-transform-stages">
        {explanation.stages.map((stage, index) => (
          <div className="tensor-transform-stage" key={stage.key}>
            {index > 0 && <span className="tensor-transform-arrow">↓</span>}
            <span className="tensor-transform-label">{t(`visual.${kind}.${stage.key}`)}</span>
            <ValueBars values={stage.values} format={format} />
          </div>
        ))}
      </div>
      <dl className="tensor-transform-stats">
        {stats.map((stat) => (
          <div key={stat.key}>
            <dt>{t(`visual.${kind}.${stat.key}`)}</dt>
            <dd>{format(stat.value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ValueBars({
  values,
  format,
}: {
  values: readonly number[];
  format: (value: number) => string;
}) {
  const extent = Math.max(...values.map(Math.abs), Number.EPSILON);
  const signed = values.some((value) => value < 0);
  return (
    <div
      className={`transform-values ${signed ? 'is-signed' : ''}`}
      style={{ '--columns': values.length } as CSSProperties}
    >
      {values.map((value, index) => {
        const magnitude = Math.max(2, (Math.abs(value) / extent) * (signed ? 45 : 90));
        return (
          <div className="transform-value" key={index} title={String(value)}>
            <span className="transform-bar-track">
              <i
                className={value < 0 ? 'is-negative' : ''}
                style={{ '--magnitude': `${magnitude}%` } as CSSProperties}
              />
            </span>
            <span>{format(value)}</span>
          </div>
        );
      })}
    </div>
  );
}
