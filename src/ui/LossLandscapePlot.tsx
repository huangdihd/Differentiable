// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useId, useRef, type PointerEvent } from 'react';
import type { ExpectedValue } from './expected-output';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;
type Mode = 'loss' | 'descent';

export interface LossLandscapeModel {
  mode: Mode;
  input: 'p' | 'w';
  current: number;
  target: number;
  prediction: number;
  loss: number;
  optimum?: number;
  gradient?: number;
  next?: number;
  domain: readonly [number, number];
  curve: readonly (readonly [number, number])[];
}

/** Keep the plotted squared-loss target fixed while translating a picked position into inputs. */
export function lossLandscapeInputChanges(
  model: LossLandscapeModel,
  inputs: Readonly<Record<string, readonly number[]>>,
  value: number,
): Readonly<Record<string, number>> {
  const bounded = Number(Math.min(model.domain[1], Math.max(model.domain[0], value)).toFixed(2));
  if (model.mode === 'loss') return { p: bounded };
  const x = inputs.x?.[0] ?? 0;
  return { w: bounded, '∂L/∂p': 2 * (bounded * x - model.target) };
}

/** Analytic teaching model shared by the SVG and tests; this is not a second level evaluator. */
export function lossLandscapeModel(
  levelId: string,
  inputs: Readonly<Record<string, readonly number[]>>,
  samples = 81,
): LossLandscapeModel | undefined {
  const at = (name: string) => inputs[name]?.[0];
  const mode: Mode | undefined =
    levelId === 'loss-intro' ? 'loss' : levelId === '3-1' ? 'descent' : undefined;
  if (!mode) return undefined;
  const input = mode === 'loss' ? 'p' : 'w';
  const current = at(input);
  if (!Number.isFinite(current)) return undefined;
  const x = mode === 'descent' ? at('x') : 1;
  if (!Number.isFinite(x)) return undefined;
  const prediction = mode === 'loss' ? current : current * x;
  const suppliedGradient = mode === 'descent' ? at('∂L/∂p') : undefined;
  const target =
    mode === 'loss'
      ? at('t')
      : suppliedGradient !== undefined
        ? prediction - suppliedGradient / 2
        : undefined;
  if (typeof target !== 'number' || !Number.isFinite(target)) return undefined;
  const loss = (prediction - target) ** 2;
  const optimum = mode === 'loss' ? target : Math.abs(x) > 1e-8 ? target / x : undefined;
  const rawGradient = mode === 'descent' ? (suppliedGradient as number) * (x as number) : undefined;
  const gradient = Object.is(rawGradient, -0) ? 0 : rawGradient;
  const next = gradient === undefined ? undefined : current - 0.1 * gradient;
  const anchors = [current, optimum, next].filter(
    (value): value is number => value !== undefined && Number.isFinite(value),
  );
  const lo = Math.min(...anchors),
    hi = Math.max(...anchors);
  const margin = Math.max(2, (hi - lo) * 0.25);
  const domain = [lo - margin, hi + margin] as const;
  const curve = Array.from({ length: samples }, (_, index) => {
    const value = domain[0] + ((domain[1] - domain[0]) * index) / (samples - 1);
    const estimate = mode === 'loss' ? value : value * x;
    return [value, (estimate - target) ** 2] as const;
  });
  return {
    mode,
    input,
    current,
    target,
    prediction,
    loss,
    optimum,
    gradient,
    next,
    domain,
    curve,
  };
}

const W = 260,
  H = 174,
  PAD_X = 17,
  PAD_TOP = 12,
  PAD_BOTTOM = 21;

/** Interactive loss bowl for the loss introduction and the first gradient-descent step. */
export default function LossLandscapePlot({
  t,
  levelId,
  inputs,
  actual,
  onPick,
}: {
  t: T;
  levelId: string;
  inputs: Readonly<Record<string, readonly number[]>>;
  actual?: ExpectedValue;
  onPick: (changes: Readonly<Record<string, number>>) => void;
}) {
  const marker = useId().replace(/:/g, '');
  const dragging = useRef<number | undefined>(undefined);
  const model = lossLandscapeModel(levelId, inputs);
  if (!model) return null;
  const [x0, x1] = model.domain;
  const rawActual = actual?.values[0];
  const actualValue = rawActual !== undefined && Number.isFinite(rawActual) ? rawActual : undefined;
  const expected = model.mode === 'loss' ? model.loss : model.next;
  const matches =
    expected !== undefined &&
    actualValue !== undefined &&
    Math.abs(actualValue - expected) <= 1e-5 + 1e-4 * Math.abs(expected);
  const actualLoss =
    model.mode === 'loss'
      ? actualValue
      : actualValue !== undefined
        ? (actualValue * (inputs.x?.[0] ?? 0) - model.target) ** 2
        : undefined;
  const yMax = Math.max(
    1,
    ...model.curve.map((point) => point[1]),
    ...(actualLoss !== undefined && Number.isFinite(actualLoss) ? [actualLoss] : []),
  );
  const sx = (value: number) => PAD_X + ((value - x0) / (x1 - x0)) * (W - PAD_X * 2);
  const sy = (value: number) =>
    H - PAD_BOTTOM - (Math.min(yMax, Math.max(0, value)) / yMax) * (H - PAD_TOP - PAD_BOTTOM);
  const point = (value: number) => {
    const prediction = model.mode === 'loss' ? value : value * (inputs.x?.[0] ?? 0);
    return [sx(value), sy((prediction - model.target) ** 2)] as const;
  };
  const currentPoint = point(model.current);
  const optimumPoint = model.optimum === undefined ? undefined : point(model.optimum);
  const nextPoint = model.next === undefined ? undefined : point(model.next);
  const gradientPoint =
    model.gradient === undefined || Math.abs(model.gradient) < 1e-8
      ? undefined
      : point(
          model.current +
            Math.sign(model.gradient) * Math.min((x1 - x0) * 0.16, Math.abs(model.gradient) * 0.1),
        );
  const curve = model.curve.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ');
  const choose = (value: number) => onPick(lossLandscapeInputChanges(model, inputs, value));
  const pick = (event: PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width) return;
    const svgX = ((event.clientX - rect.left) / rect.width) * W;
    choose(x0 + ((svgX - PAD_X) / (W - PAD_X * 2)) * (x1 - x0));
  };
  const number = (value: number) => String(Number(value.toPrecision(4)));
  const axisName = model.mode === 'loss' ? 'p' : 'w';

  return (
    <figure className={`function-plot loss-landscape is-${model.mode}`}>
      <div className="loss-flow" aria-label={t(`plot.loss.${model.mode}Flow`)}>
        {model.mode === 'loss' ? (
          <>
            <span className="is-current">p = {number(model.current)}</span>
            <span>−</span>
            <span className="is-optimum">t = {number(model.target)}</span>
            <span>→</span>
            <span>|p−t| = {number(Math.abs(model.current - model.target))}</span>
            <span>→</span>
            <strong>L = {number(model.loss)}</strong>
          </>
        ) : (
          <>
            <span className="is-current">w = {number(model.current)}</span>
            <span>→</span>
            <span>p = {number(model.prediction)}</span>
            <span>→</span>
            <strong>L = {number(model.loss)}</strong>
            <span>→</span>
            <span>∂L/∂w = {number(model.gradient ?? 0)}</span>
          </>
        )}
      </div>
      <svg
        className="data-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="slider"
        tabIndex={0}
        aria-label={t(`plot.loss.${model.mode}Control`, { name: model.input })}
        aria-orientation="horizontal"
        aria-valuemin={x0}
        aria-valuemax={x1}
        aria-valuenow={model.current}
        onKeyDown={(event) => {
          const step = (x1 - x0) / 40;
          if (event.key === 'ArrowLeft') choose(model.current - step);
          else if (event.key === 'ArrowRight') choose(model.current + step);
          else return;
          event.preventDefault();
        }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          dragging.current = event.pointerId;
          event.currentTarget.setPointerCapture(event.pointerId);
          pick(event);
        }}
        onPointerMove={(event) => dragging.current === event.pointerId && pick(event)}
        onPointerUp={(event) => {
          if (dragging.current !== event.pointerId) return;
          pick(event);
          dragging.current = undefined;
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          dragging.current = undefined;
        }}
      >
        <defs>
          <marker
            id={`loss-step-${marker}`}
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path className="loss-step-head" d="M0,0 L7,3.5 L0,7 Z" />
          </marker>
          <marker
            id={`loss-gradient-${marker}`}
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path className="loss-gradient-head" d="M0,0 L7,3.5 L0,7 Z" />
          </marker>
        </defs>
        <line
          className="plot-axis"
          x1={PAD_X}
          x2={W - PAD_X}
          y1={H - PAD_BOTTOM}
          y2={H - PAD_BOTTOM}
        />
        <polyline className="loss-curve" points={curve} />
        <text className="plot-tick" x={W - PAD_X} y={H - 5} textAnchor="end">
          {axisName}
        </text>
        <text className="plot-tick" x={4} y={PAD_TOP + 2}>
          L
        </text>
        {model.mode === 'loss' && optimumPoint && (
          <g className="loss-error-distance">
            <line
              x1={Math.min(currentPoint[0], optimumPoint[0])}
              x2={Math.max(currentPoint[0], optimumPoint[0])}
              y1={H - PAD_BOTTOM - 7}
              y2={H - PAD_BOTTOM - 7}
            />
          </g>
        )}
        {optimumPoint && (
          <g className="loss-optimum">
            <line x1={optimumPoint[0]} x2={optimumPoint[0]} y1={PAD_TOP} y2={H - PAD_BOTTOM} />
            <circle cx={optimumPoint[0]} cy={optimumPoint[1]} r="4" />
            <text x={optimumPoint[0]} y={H - 7} textAnchor="middle">
              {model.mode === 'loss' ? 't' : 'w*'}
            </text>
          </g>
        )}
        <line
          className="loss-current-drop"
          x1={currentPoint[0]}
          x2={currentPoint[0]}
          y1={currentPoint[1]}
          y2={H - PAD_BOTTOM}
        />
        <circle className="loss-current" cx={currentPoint[0]} cy={currentPoint[1]} r="5" />
        <text className="loss-point-label" x={currentPoint[0] + 6} y={currentPoint[1] - 6}>
          {model.mode === 'loss' ? 'p' : 'w'}
        </text>
        {model.mode === 'loss' && actualValue !== undefined && (
          <circle
            className={`loss-player ${matches ? 'is-ok' : ''}`}
            cx={currentPoint[0]}
            cy={sy(actualValue)}
            r="3.3"
          />
        )}
        {model.mode === 'descent' && gradientPoint && (
          <line
            className="loss-gradient-direction"
            x1={currentPoint[0]}
            y1={currentPoint[1]}
            x2={gradientPoint[0]}
            y2={gradientPoint[1]}
            markerEnd={`url(#loss-gradient-${marker})`}
          />
        )}
        {model.mode === 'descent' && nextPoint && (
          <line
            className="loss-step"
            x1={currentPoint[0]}
            y1={currentPoint[1]}
            x2={nextPoint[0]}
            y2={nextPoint[1]}
            markerEnd={`url(#loss-step-${marker})`}
          />
        )}
        {model.mode === 'descent' && actualValue !== undefined && (
          <g className={`loss-player-update ${matches ? 'is-ok' : ''}`}>
            <line
              x1={sx(actualValue)}
              x2={sx(actualValue)}
              y1={sy(actualLoss ?? 0)}
              y2={H - PAD_BOTTOM}
            />
            <circle cx={sx(actualValue)} cy={sy(actualLoss ?? 0)} r="4" />
          </g>
        )}
      </svg>
      <figcaption className="hint plot-legend">
        <span className="plot-key is-loss-curve">{t('plot.loss.curve')}</span>
        <span className="plot-key is-loss-current">{t('plot.loss.current')}</span>
        <span className="plot-key is-loss-optimum">{t('plot.loss.minimum')}</span>
        {model.mode === 'descent' && (
          <>
            <span className="plot-key is-loss-gradient">{t('plot.loss.gradientDirection')}</span>
            <span className="plot-key is-loss-step">{t('plot.loss.targetStep')}</span>
          </>
        )}
        <span className={`plot-key is-actual${matches ? ' is-ok' : ''}`}>
          {t(model.mode === 'loss' ? 'plot.loss.yourLoss' : 'plot.loss.yourUpdate')}
        </span>
        <span className="plot-help">{t('plot.loss.help', { name: model.input })}</span>
      </figcaption>
    </figure>
  );
}
