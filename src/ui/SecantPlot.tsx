// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * calc-numeric: the player's slope circuit drawn as a line through (x, x²). When the circuit is
 * right that line is the secant through (x+h, (x+h)²); shrinking h turns it into the tangent.
 */
import { useEffect, useRef, useState, type PointerEvent } from 'react';

type T = (key: string, vars?: Record<string, string | number>) => string;

const W = 260;
const H = 200;
const X0 = -3;
const X1 = 3;
const Y0 = -1.5;
const Y1 = 9;
const MIN_H = 0.001;
const MAX_H = 2;
/** How close the player's line must come to the true secant slope to count as passing B. */
const SLOPE_TOLERANCE = 1e-3;

const f = (x: number) => x * x;
const sx = (x: number) => ((x - X0) / (X1 - X0)) * W;
const sy = (y: number) => H - ((y - Y0) / (Y1 - Y0)) * H;
const dataX = (px: number) => X0 + (px / W) * (X1 - X0);
const fmt = (value: number) => {
  const text = String(Number(value.toFixed(4)));
  return text === '-0' ? '0' : text.replace('-', '−');
};

function curvePoints(): string {
  const points: string[] = [];
  for (let i = 0; i <= 120; i += 1) {
    const x = X0 + ((X1 - X0) * i) / 120;
    points.push(`${sx(x).toFixed(1)},${sy(Math.min(f(x), Y1 + 4)).toFixed(1)}`);
  }
  return points.join(' ');
}
const CURVE = curvePoints();

/** A line through (x, y) with `slope`, long enough to cross the whole plot. */
function lineThrough(x: number, y: number, slope: number) {
  const reach = 20;
  return {
    x1: sx(x - reach),
    y1: sy(y - slope * reach),
    x2: sx(x + reach),
    y2: sy(y + slope * reach),
  };
}

/** Corner magnifier around A and B: zooms with h so the rise/run triangle stays readable. */
const LENS = 92;
function Magnifier({
  x,
  h,
  actual,
  hitsB,
}: {
  x: number;
  h: number;
  actual?: number;
  hitsB: boolean;
}) {
  const half = Math.max(1.1 * h, 0.004);
  const cx = x + h / 2;
  const cy = (f(x) + f(x + h)) / 2;
  // Equal scale on both axes, so on-screen angles are true slopes.
  const lx = (value: number) => ((value - (cx - half)) / (2 * half)) * LENS;
  const ly = (value: number) => LENS - ((value - (cy - half)) / (2 * half)) * LENS;
  const through = (slope: number) => ({
    x1: lx(x - 4 * half),
    y1: ly(f(x) - slope * 4 * half),
    x2: lx(x + 4 * half),
    y2: ly(f(x) + slope * 4 * half),
  });
  const curve = Array.from({ length: 41 }, (_, i) => {
    const value = cx - half + (2 * half * i) / 40;
    return `${lx(value).toFixed(1)},${ly(f(value)).toFixed(1)}`;
  }).join(' ');
  const zoom = Math.round((X1 - X0) / (2 * half));
  return (
    <g className="secant-lens" transform={`translate(${W - LENS - 4} ${H - LENS - 4})`}>
      <rect className="secant-lens-frame" width={LENS} height={LENS} />
      <svg width={LENS} height={LENS} viewBox={`0 0 ${LENS} ${LENS}`} overflow="hidden">
        <polyline className="plot-function" points={curve} />
        <line className="plot-tangent" {...through(2 * x)} />
        {actual !== undefined && Number.isFinite(actual) && (
          <line className={`plot-tangent is-yours${hitsB ? ' is-ok' : ''}`} {...through(actual)} />
        )}
        <g className="secant-rise-run">
          <line x1={lx(x)} y1={ly(f(x))} x2={lx(x + h)} y2={ly(f(x))} />
          <line x1={lx(x + h)} y1={ly(f(x))} x2={lx(x + h)} y2={ly(f(x + h))} />
        </g>
        <circle className="secant-handle is-a" cx={lx(x)} cy={ly(f(x))} r={3.5} />
        <circle className="secant-handle is-b" cx={lx(x + h)} cy={ly(f(x + h))} r={3.5} />
      </svg>
      <text className="plot-tick secant-lens-zoom" x={4} y={11}>
        ×{zoom}
      </text>
    </g>
  );
}

export default function SecantPlot({
  t,
  x,
  h,
  actual,
  onChange,
}: {
  t: T;
  x?: number;
  h?: number;
  /** The player's circuit output for the current inputs. */
  actual?: number;
  onChange: (changes: { x?: number; h?: number }) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const dragging = useRef<'a' | 'b'>(undefined);
  const animation = useRef<number>(undefined);
  const [animating, setAnimating] = useState(false);
  useEffect(() => () => cancelAnimationFrame(animation.current ?? 0), []);

  if (x === undefined || h === undefined || !Number.isFinite(x) || !Number.isFinite(h)) return null;
  const bx = x + h;
  const secant = (f(bx) - f(x)) / h;
  const hasActual = actual !== undefined && Number.isFinite(actual);
  const hitsB =
    hasActual && Math.abs(actual - secant) <= SLOPE_TOLERANCE * Math.max(1, Math.abs(secant));
  const inView = (value: number) => value >= X0 && value <= X1;

  const pointerX = (event: PointerEvent<SVGSVGElement>) => {
    const box = svg.current!.getBoundingClientRect();
    return dataX(((event.clientX - box.left) / box.width) * W);
  };
  const drag = (event: PointerEvent<SVGSVGElement>) => {
    if (!dragging.current) return;
    const px = pointerX(event);
    if (dragging.current === 'a')
      onChange({ x: Math.round(Math.max(X0, Math.min(X1, px)) * 100) / 100 });
    else onChange({ h: Number(Math.max(MIN_H, Math.min(MAX_H, px - x)).toPrecision(3)) });
  };
  // Shrink h geometrically to MIN_H over ~1.6 s; each frame re-probes the player's circuit.
  const shrink = () => {
    cancelAnimationFrame(animation.current ?? 0);
    const start = performance.now();
    const from = Math.max(h, 0.5);
    setAnimating(true);
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / 1600);
      onChange({ h: Number((from * (MIN_H / from) ** progress).toPrecision(3)) });
      if (progress < 1) animation.current = requestAnimationFrame(step);
      else setAnimating(false);
    };
    animation.current = requestAnimationFrame(step);
  };

  return (
    <figure className="function-plot secant-plot">
      <svg
        ref={svg}
        className="data-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={t('plot.secant.alt')}
        onPointerMove={drag}
        onPointerUp={() => (dragging.current = undefined)}
        onPointerCancel={() => (dragging.current = undefined)}
      >
        <defs>
          <clipPath id="secant-plot-clip">
            <rect width={W} height={H} />
          </clipPath>
        </defs>
        <line className="plot-axis" x1={sx(0)} x2={sx(0)} y1={0} y2={H} />
        <line className="plot-axis" x1={0} x2={W} y1={sy(0)} y2={sy(0)} />
        <g clipPath="url(#secant-plot-clip)">
          <polyline className="plot-function" points={CURVE} />
          <line className="plot-tangent" {...lineThrough(x, f(x), 2 * x)} />
          {hasActual && (
            <line
              className={`plot-tangent is-yours${hitsB ? ' is-ok' : ''}`}
              {...lineThrough(x, f(x), actual)}
            />
          )}
          {inView(bx) && (
            <g className="secant-rise-run" aria-hidden="true">
              <line x1={sx(x)} y1={sy(f(x))} x2={sx(bx)} y2={sy(f(x))} />
              <line x1={sx(bx)} y1={sy(f(x))} x2={sx(bx)} y2={sy(f(bx))} />
            </g>
          )}
          <circle
            className="secant-handle is-a"
            cx={sx(x)}
            cy={sy(f(x))}
            r={5}
            onPointerDown={(event) => {
              dragging.current = 'a';
              svg.current!.setPointerCapture(event.pointerId);
            }}
          />
          {inView(bx) && (
            <circle
              className="secant-handle is-b"
              cx={sx(bx)}
              cy={sy(f(bx))}
              r={5}
              onPointerDown={(event) => {
                dragging.current = 'b';
                svg.current!.setPointerCapture(event.pointerId);
              }}
            />
          )}
        </g>
        <Magnifier x={x} h={h} actual={hasActual ? actual : undefined} hitsB={hitsB} />
      </svg>
      <figcaption className="hint plot-legend">
        <span className="plot-key is-tangent">
          {t('plot.secant.tangent', { slope: fmt(2 * x) })}
        </span>
        <span className={`plot-key is-your-tangent${hitsB ? ' is-ok' : ''}`}>
          {hasActual
            ? t(hitsB ? 'plot.secant.yoursHits' : 'plot.secant.yoursMisses', { slope: fmt(actual) })
            : t('plot.secant.yoursNone')}
        </span>
        <span className="plot-help">{t('plot.secant.help')}</span>
      </figcaption>
      <button type="button" className="btn secant-shrink" disabled={animating} onClick={shrink}>
        {t('plot.secant.shrink')}
      </button>
    </figure>
  );
}
