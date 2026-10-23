import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ForwardLevel } from '../engine/levels';
import { translateValueName } from '../i18n';
import { expectedOutput } from './expected-output';
import {
  functionAsymptotes,
  functionPoint,
  sampleValue,
  sampleXs,
  segments,
  yRange,
} from './function-plot';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;
type DerivativeView = 'derivative' | 'tangent';

const W = 260,
  H = 170,
  PAD = 14;

/**
 * The lesson's target output as a curve with the probe input and player's result. Derivative
 * lessons separate f′ from f and its tangent so neither view is crowded. Clicking or dragging on
 * the plot moves the probe input.
 */
export default function FunctionPlot({
  t,
  level,
  input,
  x,
  inputs,
  actual,
  onPick,
}: {
  t: T;
  level: ForwardLevel;
  input: string;
  x?: number;
  inputs: Readonly<Record<string, readonly number[]>>;
  actual?: number;
  onPick: (x: number) => void;
}) {
  const clipId = `function-plot-${useId().replace(/:/g, '')}`;
  const xs = useMemo(() => sampleXs(level, input), [level, input]);
  const fixed = Object.keys(level.interface.inputs).filter((name) => name !== input);
  const fixedKey = fixed.map((name) => `${name}:${inputs[name]?.[0]}`).join('|');
  const [view, setView] = useState<DerivativeView>('derivative');
  const [curve, setCurve] = useState<{ target: number[]; original?: number[] }>();
  const [point, setPoint] = useState<{ x: number; target: number; original?: number }>();
  // Keep the pointer's x locally while scrubbing. Routing it through the editor input alone makes
  // rapid moves wait behind probe work and gives the impression that the tangent updates on release.
  const dragging = useRef<number | undefined>(undefined);
  const lastPicked = useRef<number | undefined>(undefined);
  const [plotX, setPlotX] = useState(x);
  const [actualPoint, setActualPoint] = useState({ x, value: actual });
  useEffect(() => {
    if (dragging.current === undefined) setPlotX(x);
    lastPicked.current = x;
  }, [x]);
  // The parent clears the old probe value before evaluating a new input. Remember which x produced
  // each actual value so a stale player tangent is never moved to a fresh probe position.
  useEffect(() => setActualPoint({ x, value: actual }), [x, actual]);
  useEffect(() => setView('derivative'), [level.id]);
  useEffect(() => {
    let cancelled = false;
    setCurve(undefined);
    const names = Object.keys(level.interface.inputs);
    const shapes = Object.fromEntries(names.map((name) => [name, [xs.length]]));
    const curveInputs = Object.fromEntries(
      names.map((name) => [name, name === input ? xs : Array(xs.length).fill(inputs[name]?.[0])]),
    );
    if (
      Object.values(curveInputs).some((values) => values.some((value) => !Number.isFinite(value)))
    )
      return;
    const target = expectedOutput(level, curveInputs, shapes);
    const original =
      level.kind === 'derivative'
        ? expectedOutput({ ...level, kind: 'forward' }, curveInputs, shapes)
        : undefined;
    Promise.all([target, original]).then(
      ([a, b]) => !cancelled && setCurve({ target: a.values, original: b?.values }),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, input, xs, fixedKey]);
  useEffect(() => {
    setPoint(undefined);
    if (plotX === undefined || !Number.isFinite(plotX)) return;
    let cancelled = false;
    functionPoint(level, input, plotX, inputs).then(
      (value) => !cancelled && setPoint({ x: plotX, ...value }),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, input, plotX, fixedKey]);
  if (!curve) return null;

  const outputLabel = translateValueName(t, level.interface.output);
  const tangentView = Boolean(curve.original) && view === 'tangent';
  const shownCurve = tangentView ? curve.original! : curve.target;
  const asymptotes = functionAsymptotes(level, tangentView ? 'original' : 'target');
  const vertical = asymptotes.filter((line) => line.axis === 'x').map((line) => line.value);
  const sampleX0 = xs[0],
    sampleX1 = xs[xs.length - 1];
  const x0 = Math.min(sampleX0, ...vertical),
    x1 = Math.max(sampleX1, ...vertical);
  let [y0, y1] = yRange(shownCurve);
  for (const line of asymptotes) {
    if (line.axis !== 'y') continue;
    const span = y1 - y0;
    if (line.value <= y0) y0 = line.value - span * 0.04;
    if (line.value >= y1) y1 = line.value + span * 0.04;
  }
  const tick = (v: number) => Number(v.toPrecision(2));
  const sx = (v: number) => PAD + ((v - x0) / (x1 - x0)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - y0) / (y1 - y0)) * (H - 2 * PAD);
  const line = (ys: readonly number[]) =>
    segments(xs, ys, [y0, y1]).map((run) => run.map(([a, b]) => `${sx(a)},${sy(b)}`).join(' '));
  const inside = (v?: number): v is number => v !== undefined && Number.isFinite(v);
  const sampledAt = (ys?: readonly number[]) =>
    ys && inside(plotX) ? sampleValue(xs, ys, plotX, vertical) : undefined;
  const atProbe = inside(plotX) && plotX >= x0 && plotX <= x1 && point?.x === plotX;
  // Sample interpolation keeps the target tangent under the pointer while the exact point resolves.
  const expected = atProbe ? point.target : sampledAt(curve.target);
  const currentActual = actualPoint.x === plotX ? actualPoint.value : undefined;
  const matches =
    atProbe &&
    inside(expected) &&
    inside(currentActual) &&
    Math.abs(currentActual - expected) <=
      (level.atol ?? 1e-5) + (level.rtol ?? 1e-4) * Math.abs(expected);
  // The tangent to f at the probe: its slope is f′(x), the value the circuit should output.
  const fx = atProbe ? point.original : sampledAt(curve.original);
  const reach = (x1 - x0) * 0.22;
  const tangent = (slope: number, className: string) =>
    inside(plotX) &&
    inside(fx) && (
      <line
        className={className}
        x1={sx(plotX - reach)}
        y1={sy(fx - slope * reach)}
        x2={sx(plotX + reach)}
        y2={sy(fx + slope * reach)}
      />
    );
  const clampY = (v: number) => Math.min(H - 3, Math.max(3, sy(v)));
  const choose = (value: number) => {
    const next = Math.min(sampleX1, Math.max(sampleX0, Math.round(value * 10) / 10));
    setPlotX(next);
    if (lastPicked.current !== next) {
      lastPicked.current = next;
      onPick(next);
    }
  };
  const pick = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const value =
      x0 + (((event.clientX - rect.left) / rect.width) * W - PAD) * ((x1 - x0) / (W - 2 * PAD));
    choose(value);
  };
  const selectedX = Math.min(sampleX1, Math.max(sampleX0, inside(plotX) ? plotX : 0));
  const useKey = (event: React.KeyboardEvent<SVGSVGElement>) => {
    const value =
      event.key === 'ArrowLeft' || event.key === 'ArrowDown'
        ? selectedX - 0.1
        : event.key === 'ArrowRight' || event.key === 'ArrowUp'
          ? selectedX + 0.1
          : event.key === 'Home'
            ? sampleX0
            : event.key === 'End'
              ? sampleX1
              : undefined;
    if (value === undefined) return;
    event.preventDefault();
    choose(value);
  };

  return (
    <figure className="function-plot">
      {fixed.length > 0 && (
        <div className="plot-slice" aria-label={t('plot.partialSlice')}>
          <strong>{t('plot.partialDirection', { name: input })}</strong>
          {fixed.map((name) => (
            <span key={name}>
              {t('plot.fixedInput', { name, value: inputs[name]?.[0] ?? '?' })}
            </span>
          ))}
        </div>
      )}
      {curve.original && (
        <div className="seg plot-view-switch" role="group" aria-label={t('plot.view')}>
          <button
            type="button"
            className={`seg-btn${view === 'derivative' ? ' is-on' : ''}`}
            aria-pressed={view === 'derivative'}
            onClick={() => setView('derivative')}
          >
            {t('plot.viewDerivative')}
          </button>
          <button
            type="button"
            className={`seg-btn${view === 'tangent' ? ' is-on' : ''}`}
            aria-pressed={view === 'tangent'}
            onClick={() => setView('tangent')}
          >
            {t('plot.viewTangent')}
          </button>
        </div>
      )}
      <svg
        className="data-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="slider"
        tabIndex={0}
        aria-label={t(tangentView ? 'plot.originalControl' : 'plot.control', { name: input })}
        aria-orientation="horizontal"
        aria-valuemin={sampleX0}
        aria-valuemax={sampleX1}
        aria-valuenow={selectedX}
        aria-valuetext={t('plot.inputValue', { name: input, value: selectedX })}
        onKeyDown={useKey}
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
        onPointerCancel={(event) => {
          if (dragging.current !== event.pointerId) return;
          dragging.current = undefined;
          setPlotX(x);
        }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={0} y={0} width={W} height={H} />
          </clipPath>
        </defs>
        {x0 < 0 && x1 > 0 && (
          <line className="plot-axis" x1={sx(0)} x2={sx(0)} y1={PAD / 2} y2={H - PAD / 2} />
        )}
        {y0 < 0 && y1 > 0 && (
          <line className="plot-axis" x1={PAD / 2} x2={W - PAD / 2} y1={sy(0)} y2={sy(0)} />
        )}
        <text className="plot-tick" x={PAD} y={H - 3}>
          {x0 < 0 ? Math.round(x0) : 0}
        </text>
        <text className="plot-tick" x={W - PAD} y={H - 3} textAnchor="end">
          {Math.round(x1)}
        </text>
        <text className="plot-tick" x={3} y={PAD - 4}>
          {tick(y1)}
        </text>
        <text className="plot-tick" x={3} y={H - PAD - 3}>
          {tick(y0)}
        </text>
        <g clipPath={`url(#${clipId})`}>
          {asymptotes.map((asymptote) =>
            asymptote.axis === 'x' && asymptote.value >= x0 && asymptote.value <= x1 ? (
              <g className="plot-asymptote" key={`x-${asymptote.value}`}>
                <line
                  x1={sx(asymptote.value)}
                  x2={sx(asymptote.value)}
                  y1={PAD / 2}
                  y2={H - PAD / 2}
                />
                <text x={sx(asymptote.value) + 4} y={PAD + 8}>
                  x = {tick(asymptote.value)}
                </text>
              </g>
            ) : asymptote.axis === 'y' && asymptote.value >= y0 && asymptote.value <= y1 ? (
              <g className="plot-asymptote" key={`y-${asymptote.value}`}>
                <line
                  x1={PAD / 2}
                  x2={W - PAD / 2}
                  y1={sy(asymptote.value)}
                  y2={sy(asymptote.value)}
                />
                <text x={W - PAD - 2} y={sy(asymptote.value) - 3} textAnchor="end">
                  y = {tick(asymptote.value)}
                </text>
              </g>
            ) : null,
          )}
          {line(shownCurve).map((points, index) => (
            <polyline
              key={index}
              className={tangentView ? 'plot-function' : 'plot-fit'}
              points={points}
            />
          ))}
          {inside(plotX) && (
            <line className="plot-probe" x1={sx(plotX)} x2={sx(plotX)} y1={0} y2={H} />
          )}
          {tangentView && inside(expected) && tangent(expected, 'plot-tangent')}
          {tangentView &&
            inside(currentActual) &&
            tangent(currentActual, `plot-tangent is-yours${matches ? ' is-ok' : ''}`)}
          {tangentView && inside(plotX) && inside(fx) && (
            <circle className="plot-touch" cx={sx(plotX)} cy={clampY(fx)} r={3.2} />
          )}
          {!tangentView && inside(plotX) && inside(expected) && (
            <circle className="plot-target" cx={sx(plotX)} cy={clampY(expected)} r={5} />
          )}
          {!tangentView && inside(plotX) && inside(currentActual) && (
            <circle
              className={`plot-actual ${matches ? 'is-ok' : 'is-off'}`}
              cx={sx(plotX)}
              cy={clampY(currentActual)}
              r={3.2}
            />
          )}
        </g>
      </svg>
      <figcaption className="hint plot-legend">
        {tangentView ? (
          <>
            <span className="plot-key is-function">{t('plot.original')}</span>
            <span className="plot-key is-tangent">{t('plot.tangent')}</span>
            <span className={`plot-key is-your-tangent${matches ? ' is-ok' : ''}`}>
              {t('plot.yourTangent')}
            </span>
          </>
        ) : (
          <>
            <span className="plot-key is-target">
              {t(level.kind === 'derivative' ? 'plot.derivative' : 'plot.target', {
                output: outputLabel,
              })}
            </span>
            <span className={`plot-key is-actual${matches ? ' is-ok' : ''}`}>
              {t('plot.yours')}
            </span>
          </>
        )}
        {asymptotes.length > 0 && (
          <span className="plot-key is-asymptote">{t('plot.asymptote')}</span>
        )}
        <span className="plot-help">{t('plot.help', { name: input })}</span>
      </figcaption>
    </figure>
  );
}
