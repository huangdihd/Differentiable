import { useEffect, useMemo, useState } from 'react';
import type { ForwardLevel } from '../engine/levels';
import { expectedOutput } from './expected-output';
import { sampleXs, segments, yRange } from './function-plot';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const W = 260,
  H = 170,
  PAD = 14;

/**
 * The lesson's target y = f(x) as a curve (and, for derivative lessons, the original f(x) behind
 * it), with the probe input marked: where the target sits and where the player's circuit lands.
 * Derivative lessons also draw the tangent to f at the probe, whose slope is the answer, and the
 * line the player's slope would draw. Clicking or dragging on the plot moves the probe input.
 */
export default function FunctionPlot({
  t,
  level,
  input,
  x,
  actual,
  onPick,
}: {
  t: T;
  level: ForwardLevel;
  input: string;
  x?: number;
  actual?: number;
  onPick: (x: number) => void;
}) {
  const xs = useMemo(() => sampleXs(level, input), [level, input]);
  const [curve, setCurve] = useState<{ target: number[]; original?: number[] }>();
  useEffect(() => {
    let cancelled = false;
    const shapes = { [input]: [xs.length] };
    const target = expectedOutput(level, { [input]: xs }, shapes);
    const original =
      level.kind === 'derivative'
        ? expectedOutput({ ...level, kind: 'forward' }, { [input]: xs }, shapes)
        : undefined;
    Promise.all([target, original]).then(
      ([a, b]) => !cancelled && setCurve({ target: a.values, original: b?.values }),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, input, xs]);
  if (!curve) return null;

  const x0 = xs[0],
    x1 = xs[xs.length - 1];
  // Derivative lessons keep f(x) in view too: f′ alone is often a sliver of f's scale.
  const [y0, y1] = yRange([...curve.target, ...(curve.original ?? [])]);
  const tick = (v: number) => Number(v.toPrecision(2));
  const sx = (v: number) => PAD + ((v - x0) / (x1 - x0)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - y0) / (y1 - y0)) * (H - 2 * PAD);
  const line = (ys: readonly number[]) =>
    segments(xs, ys, [y0, y1]).map((run) => run.map(([a, b]) => `${sx(a)},${sy(b)}`).join(' '));
  const inside = (v?: number): v is number => v !== undefined && Number.isFinite(v);
  const at = (ys: readonly number[]) =>
    inside(x) && x >= x0 && x <= x1
      ? ys[Math.round(((x - x0) / (x1 - x0)) * (xs.length - 1))]
      : undefined;
  const expected = at(curve.target);
  const matches =
    inside(expected) &&
    inside(actual) &&
    Math.abs(actual - expected) <= 1e-4 + 1e-3 * Math.abs(expected);
  // The tangent to f at the probe: its slope is f′(x), the value the circuit should output.
  const fx = curve.original && at(curve.original);
  const reach = (x1 - x0) * 0.22;
  const tangent = (slope: number, className: string) =>
    inside(x) &&
    inside(fx) && (
      <line
        className={className}
        x1={sx(x - reach)}
        y1={sy(fx - slope * reach)}
        x2={sx(x + reach)}
        y2={sy(fx + slope * reach)}
      />
    );
  const clampY = (v: number) => Math.min(H - 3, Math.max(3, sy(v)));
  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const v = x0 + (((e.clientX - r.left) / r.width) * W - PAD) * ((x1 - x0) / (W - 2 * PAD));
    onPick(Math.round(Math.min(x1, Math.max(x0, v)) * 10) / 10);
  };

  return (
    <figure className="function-plot">
      <svg
        className="data-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={t('plot.label')}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          pick(e);
        }}
        onPointerMove={(e) => e.buttons === 1 && pick(e)}
      >
        <defs>
          <clipPath id="function-plot-clip">
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
        <g clipPath="url(#function-plot-clip)">
          {curve.original &&
            line(curve.original).map((points, i) => (
              <polyline key={`o${i}`} className="plot-original" points={points} />
            ))}
          {line(curve.target).map((points, i) => (
            <polyline key={`t${i}`} className="plot-fit" points={points} />
          ))}
          {inside(x) && <line className="plot-probe" x1={sx(x)} x2={sx(x)} y1={0} y2={H} />}
          {inside(expected) && tangent(expected, `plot-tangent${matches ? ' is-ok' : ''}`)}
          {inside(actual) && !matches && tangent(actual, 'plot-tangent is-yours')}
          {inside(x) && inside(fx) && (
            <circle className="plot-touch" cx={sx(x)} cy={clampY(fx)} r={2.6} />
          )}
          {inside(x) && inside(expected) && (
            <circle className="plot-target" cx={sx(x)} cy={clampY(expected)} r={5} />
          )}
          {inside(x) && inside(actual) && (
            <circle
              className={`plot-actual ${matches ? 'is-ok' : 'is-off'}`}
              cx={sx(x)}
              cy={clampY(actual)}
              r={3.2}
            />
          )}
        </g>
      </svg>
      <figcaption className="hint plot-legend">
        <span className="plot-key is-target">
          {t(level.kind === 'derivative' ? 'plot.derivative' : 'plot.target')}
        </span>
        {curve.original && <span className="plot-key is-original">{t('plot.original')}</span>}
        {curve.original && <span className="plot-key is-tangent">{t('plot.tangent')}</span>}
        <span className="plot-key is-actual">{t('plot.yours')}</span>
        <span className="plot-help">{t('plot.help')}</span>
      </figcaption>
    </figure>
  );
}
