import { useId } from 'react';

/** What a training level learns from, in the plane: the points and what the model makes of them. */
export type PlotData =
  | {
      kind: 'regression';
      x: number[];
      t: number[];
      /** The model's prediction at each x, when known. */
      p?: number[];
    }
  | {
      kind: 'classes';
      /** Two features per point. */
      points: [number, number][];
      label: number[];
      /** Predicted class per point, when known. */
      predicted?: number[];
      /** The model's class over the plane: the colors meet at its decision boundary. */
      field?: ClassField;
    };

/** Predicted class at the centres of an nx×ny grid over [x0,x1]×[y0,y1]; row 0 is the bottom. */
export interface ClassField {
  nx: number;
  ny: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  classes: number[];
  /** Two classes only: score of class 1 minus class 0 per cell; the boundary is where it is 0. */
  margin?: number[];
}

/** The boundary as SVG path data in field-cell coordinates (cell centres at i + 0.5). */
export function boundaryPath(
  field: ClassField,
  px: (i: number) => number,
  py: (j: number) => number,
): string {
  const { nx, ny, classes, margin } = field;
  const at = (i: number, j: number) => j * nx + i;
  const parts: string[] = [];
  const seg = (a: [number, number], b: [number, number]) =>
    parts.push(
      `M${px(a[0]).toFixed(1)},${py(a[1]).toFixed(1)}L${px(b[0]).toFixed(1)},${py(b[1]).toFixed(1)}`,
    );
  if (margin) {
    // Marching squares over cell centres, interpolating the zero crossing: a smooth line.
    const v = (i: number, j: number) => margin[at(i, j)];
    const cross = (i0: number, j0: number, i1: number, j1: number): [number, number] => {
      const a = v(i0, j0),
        b = v(i1, j1),
        k = a / (a - b);
      return [i0 + 0.5 + (i1 - i0) * k, j0 + 0.5 + (j1 - j0) * k];
    };
    for (let j = 0; j + 1 < ny; j++)
      for (let i = 0; i + 1 < nx; i++) {
        const corners: [number, number][] = [
          [i, j],
          [i + 1, j],
          [i + 1, j + 1],
          [i, j + 1],
        ];
        const points: [number, number][] = [];
        for (let k = 0; k < 4; k++) {
          const [i0, j0] = corners[k],
            [i1, j1] = corners[(k + 1) % 4];
          if (v(i0, j0) > 0 !== v(i1, j1) > 0) points.push(cross(i0, j0, i1, j1));
        }
        if (points.length >= 2) seg(points[0], points[1]);
        if (points.length === 4) seg(points[2], points[3]);
      }
  } else {
    // Several classes: trace the cell edges where neighbours disagree.
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        if (i + 1 < nx && classes[at(i, j)] !== classes[at(i + 1, j)])
          seg([i + 1, j], [i + 1, j + 1]);
        if (j + 1 < ny && classes[at(i, j)] !== classes[at(i, j + 1)])
          seg([i, j + 1], [i + 1, j + 1]);
      }
  }
  return parts.join('');
}

const W = 260,
  H = 180,
  PAD = 14;

export default function DataPlot({
  data,
  label,
  highlight,
  onHover,
  axisLabels,
}: {
  data: PlotData;
  label: string;
  /** Sample index to ring, e.g. the row the player points at in a table. */
  highlight?: number;
  onHover?: (index: number | undefined) => void;
  /** What the horizontal and vertical axes are, e.g. which column of x. */
  axisLabels?: { x: string; y: string };
}) {
  const clipId = `data-plot-${useId().replace(/:/g, '')}`;
  const xs = data.kind === 'regression' ? data.x : data.points.map((p) => p[0]);
  const ys =
    data.kind === 'regression'
      ? [...data.t, ...(data.p ?? []).filter(Number.isFinite)]
      : data.points.map((p) => p[1]);
  const range = (v: number[]) => {
    const lo = Math.min(...v),
      hi = Math.max(...v);
    const pad = (hi - lo || 1) * 0.08;
    return [lo - pad, hi + pad] as const;
  };
  // Regression keeps the target's scale so a wild prediction runs off the plot, not the data.
  const [x0, x1] = range(xs);
  const [y0, y1] = range(data.kind === 'regression' ? data.t : ys);
  const sx = (v: number) => PAD + ((v - x0) / (x1 - x0)) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - y0) / (y1 - y0)) * (H - 2 * PAD);
  const axes = (
    <>
      {x0 < 0 && x1 > 0 && (
        <line className="plot-axis" x1={sx(0)} x2={sx(0)} y1={PAD / 2} y2={H - PAD / 2} />
      )}
      {y0 < 0 && y1 > 0 && (
        <line className="plot-axis" x1={PAD / 2} x2={W - PAD / 2} y1={sy(0)} y2={sy(0)} />
      )}
    </>
  );
  return (
    <svg className="data-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={W} height={H} />
        </clipPath>
      </defs>
      {axes}
      {data.kind === 'regression' ? (
        <g clipPath={`url(#${clipId})`}>
          {data.p && data.p.every(Number.isFinite) && (
            <polyline
              className="plot-fit"
              points={data.x
                .map((x, i) => [x, data.p![i]] as const)
                .sort((a, b) => a[0] - b[0])
                .map(([x, p]) => `${sx(x)},${sy(p)}`)
                .join(' ')}
            />
          )}
          {data.x.map((x, i) => (
            <circle
              key={i}
              className="plot-point"
              cx={sx(x)}
              cy={sy(data.t[i])}
              r={3}
              onMouseEnter={onHover && (() => onHover(i))}
              onMouseLeave={onHover && (() => onHover(undefined))}
            />
          ))}
        </g>
      ) : (
        <>
          {data.field && <ClassRegions field={data.field} sx={sx} sy={sy} clipId={clipId} />}
          {data.points.map(([a, b], i) => {
            const wrong = data.predicted && data.predicted[i] !== data.label[i];
            return (
              <circle
                key={i}
                className={`plot-point is-class-${data.label[i] % 4}${wrong ? ' is-wrong' : ''}`}
                cx={sx(a)}
                cy={sy(b)}
                r={3.2}
                onMouseEnter={onHover && (() => onHover(i))}
                onMouseLeave={onHover && (() => onHover(undefined))}
              />
            );
          })}
        </>
      )}
      {highlight !== undefined && (
        <circle
          className="plot-highlight"
          cx={sx(data.kind === 'regression' ? data.x[highlight] : data.points[highlight][0])}
          cy={sy(data.kind === 'regression' ? data.t[highlight] : data.points[highlight][1])}
          r={7}
        />
      )}
      {axisLabels && (
        <>
          <text className="plot-label" x={W - 4} y={H - 4} textAnchor="end">
            {axisLabels.x} →
          </text>
          <text className="plot-label" x={4} y={11}>
            ↑ {axisLabels.y}
          </text>
        </>
      )}
    </svg>
  );
}

function ClassRegions({
  field,
  sx,
  sy,
  clipId,
}: {
  field: ClassField;
  sx: (v: number) => number;
  sy: (v: number) => number;
  clipId: string;
}) {
  const { nx, ny, x0, x1, y0, y1, classes } = field;
  // Field-cell coordinates -> plot pixels.
  const px = (i: number) => sx(x0 + (i / nx) * (x1 - x0));
  const py = (j: number) => sy(y0 + (j / ny) * (y1 - y0));
  return (
    <g className="plot-field" clipPath={`url(#${clipId})`}>
      {classes.map((c, k) => {
        const i = k % nx,
          j = Math.floor(k / nx);
        const x = px(i),
          y = py(j + 1);
        return (
          <rect
            key={k}
            className={`plot-region is-class-${c % 4}`}
            x={x}
            y={y}
            width={px(i + 1) - x + 0.4}
            height={py(j) - y + 0.4}
          />
        );
      })}
      <path className="plot-boundary" d={boundaryPath(field, px, py)} />
    </g>
  );
}
