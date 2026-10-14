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
    };

const W = 260,
  H = 180,
  PAD = 14;

export default function DataPlot({ data, label }: { data: PlotData; label: string }) {
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
        <clipPath id="data-plot-clip">
          <rect x={0} y={0} width={W} height={H} />
        </clipPath>
      </defs>
      {axes}
      {data.kind === 'regression' ? (
        <g clipPath="url(#data-plot-clip)">
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
            <circle key={i} className="plot-point" cx={sx(x)} cy={sy(data.t[i])} r={3} />
          ))}
        </g>
      ) : (
        data.points.map(([a, b], i) => {
          const wrong = data.predicted && data.predicted[i] !== data.label[i];
          return (
            <circle
              key={i}
              className={`plot-point is-class-${data.label[i] % 4}${wrong ? ' is-wrong' : ''}`}
              cx={sx(a)}
              cy={sy(b)}
              r={3.2}
            />
          );
        })
      )}
    </svg>
  );
}
