import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ForwardLevel } from '../engine/levels';
import { expectedOutputs, type ExpectedValue } from './expected-output';
import {
  gradientSurfaceData,
  projectSurfacePoint,
  surfaceValue,
  unprojectSurfaceFloor,
  yRange,
  type GradientSurfaceData,
  type SurfaceView,
} from './function-plot';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;
type Point = readonly [number, number];
type Interaction = 'pick' | 'rotate';
type Drag = {
  id: number;
  x: number;
  y: number;
  view: SurfaceView;
};

const W = 260,
  H = 190,
  DEFAULT_VIEW: SurfaceView = { yaw: Math.PI / 4, pitch: 0.65 };

/** A two-input function surface with its gradient vector drawn in the input plane. */
export default function GradientSurfacePlot({
  t,
  level,
  names,
  inputs,
  actual,
  onPick,
}: {
  t: T;
  level: ForwardLevel;
  names: readonly [string, string];
  inputs: Readonly<Record<string, readonly number[]>>;
  actual: Readonly<Record<string, ExpectedValue>>;
  onPick: (inputs: Readonly<Record<string, number>>) => void;
}) {
  const marker = useId().replace(/:/g, '');
  const [data, setData] = useState<GradientSurfaceData>();
  const [failed, setFailed] = useState(false);
  const inputPoint: Point = [inputs[names[0]]?.[0] ?? 0, inputs[names[1]]?.[0] ?? 0];
  const [point, setPoint] = useState<Point>(inputPoint);
  const [exact, setExact] = useState<{ point: Point; values: readonly [number, number] }>();
  const [interaction, setInteraction] = useState<Interaction>('pick');
  const [view, setView] = useState<SurfaceView>(DEFAULT_VIEW);
  const dragging = useRef<Drag | undefined>(undefined);
  const lastPicked = useRef('');

  useEffect(() => {
    let cancelled = false;
    setData(undefined);
    setFailed(false);
    gradientSurfaceData(level, names).then(
      (result) => !cancelled && setData(result),
      () => !cancelled && setFailed(true),
    );
    return () => {
      cancelled = true;
    };
  }, [level, names[0], names[1]]);
  useEffect(() => {
    if (!dragging.current) setPoint(inputPoint);
  }, [inputPoint[0], inputPoint[1]]);
  useEffect(() => {
    if (!point.every(Number.isFinite)) return;
    let cancelled = false;
    const probe = Object.fromEntries(names.map((name, i) => [name, [point[i]]]));
    expectedOutputs(level, probe).then(
      (values) => {
        const outputs = Object.keys(level.interface.outputs ?? {});
        if (!cancelled)
          setExact({
            point,
            values: [values[outputs[0]].values[0], values[outputs[1]].values[0]],
          });
      },
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, names[0], names[1], point[0], point[1]]);

  const outputNames = Object.keys(level.interface.outputs ?? {});
  const actualVector = outputNames.every((name) => Number.isFinite(actual[name]?.values[0]))
    ? ([actual[outputNames[0]].values[0], actual[outputNames[1]].values[0]] as const)
    : undefined;
  const actualAtPoint = inputPoint[0] === point[0] && inputPoint[1] === point[1];
  const targetVector = useMemo((): Point | undefined => {
    if (!data) return undefined;
    if (exact?.point[0] === point[0] && exact.point[1] === point[1]) return exact.values;
    const first = surfaceValue(data.axes, data.target[0], point[0], point[1]);
    const second = surfaceValue(data.axes, data.target[1], point[0], point[1]);
    return first !== undefined && second !== undefined ? [first, second] : undefined;
  }, [data, exact, point[0], point[1]]);
  if (!data)
    return (
      <figure className="function-plot gradient-surface">
        <p className="hint surface-loading" role={failed ? 'alert' : undefined}>
          {t(failed ? 'plot.surfaceError' : 'plot.surfaceLoading')}
        </p>
      </figure>
    );

  const [as, bs] = data.axes;
  const a0 = as[0],
    a1 = as.at(-1)!,
    b0 = bs[0],
    b1 = bs.at(-1)!;
  const [z0, z1] = yRange(data.surface);
  const zSpan = z1 - z0;
  const floor = z0 - zSpan * 0.08;
  const norm = (value: number, lo: number, hi: number) => ((value - lo) / (hi - lo)) * 2 - 1;
  const project = (a: number, b: number, z: number): Point =>
    projectSurfacePoint(norm(a, a0, a1), norm(b, b0, b1), (z - (z0 + z1) / 2) / (zSpan / 2), view);
  const mesh = (axis: 'a' | 'b') =>
    (axis === 'a' ? bs : as).map((_, outer) =>
      (axis === 'a' ? as : bs)
        .map((__, inner) => {
          const i = axis === 'a' ? inner : outer,
            j = axis === 'a' ? outer : inner;
          return project(as[i], bs[j], data.surface[j * as.length + i]).join(',');
        })
        .join(' '),
    );
  const floorPoints = [
    project(a0, b0, floor),
    project(a1, b0, floor),
    project(a1, b1, floor),
    project(a0, b1, floor),
  ];
  const currentZ = surfaceValue(data.axes, data.surface, point[0], point[1]);
  const floorPoint = project(point[0], point[1], floor);
  const surfacePoint = currentZ === undefined ? undefined : project(point[0], point[1], currentZ);
  const tolerance = (target: number) =>
    (level.atol ?? 1e-5) + (level.rtol ?? 1e-4) * Math.abs(target);
  const matches =
    actualAtPoint &&
    targetVector !== undefined &&
    actualVector !== undefined &&
    actualVector.every(
      (value, i) => Math.abs(value - targetVector[i]) <= tolerance(targetVector[i]),
    );
  const vectors = [targetVector, actualAtPoint ? actualVector : undefined].filter(
    (value): value is Point => value !== undefined,
  );
  const maxMagnitude = Math.max(1, ...vectors.map(([a, b]) => Math.hypot(a, b)));
  const vectorScale = (Math.min(a1 - a0, b1 - b0) * 0.3) / maxMagnitude;
  const arrow = (vector: Point, kind: 'target' | 'actual') => {
    const end = project(
      point[0] + vector[0] * vectorScale,
      point[1] + vector[1] * vectorScale,
      floor,
    );
    const state = matches ? 'good' : kind;
    return (
      <line
        className={`surface-gradient is-${state}`}
        x1={floorPoint[0]}
        y1={floorPoint[1]}
        x2={end[0]}
        y2={end[1]}
        markerEnd={`url(#gradient-${state}-${marker})`}
      />
    );
  };
  const choose = (a: number, b: number) => {
    const next: Point = [
      Math.min(a1, Math.max(a0, Math.round(a * 10) / 10)),
      Math.min(b1, Math.max(b0, Math.round(b * 10) / 10)),
    ];
    setPoint(next);
    const key = next.join('|');
    if (key === lastPicked.current) return;
    lastPicked.current = key;
    onPick({ [names[0]]: next[0], [names[1]]: next[1] });
  };
  const pick = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const screenX = ((event.clientX - rect.left) / rect.width) * W;
    const screenY = ((event.clientY - rect.top) / rect.height) * H;
    const floorHeight = (floor - (z0 + z1) / 2) / (zSpan / 2);
    const [x, y] = unprojectSurfaceFloor(screenX, screenY, floorHeight, view);
    choose(a0 + ((x + 1) / 2) * (a1 - a0), b0 + ((y + 1) / 2) * (b1 - b0));
  };
  const rotate = (event: React.PointerEvent<SVGSVGElement>, drag: Drag) => {
    setView({
      yaw: drag.view.yaw - (event.clientX - drag.x) * 0.012,
      pitch: Math.max(0.18, Math.min(1.35, drag.view.pitch + (event.clientY - drag.y) * 0.01)),
    });
  };
  const useKey = (event: React.KeyboardEvent<SVGSVGElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowDown', 'ArrowUp'].includes(event.key)) return;
    event.preventDefault();
    if (interaction === 'rotate') {
      setView((current) => ({
        yaw:
          current.yaw + (event.key === 'ArrowLeft' ? 0.08 : event.key === 'ArrowRight' ? -0.08 : 0),
        pitch: Math.max(
          0.18,
          Math.min(
            1.35,
            current.pitch +
              (event.key === 'ArrowDown' ? 0.06 : event.key === 'ArrowUp' ? -0.06 : 0),
          ),
        ),
      }));
      return;
    }
    const next: [number, number] = [point[0], point[1]];
    if (event.key === 'ArrowLeft') next[0] -= 0.1;
    else if (event.key === 'ArrowRight') next[0] += 0.1;
    else if (event.key === 'ArrowDown') next[1] -= 0.1;
    else next[1] += 0.1;
    choose(next[0], next[1]);
  };

  return (
    <figure className="function-plot gradient-surface">
      <div className="plot-slice">
        <strong>{t('plot.gradientSurface')}</strong>
        <span>{t('plot.inputValue', { name: names[0], value: point[0] })}</span>
        <span>{t('plot.inputValue', { name: names[1], value: point[1] })}</span>
      </div>
      <div className="seg plot-view-switch" role="group" aria-label={t('plot.surfaceMode')}>
        {(['pick', 'rotate'] as const).map((mode) => (
          <button
            type="button"
            className={`seg-btn${interaction === mode ? ' is-on' : ''}`}
            aria-pressed={interaction === mode}
            onClick={() => setInteraction(mode)}
            key={mode}
          >
            {t(mode === 'pick' ? 'plot.surfacePick' : 'plot.surfaceRotate')}
          </button>
        ))}
        <button type="button" className="seg-btn" onClick={() => setView(DEFAULT_VIEW)}>
          {t('plot.surfaceReset')}
        </button>
      </div>
      <svg
        className="data-plot"
        data-interaction={interaction}
        viewBox={`0 0 ${W} ${H}`}
        role="application"
        tabIndex={0}
        aria-label={t(
          interaction === 'pick' ? 'plot.surfaceControl' : 'plot.surfaceRotateControl',
          { first: names[0], second: names[1] },
        )}
        onKeyDown={useKey}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          dragging.current = {
            id: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            view,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
          if (interaction === 'pick') pick(event);
        }}
        onPointerMove={(event) => {
          const drag = dragging.current;
          if (drag?.id !== event.pointerId) return;
          if (interaction === 'pick') pick(event);
          else rotate(event, drag);
        }}
        onPointerUp={(event) => {
          const drag = dragging.current;
          if (drag?.id !== event.pointerId) return;
          if (interaction === 'pick') pick(event);
          else rotate(event, drag);
          dragging.current = undefined;
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={(event) => {
          const drag = dragging.current;
          if (drag?.id !== event.pointerId) return;
          dragging.current = undefined;
          if (interaction === 'pick') setPoint(inputPoint);
          else setView(drag.view);
        }}
      >
        <defs>
          {(['target', 'actual', 'good'] as const).map((kind) => (
            <marker
              id={`gradient-${kind}-${marker}`}
              key={kind}
              markerWidth="7"
              markerHeight="7"
              refX="5.5"
              refY="3.5"
              orient="auto"
            >
              <path className={`surface-arrow is-${kind}`} d="M0,0 L7,3.5 L0,7 Z" />
            </marker>
          ))}
        </defs>
        <polygon className="surface-floor" points={floorPoints.map((p) => p.join(',')).join(' ')} />
        {floorPoints.map((point, i) => (
          <line
            className="surface-floor-edge"
            key={i}
            x1={point[0]}
            y1={point[1]}
            x2={floorPoints[(i + 1) % floorPoints.length][0]}
            y2={floorPoints[(i + 1) % floorPoints.length][1]}
          />
        ))}
        {[...mesh('a'), ...mesh('b')].map((points, i) => (
          <polyline className="surface-mesh" points={points} key={i} />
        ))}
        {surfacePoint && (
          <>
            <line
              className="surface-drop"
              x1={floorPoint[0]}
              y1={floorPoint[1]}
              x2={surfacePoint[0]}
              y2={surfacePoint[1]}
            />
            <circle className="surface-point" cx={surfacePoint[0]} cy={surfacePoint[1]} r="3.2" />
          </>
        )}
        {targetVector && arrow(targetVector, 'target')}
        {actualAtPoint && actualVector && arrow(actualVector, 'actual')}
        <text
          className="plot-tick surface-axis-label"
          x={floorPoints[1][0] + 3}
          y={floorPoints[1][1] + 10}
        >
          {names[0]}
        </text>
        <text
          className="plot-tick surface-axis-label"
          x={floorPoints[3][0] - 3}
          y={floorPoints[3][1] + 10}
          textAnchor="end"
        >
          {names[1]}
        </text>
      </svg>
      <figcaption className="hint plot-legend">
        <span className="plot-key is-surface-target">{t('plot.targetGradient')}</span>
        <span className={`plot-key is-surface-actual${matches ? ' is-ok' : ''}`}>
          {t('plot.yourGradient')}
        </span>
        <span className="plot-help">{t('plot.gradientDirectionHelp')}</span>
        <span className="plot-help">
          {t(interaction === 'pick' ? 'plot.surfaceHelp' : 'plot.surfaceRotateHelp')}
        </span>
      </figcaption>
    </figure>
  );
}
