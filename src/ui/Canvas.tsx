import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import type { ComponentRegistry, GraphJSON, GraphEndpoint, GraphNode } from '../engine/graph';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';
import { translatePort, translateValueName } from '../i18n';
import { parameterName } from './parameter-label';
import {
  NODE_W,
  SYMBOLS,
  bodyHeight,
  categoryOf,
  graphBounds,
  layoutGraph,
  nearestInput,
  nodePorts,
  sidePortY,
  type Point,
} from './graph-geometry';

const IO = ['input', 'output'];
const BROADCASTING = new Set(['add', 'sub', 'mul', 'div', 'max', 'ge']);

/** Shapes shown when a binary element-wise component expands mismatched operands. */
export function broadcastDisplayShapes(
  component: string,
  a?: readonly number[],
  b?: readonly number[],
  output?: readonly number[],
): { a: readonly number[]; b: readonly number[]; output: readonly number[] } | undefined {
  if (!BROADCASTING.has(component) || !a || !b || !output || a.join() === b.join())
    return undefined;
  return { a, b, output };
}

/** `[]` reads as "scalar"; anything else as `[4×2]`. */
export function shapeText(shape: readonly number[], t: CanvasProps['t']): string {
  return shape.length ? `[${shape.join('×')}]` : t('ui.scalar');
}

/** Fallback position for callers outside the canvas (e.g. placing an inserted expression). */
export const position = (n: GraphNode, i = 0) => ({
  x: Number(n.ui?.x ?? 60 + i * 190),
  y: Number(n.ui?.y ?? 100),
});

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}
export type NonfiniteKind = 'positive' | 'negative' | 'invalid';

export interface NodeValue {
  /** Short text for the node foot. */
  text: string;
  /** Full text for the tooltip. */
  title: string;
  ok: boolean;
  /** Distinguishes positive overflow, negative overflow and indeterminate NaN values. */
  nonfinite?: NonfiniteKind;
  /** Output shape and raw values; the side plots use them without parsing localized text. */
  shape?: readonly number[];
  values?: readonly number[];
  /** What the outgoing wires show: the value's sign and size (see `signal`). */
  signal?: number;
}

/** One number standing for a node's output: the value itself, or a tensor's mean. */
export function signal(values: readonly number[]): number | undefined {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((a, b) => a + b, 0) / finite.length : undefined;
}

/** The colour family for a failed numeric value; mixed infinities are indeterminate like NaN. */
export function nonfiniteKind(values: readonly number[]): NonfiniteKind | undefined {
  if (values.some(Number.isNaN)) return 'invalid';
  const positive = values.includes(Infinity);
  const negative = values.includes(-Infinity);
  if (positive && negative) return 'invalid';
  return positive ? 'positive' : negative ? 'negative' : undefined;
}

/**
 * Wires read like a neural-network diagram: warm for positive values, blue for negative, and
 * thicker the larger the value (log scale, so 0.01 and 1000 both stay readable).
 */
function wireStyle(v: number | undefined): { className: string; style?: CSSProperties } {
  if (v === undefined || v === 0) return { className: '' };
  const size = Math.min(1, Math.log10(1 + Math.abs(v)) / 1.3);
  return {
    className: v > 0 ? 'is-pos' : 'is-neg',
    style: { strokeWidth: 1.6 + size * 3.4 },
  };
}

/**
 * Textbook neural-net wire geometry.
 *
 * Prefer a pure straight segment (like a diagram in a textbook). Use a very light cubic arc
 * only when a straight line would tangle badly:
 *   - backward / leftward connections (wrap around neighbouring nodes), or
 *   - forward connections whose |Δy| is steep vs the horizontal gap (steep diagonals that
 *     knife through other nodes). Threshold: |Δy| > 0.85·Δx and |Δy| > 40.
 * The arc keeps control points near the endpoints so it stays near-straight.
 */
export function wireNeedsArc(a: Point, b: Point): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return dx <= 0 || (Math.abs(dy) > dx * 0.85 && Math.abs(dy) > 40);
}

export function wirePath(a: Point, b: Point): string {
  if (!wireNeedsArc(a, b)) return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
  const dx = b.x - a.x;
  // Light pull: ~20% of a forward gap, or a modest pad when wrapping backward.
  const pull = dx > 0 ? Math.max(24, dx * 0.2) : Math.max(40, -dx * 0.45);
  return `M ${a.x} ${a.y} C ${a.x + pull} ${a.y}, ${b.x - pull} ${b.y}, ${b.x} ${b.y}`;
}

/** Midpoint and tangent of a wire, where an overflowing value tears the line open. */
export function wireRupture(a: Point, b: Point): Point & { angle: number } {
  if (!wireNeedsArc(a, b)) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      angle: (Math.atan2(dy, dx) * 180) / Math.PI,
    };
  }
  const dx = b.x - a.x;
  const pull = dx > 0 ? Math.max(24, dx * 0.2) : Math.max(40, -dx * 0.45);
  const c1 = { x: a.x + pull, y: a.y };
  const c2 = { x: b.x - pull, y: b.y };
  const x = (a.x + 3 * c1.x + 3 * c2.x + b.x) / 8;
  const y = (a.y + 3 * c1.y + 3 * c2.y + b.y) / 8;
  const tx = 0.75 * (c1.x - a.x + c2.x - c1.x + b.x - c2.x);
  const ty = 0.75 * (c1.y - a.y + c2.y - c1.y + b.y - c2.y);
  return { x, y, angle: (Math.atan2(ty, tx) * 180) / Math.PI };
}
export interface CanvasHandle {
  /** Converts browser client coordinates to graph coordinates, or null when outside the canvas. */
  toGraph(clientX: number, clientY: number): Point | null;
  fit(): void;
}
export interface CanvasProps {
  registry: ComponentRegistry;
  graph: GraphJSON;
  selected: string[];
  readonly?: boolean;
  viewport: Viewport;
  onViewport: (view: Viewport) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  errors: { node?: string; port?: string; message: string; soft?: boolean }[];
  /** Probe results by node id. When null, values are not being probed and every wire is drawn live. */
  values: Record<string, NodeValue> | null;
  /** Shapes required by the level for its input and output nodes, by node id. */
  ioShapes?: Record<string, readonly number[]>;
  onSelect: (ids: string[], additive: boolean) => void;
  onMove: (positions: Record<string, Point>) => void;
  onConnect: (from: GraphEndpoint, to: GraphEndpoint, previous?: GraphEndpoint) => void;
  onDeleteEdge: (index: number) => void;
  onDeleteNode: (id: string) => void;
  onInspect: (node: GraphNode) => void;
  /**
   * One training step played back on the canvas: values flow forward along the wires, then
   * gradients flow back, then parameters change. Text is preformatted by the caller.
   */
  flow?: {
    phase: 'forward' | 'backward' | 'update';
    gradient: Record<string, string>;
    update: Record<string, string>;
  };
  /** Bumped by the parent to make a node shake (rejected connection). */
  shake?: { node: string; tick: number };
}

/** A wire being dragged; `snap` is the input port its loose end has jumped onto. */
type PendingWire = { from: GraphEndpoint; previous?: GraphEndpoint; snap?: GraphEndpoint } & Point;

const Canvas = forwardRef<CanvasHandle, CanvasProps>(function Canvas(p, ref) {
  const registry = p.registry;
  const root = useRef<HTMLDivElement>(null);
  const view = p.viewport,
    setView = p.onViewport;
  const [preview, setPreview] = useState<Record<string, Point>>({});
  const [wire, setWire] = useState<PendingWire>();
  const [box, setBox] = useState<{ x: number; y: number; w: number; h: number }>();
  /** Ports that just connected: brief flash, then cleared. Key = `node:port:in|out`. */
  const [flashPorts, setFlashPorts] = useState<ReadonlySet<string>>(() => new Set());
  const flashTimers = useRef<Map<string, number>>(new Map());
  const space = useRef(false);
  const stopTracking = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      stopTracking.current?.();
      for (const id of flashTimers.current.values()) window.clearTimeout(id);
      flashTimers.current.clear();
    },
    [],
  );

  const flashPort = (node: string, port: string, side: 'in' | 'out') => {
    const key = `${node}:${port}:${side}`;
    setFlashPorts((prev) => {
      const next = new Set(prev);
      next.add(key);
      return next;
    });
    const prior = flashTimers.current.get(key);
    if (prior !== undefined) window.clearTimeout(prior);
    flashTimers.current.set(
      key,
      window.setTimeout(() => {
        flashTimers.current.delete(key);
        setFlashPorts((prev) => {
          if (!prev.has(key)) return prev;
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
      }, 420),
    );
  };

  const connect = (from: GraphEndpoint, to: GraphEndpoint, previous?: GraphEndpoint) => {
    flashPort(from.node, from.port, 'out');
    flashPort(to.node, to.port, 'in');
    p.onConnect(from, to, previous);
  };

  const layout = useMemo(() => layoutGraph(p.graph, registry), [p.graph, registry]);
  const pos = (n: GraphNode): Point => preview[n.id] ?? layout.get(n.id) ?? { x: 0, y: 0 };
  const point = (x: number, y: number) => {
    const r = root.current!.getBoundingClientRect();
    return { x: (x - r.left - view.x) / view.zoom, y: (y - r.top - view.y) / view.zoom };
  };

  // React registers wheel listeners as passive, so preventDefault there cannot stop the page from
  // scrolling or the browser from zooming. Listen natively, non-passive, on the whole canvas
  // (nodes and wires included). ctrlKey marks a trackpad pinch on macOS/Chrome.
  const wheelZoom = useRef<(e: WheelEvent) => void>(() => {});
  wheelZoom.current = (e) => {
    const a = point(e.clientX, e.clientY);
    const scale = e.ctrlKey ? 0.01 : 0.001;
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const zoom = Math.max(0.25, Math.min(2, view.zoom * Math.exp(-e.deltaY * unit * scale)));
    const r = root.current!.getBoundingClientRect();
    setView({ zoom, x: e.clientX - r.left - a.x * zoom, y: e.clientY - r.top - a.y * zoom });
  };
  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      wheelZoom.current(e);
    };
    // Safari trackpad pinch arrives as gesture events rather than ctrl+wheel.
    const stopGesture = (e: Event) => e.preventDefault();
    element.addEventListener('wheel', onWheel, { passive: false });
    element.addEventListener('gesturestart', stopGesture);
    element.addEventListener('gesturechange', stopGesture);
    return () => {
      element.removeEventListener('wheel', onWheel);
      element.removeEventListener('gesturestart', stopGesture);
      element.removeEventListener('gesturechange', stopGesture);
    };
  }, []);

  useImperativeHandle(ref, () => ({
    toGraph(clientX, clientY) {
      const r = root.current?.getBoundingClientRect();
      if (!r) return null;
      if (clientX < r.left || clientX > r.right || clientY < r.top || clientY > r.bottom)
        return null;
      return point(clientX, clientY);
    },
    fit() {
      const b = graphBounds(p.graph, layout, registry);
      const r = root.current?.getBoundingClientRect();
      if (!b || !r) return;
      const margin = 48;
      const zoom = Math.max(
        0.25,
        Math.min(
          1.25,
          (r.width - margin * 2) / (b.x1 - b.x0),
          (r.height - margin * 2) / (b.y1 - b.y0),
        ),
      );
      setView({
        zoom,
        x: (r.width - (b.x1 - b.x0) * zoom) / 2 - b.x0 * zoom,
        y: (r.height - (b.y1 - b.y0) * zoom) / 2 - b.y0 * zoom,
      });
    },
  }));

  const ports = (n: GraphNode, side: 'inputs' | 'outputs') => nodePorts(n, registry, side);
  const portOffset = (n: GraphNode, side: 'inputs' | 'outputs', port: string) => {
    const list = ports(n, side);
    return sidePortY(n, registry, Math.max(0, list.indexOf(port)), list.length);
  };
  const endpoint = (e: GraphEndpoint, output: boolean): Point => {
    const n = p.graph.nodes.find((n) => n.id === e.node);
    if (!n) return { x: 0, y: 0 };
    const a = pos(n);
    return {
      x: a.x + (output ? NODE_W : 0),
      y: a.y + portOffset(n, output ? 'outputs' : 'inputs', e.port),
    };
  };
  const path = wirePath;
  const rupture = wireRupture;

  function track(
    e: ReactPointerEvent,
    move: (e: PointerEvent) => void,
    end: (e: PointerEvent) => void,
    cancel: () => void = () => {},
  ) {
    e.preventDefault();
    stopTracking.current?.();
    const cleanup = () => {
      window.removeEventListener('pointermove', moving);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', cancelled);
      window.removeEventListener('blur', abort);
      stopTracking.current = null;
    };
    const moving = (event: PointerEvent) => {
      if (event.pointerId === e.pointerId) move(event);
    };
    const up = (event: PointerEvent) => {
      if (event.pointerId !== e.pointerId) return;
      cleanup();
      end(event);
    };
    const abort = () => {
      cleanup();
      cancel();
    };
    const cancelled = (event: PointerEvent) => {
      if (event.pointerId === e.pointerId) abort();
    };
    stopTracking.current = abort;
    window.addEventListener('pointermove', moving);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', cancelled);
    window.addEventListener('blur', abort);
  }

  function startWire(e: ReactPointerEvent, from: GraphEndpoint, previous?: GraphEndpoint) {
    e.stopPropagation();
    if (p.readonly || e.button !== 0) return;
    root.current?.focus();
    // Within this many screen pixels of an input port, the wire end jumps onto it.
    const SNAP_PX = 28;
    const aim = (clientX: number, clientY: number): PendingWire => {
      const at = point(clientX, clientY);
      const hit = nearestInput(p.graph, registry, layout, at, SNAP_PX / view.zoom, from.node);
      return hit
        ? { from, previous, snap: { node: hit.node, port: hit.port }, ...hit.point }
        : { from, previous, ...at };
    };
    let current = aim(e.clientX, e.clientY);
    setWire(current);
    track(
      e,
      (event) => {
        current = aim(event.clientX, event.clientY);
        setWire(current);
      },
      (event) => {
        setWire(undefined);
        const target = document
          .elementFromPoint(event.clientX, event.clientY)
          ?.closest<HTMLElement>('[data-node]');
        const inside = target && root.current?.contains(target) ? target : undefined;
        // An exact drop on a port wins; otherwise the snapped port; otherwise a node body.
        if (inside?.dataset.input) {
          connect(from, { node: inside.dataset.node!, port: inside.dataset.input }, previous);
          return;
        }
        if (current.snap) {
          connect(from, current.snap, previous);
          return;
        }
        if (!inside || inside.dataset.output) return;
        const id = inside.dataset.node!;
        const n = p.graph.nodes.find((n) => n.id === id);
        // Dropping on a node body connects to its first free input.
        const port = n
          ? ports(n, 'inputs').find(
              (port) => !p.graph.edges.some((edge) => edge.to.node === id && edge.to.port === port),
            )
          : undefined;
        if (port) connect(from, { node: id, port }, previous);
      },
      () => setWire(undefined),
    );
  }

  const live = (id: string) => p.values === null || !!p.values[id]?.ok;
  const shaking = p.shake;

  return (
    <div
      ref={root}
      className="canvas blueprint-bg"
      tabIndex={0}
      style={{
        // The blueprint grid lives in graph space: it pans and scales with the nodes.
        backgroundPosition: `${view.x}px ${view.y}px`,
        backgroundSize: [100, 100, 20, 20]
          .map((size) => `${size * view.zoom}px ${size * view.zoom}px`)
          .join(', '),
      }}
      onKeyDown={(e) => {
        if (e.code === 'Space') {
          space.current = true;
          e.preventDefault();
        }
      }}
      onKeyUp={(e) => {
        if (e.code === 'Space') space.current = false;
      }}
      onBlur={() => (space.current = false)}
      onPointerDown={(e) => {
        if (e.button !== 0 && e.button !== 1) return;
        const onWorld =
          e.target === e.currentTarget ||
          (e.target as Element).closest('.graph-world') === e.target;
        if (!onWorld && !space.current && e.button !== 1) return;
        root.current?.focus();
        const sx = e.clientX,
          sy = e.clientY;
        const a = point(sx, sy);
        if (e.shiftKey && !space.current && e.button === 0) {
          track(
            e,
            (event) => {
              const b = point(event.clientX, event.clientY);
              setBox({
                x: Math.min(a.x, b.x),
                y: Math.min(a.y, b.y),
                w: Math.abs(a.x - b.x),
                h: Math.abs(a.y - b.y),
              });
            },
            (event) => {
              const b = point(event.clientX, event.clientY);
              const [x0, x1] = [Math.min(a.x, b.x), Math.max(a.x, b.x)];
              const [y0, y1] = [Math.min(a.y, b.y), Math.max(a.y, b.y)];
              const ids = p.graph.nodes
                .filter((n) => {
                  const q = pos(n);
                  const h =
                    root.current?.querySelector<HTMLElement>(
                      `[data-node="${CSS.escape(n.id)}"].node`,
                    )?.offsetHeight ?? 90;
                  return q.x <= x1 && q.x + NODE_W >= x0 && q.y <= y1 && q.y + h >= y0;
                })
                .map((n) => n.id);
              p.onSelect([...new Set([...p.selected, ...ids])], false);
              setBox(undefined);
            },
            () => setBox(undefined),
          );
        } else {
          track(
            e,
            (event) =>
              setView({ ...view, x: view.x + event.clientX - sx, y: view.y + event.clientY - sy }),
            (event) => {
              if (!space.current && e.button === 0 && event.clientX === sx && event.clientY === sy)
                p.onSelect([], false);
            },
          );
        }
      }}
    >
      <div
        className="graph-world"
        style={{ transform: `translate(${view.x}px,${view.y}px) scale(${view.zoom})` }}
      >
        <svg className="wires">
          {p.graph.edges.map((edge, i) => {
            const a = endpoint(edge.from, true),
              b = endpoint(edge.to, false);
            const d = path(a, b);
            const value = p.values?.[edge.from.node];
            const sign = wireStyle(value?.signal);
            // A nonfinite value tears its wire open; colour preserves +∞, −∞ and NaN semantics.
            const nonfinite = value?.ok === false ? (value.nonfinite ?? 'invalid') : undefined;
            const breakAt = nonfinite ? rupture(a, b) : undefined;
            return (
              <g key={`${edge.from.node}:${edge.from.port}>${edge.to.node}:${edge.to.port}`}>
                <path
                  className="wire-hit"
                  d={d}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={() => !p.readonly && p.onDeleteEdge(i)}
                />
                <path
                  style={!nonfinite && live(edge.from.node) ? sign.style : undefined}
                  pathLength={nonfinite ? 100 : undefined}
                  className={`wire ${nonfinite ? `is-burst is-overflow-${nonfinite}` : ''} ${live(edge.from.node) ? `is-live ${sign.className}` : ''} ${
                    p.flow?.phase === 'forward'
                      ? 'is-flow-forward'
                      : p.flow?.phase === 'backward' &&
                          p.flow.gradient[edge.from.node] &&
                          p.flow.gradient[edge.to.node]
                        ? 'is-flow-backward'
                        : ''
                  }`}
                  d={d}
                />
                {/* Faint dashed flow on top of the solid wire; step playback still uses the classes above. */}
                {!nonfinite && live(edge.from.node) && (
                  <path
                    className={`wire-flow ${
                      p.flow?.phase === 'forward'
                        ? 'is-flow-forward'
                        : p.flow?.phase === 'backward' &&
                            p.flow.gradient[edge.from.node] &&
                            p.flow.gradient[edge.to.node]
                          ? 'is-flow-backward'
                          : 'is-flow-ambient'
                    }`}
                    d={d}
                  />
                )}
                {breakAt && (
                  <g
                    className={`wire-rupture is-overflow-${nonfinite}`}
                    transform={`translate(${breakAt.x} ${breakAt.y}) rotate(${breakAt.angle})`}
                  >
                    <path d="M -18 0 C -12 0 -10 -2 -7 -7 L -2 -2" />
                    <path d="M 18 0 C 12 0 10 2 7 7 L 2 2" />
                    <line x1={-5} y1={-10} x2={-3} y2={-15} />
                    <line x1={5} y1={10} x2={3} y2={15} />
                    <line x1={0} y1={-8} x2={1} y2={-12} />
                  </g>
                )}
              </g>
            );
          })}
          {wire && (
            <path
              className={`wire is-pending ${wire.snap ? 'is-snapped' : ''}`}
              d={path(endpoint(wire.from, true), wire)}
            />
          )}
        </svg>
        {p.graph.nodes.map((n) => {
          const a = pos(n);
          const notes = p.errors.filter((e) => e.node === n.id);
          const errors = notes.filter((e) => !e.soft);
          const io = IO.includes(n.component);
          const value = p.values?.[n.id];
          const name = io ? String(n.attrs?.name ?? '') : (SYMBOLS[n.component] ?? '');
          const sym =
            parameterName(n) ??
            (n.component === 'param' && Array.isArray(n.attrs?.shape)
              ? `θ ${(n.attrs!.shape as number[]).join('×') || p.t('ui.scalar')}`
              : n.component === 'const'
                ? Array.isArray(n.attrs?.value)
                  ? `[${(n.attrs!.value as number[]).length}]`
                  : typeof n.attrs?.value === 'number'
                    ? String(Number(n.attrs.value.toPrecision(4)))
                    : String(n.attrs?.value ?? '')
                : n.component === 'output'
                  ? translateValueName(p.t, name)
                  : name);
          const displayShape =
            value?.shape ??
            (parameterName(n) && Array.isArray(n.attrs?.shape)
              ? (n.attrs.shape as number[])
              : undefined);
          const classes = [
            'node',
            p.selected.includes(n.id) && 'is-selected',
            errors.length && 'is-error',
            io && 'is-io',
            registry.get(n.component)?.subgraph && 'is-composite',
            shaking?.node === n.id && 'is-shaking',
            p.flow?.phase === 'update' && p.flow.update[n.id] && 'is-updating',
          ]
            .filter(Boolean)
            .join(' ');
          return (
            <div
              key={shaking?.node === n.id ? `${n.id}:${shaking.tick}` : n.id}
              data-node={n.id}
              data-category={categoryOf(n.component)}
              className={classes}
              style={{ left: a.x, top: a.y, width: NODE_W }}
              onDoubleClick={() => p.onInspect(n)}
              onPointerDown={(e) => {
                if (space.current || e.button === 1) return;
                if (e.button !== 0) return;
                if ((e.target as HTMLElement).closest('.node-delete')) return;
                e.stopPropagation();
                root.current?.focus();
                if (e.shiftKey || !p.selected.includes(n.id)) p.onSelect([n.id], e.shiftKey);
                // Shift-click toggles selection only; do not drag the stale selection.
                if (p.readonly || e.shiftKey) return;
                const ids = p.selected.includes(n.id) ? p.selected : [n.id];
                const start = point(e.clientX, e.clientY);
                let next: Record<string, Point> = {};
                track(
                  e,
                  (event) => {
                    const b = point(event.clientX, event.clientY);
                    next = Object.fromEntries(
                      p.graph.nodes
                        .filter((n) => ids.includes(n.id))
                        .map((n) => {
                          const a = pos(n);
                          return [n.id, { x: a.x + b.x - start.x, y: a.y + b.y - start.y }];
                        }),
                    );
                    setPreview(next);
                  },
                  () => {
                    setPreview({});
                    if (Object.keys(next).length) p.onMove(next);
                  },
                  () => setPreview({}),
                );
              }}
            >
              <div className="node-head">
                <span className="node-name">
                  {p.t(`components.${n.component}`)}
                  {io && p.ioShapes?.[n.id] && (
                    <span className="node-shape"> · {shapeText(p.ioShapes[n.id], p.t)}</span>
                  )}
                </span>
                {!io && !p.readonly && (
                  <button
                    className="icon-btn node-delete"
                    aria-label={p.t('ui.delete')}
                    title={p.t('ui.delete')}
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onDeleteNode(n.id);
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
              <div className="node-body" style={{ height: bodyHeight(n, registry) }}>
                <span className={`node-sym ${sym.length > 3 ? 'is-long' : ''}`}>{sym}</span>
              </div>
              <div
                className={`node-foot ${
                  value
                    ? value.ok
                      ? 'is-ok'
                      : `is-bad is-overflow-${value.nonfinite ?? 'invalid'}`
                    : ''
                }`}
                title={value?.title}
              >
                {displayShape && displayShape.length > 0 && (
                  <span className="node-shape-tag">{shapeText(displayShape, p.t)}</span>
                )}
                <span className="node-value">{value?.text ?? '—'}</span>
              </div>
              {p.flow && p.flow.phase !== 'forward' && p.flow.gradient[n.id] && (
                <div className="node-grad" title={p.t('ui.gradHelp')}>
                  <span className="node-grad-tag">∂</span>
                  <span className="node-value">{p.flow.gradient[n.id]}</span>
                </div>
              )}
              {p.flow?.phase === 'update' && p.flow.update[n.id] && (
                <div className="node-update">{p.flow.update[n.id]}</div>
              )}
              {(() => {
                // Element-wise ops with mismatched operand shapes: say out loud that broadcasting
                // stretched them, so [4×1]·[1×3] → [4×3] doesn't look like a matrix product.
                const operands = ['a', 'b'].map((port) => {
                  const edge = p.graph.edges.find((e) => e.to.node === n.id && e.to.port === port);
                  return edge && p.values?.[edge.from.node]?.shape;
                });
                const broadcast = broadcastDisplayShapes(
                  n.component,
                  operands[0],
                  operands[1],
                  value?.shape,
                );
                if (!broadcast) return null;
                const dims = (s: readonly number[]) => (s.length ? s.join('×') : p.t('ui.scalar'));
                return (
                  <div className="node-broadcast" title={p.t('ui.broadcastHelp')}>
                    <span className="node-broadcast-tag">{p.t('ui.broadcast')}</span>
                    {dims(broadcast.a)} · {dims(broadcast.b)} → {dims(broadcast.output)}
                  </div>
                );
              })()}
              {(['inputs', 'outputs'] as const).map((side) => {
                const list = ports(n, side);
                return list.map((port, i) => {
                  const y = sidePortY(n, registry, i, list.length);
                  const connected =
                    side === 'inputs'
                      ? p.graph.edges.some((e) => e.to.node === n.id && e.to.port === port)
                      : p.graph.edges.some((e) => e.from.node === n.id && e.from.port === port);
                  return (
                    <span key={side + port}>
                      {list.length > 1 && (
                        <span
                          className={`port-label ${side === 'inputs' ? 'is-in' : 'is-out'}`}
                          title={translatePort(p.t, n.component, port)}
                          style={{ top: y - 8 }}
                        >
                          {port}
                        </span>
                      )}
                      <span
                        data-node={n.id}
                        data-input={side === 'inputs' ? port : undefined}
                        data-output={side === 'outputs' ? port : undefined}
                        title={translatePort(p.t, n.component, port)}
                        className={[
                          'port',
                          side === 'inputs' ? 'is-in' : 'is-out',
                          connected && 'is-connected',
                          side === 'inputs' &&
                            wire?.snap?.node === n.id &&
                            wire.snap.port === port &&
                            'is-target',
                          flashPorts.has(`${n.id}:${port}:${side === 'inputs' ? 'in' : 'out'}`) &&
                            'is-flash',
                          errors.some((e) => e.port === port) && 'is-error',
                        ]
                          .filter(Boolean)
                          .join(' ')}
                        style={{ top: y - 6 }}
                        onPointerDown={(e) => {
                          if (space.current || e.button === 1) return;
                          e.stopPropagation();
                          if (side === 'outputs') startWire(e, { node: n.id, port });
                          else {
                            // Grabbing a connected input picks the wire up so it can be re-plugged.
                            const edge = p.graph.edges.find(
                              (edge) => edge.to.node === n.id && edge.to.port === port,
                            );
                            if (edge) startWire(e, edge.from, edge.to);
                          }
                        }}
                      />
                    </span>
                  );
                });
              })}
              {notes.map((e, i) => (
                <div className={e.soft ? 'node-note' : 'node-error'} key={i}>
                  {e.message}
                </div>
              ))}
            </div>
          );
        })}
        {box && (
          <div
            className="selection-box"
            style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
          />
        )}
      </div>
    </div>
  );
});
export default Canvas;
