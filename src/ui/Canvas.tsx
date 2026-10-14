import { useEffect, useRef, useState } from 'react';
import type { GraphJSON, GraphEndpoint, GraphNode } from '../engine/graph';
import { createExampleRegistry } from '../levels/examples';
import type { PointerEvent as ReactPointerEvent } from 'react';

const registry = createExampleRegistry();
export const position = (n: GraphNode, i = 0) => ({
  x: Number(n.ui?.x ?? 60 + i * 190),
  y: Number(n.ui?.y ?? 100),
});
export interface CanvasProps {
  graph: GraphJSON;
  selected: string[];
  readonly?: boolean;
  viewport: { x: number; y: number; zoom: number };
  onViewport: (view: { x: number; y: number; zoom: number }) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  errors: { node?: string; port?: string; message: string }[];
  values: Record<string, string>;
  onSelect: (ids: string[], additive: boolean) => void;
  onMove: (positions: Record<string, { x: number; y: number }>) => void;
  onConnect: (from: GraphEndpoint, to: GraphEndpoint, previous?: GraphEndpoint) => void;
  onDeleteEdge: (index: number) => void;
  onAdd: (component: string, x: number, y: number) => void;
  onInspect: (node: GraphNode) => void;
}
export default function Canvas(p: CanvasProps) {
  const root = useRef<HTMLDivElement>(null);
  const view = p.viewport,
    setView = p.onViewport;
  const [preview, setPreview] = useState<Record<string, { x: number; y: number }>>({});
  const [wire, setWire] = useState<{
    from: GraphEndpoint;
    previous?: GraphEndpoint;
    x: number;
    y: number;
  }>();
  const [box, setBox] = useState<{ x: number; y: number; w: number; h: number }>();
  const space = useRef(false);
  const stopTracking = useRef<(() => void) | null>(null);
  useEffect(() => () => stopTracking.current?.(), []);
  const point = (x: number, y: number) => {
    const r = root.current!.getBoundingClientRect();
    return { x: (x - r.left - view.x) / view.zoom, y: (y - r.top - view.y) / view.zoom };
  };
  const pos = (n: GraphNode) => preview[n.id] ?? position(n, p.graph.nodes.indexOf(n));
  const endpoint = (e: GraphEndpoint, output: boolean) => {
    const n = p.graph.nodes.find((n) => n.id === e.node);
    if (!n) return { x: 0, y: 0 };
    const ports = registry.get(n.component)?.[output ? 'outputs' : 'inputs'] ?? [];
    const a = pos(n);
    return { x: a.x + (output ? 150 : 0), y: a.y + 42 + Math.max(0, ports.indexOf(e.port)) * 22 };
  };
  const path = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    `M ${a.x} ${a.y} C ${a.x + 65} ${a.y}, ${b.x - 65} ${b.y}, ${b.x} ${b.y}`;
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
    setWire({ from, previous, ...point(e.clientX, e.clientY) });
    track(
      e,
      (event) => setWire({ from, previous, ...point(event.clientX, event.clientY) }),
      (event) => {
        setWire(undefined);
        const target = document
          .elementFromPoint(event.clientX, event.clientY)
          ?.closest<HTMLElement>('[data-node]');
        if (!target || !root.current?.contains(target) || target.dataset.output) return;
        const id = target.dataset.node!;
        let port = target.dataset.input;
        if (!port) {
          const n = p.graph.nodes.find((n) => n.id === id);
          port = registry
            .get(n?.component ?? '')
            ?.inputs.find(
              (port) => !p.graph.edges.some((edge) => edge.to.node === id && edge.to.port === port),
            );
        }
        if (port) p.onConnect(from, { node: id, port }, previous);
      },
      () => setWire(undefined),
    );
  }
  return (
    <div
      ref={root}
      className="canvas blueprint-bg"
      tabIndex={0}
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
      onWheel={(e) => {
        const a = point(e.clientX, e.clientY);
        const zoom = Math.max(0.25, Math.min(2, view.zoom * Math.exp(-e.deltaY * 0.001)));
        const r = root.current!.getBoundingClientRect();
        setView({ zoom, x: e.clientX - r.left - a.x * zoom, y: e.clientY - r.top - a.y * zoom });
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (p.readonly) return;
        const a = point(e.clientX, e.clientY);
        p.onAdd(e.dataTransfer.getData('application/x-component'), a.x, a.y);
      }}
      onPointerDown={(e) => {
        if (e.button !== 0 && e.button !== 1) return;
        if (e.target !== e.currentTarget && !space.current && e.button !== 1) return;
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
              const ids = p.graph.nodes
                .filter((n) => {
                  const q = pos(n);
                  return (
                    q.x <= Math.max(a.x, b.x) &&
                    q.x + 150 >= Math.min(a.x, b.x) &&
                    q.y <= Math.max(a.y, b.y) &&
                    q.y +
                      (root.current?.querySelector<HTMLElement>(
                        `[data-node="${CSS.escape(n.id)}"].node`,
                      )?.offsetHeight ?? 90) >=
                      Math.min(a.y, b.y)
                  );
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
            const d = path(endpoint(edge.from, true), endpoint(edge.to, false));
            return (
              <g key={i}>
                <path className="wire-hit" d={d} onClick={() => !p.readonly && p.onDeleteEdge(i)} />
                <path className="wire is-live" d={d} />
              </g>
            );
          })}
          {wire && <path className="wire is-pending" d={path(endpoint(wire.from, true), wire)} />}
        </svg>
        {p.graph.nodes.map((n) => {
          const def = registry.get(n.component);
          const a = pos(n);
          const errors = p.errors.filter((e) => e.node === n.id);
          const height = Math.max(
            84,
            60 + Math.max(def?.inputs.length ?? 0, def?.outputs.length ?? 0) * 22,
          );
          return (
            <div
              key={n.id}
              data-node={n.id}
              className={`node ${p.selected.includes(n.id) ? 'is-selected' : ''} ${errors.length ? 'is-error' : ''} ${['input', 'output'].includes(n.component) ? 'is-io' : ''}`}
              style={{ left: a.x, top: a.y, width: 150, minHeight: height }}
              onDoubleClick={() => p.onInspect(n)}
              onPointerDown={(e) => {
                if (space.current || e.button === 1) return;
                if (e.button !== 0) return;
                e.stopPropagation();
                root.current?.focus();
                if (e.shiftKey || !p.selected.includes(n.id)) p.onSelect([n.id], e.shiftKey);
                // Shift-click toggles selection only; do not drag the stale selection.
                if (p.readonly || e.shiftKey) return;
                const ids = p.selected.includes(n.id) ? p.selected : [n.id];
                const start = point(e.clientX, e.clientY);
                let next: Record<string, { x: number; y: number }> = {};
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
              <div className="node-head">{p.t(`components.${n.component}`)}</div>
              <div className="node-body">
                {['input', 'output'].includes(n.component)
                  ? String(n.attrs?.name ?? '')
                  : n.component === 'const'
                    ? JSON.stringify(n.attrs?.value)
                    : ''}
              </div>
              {(['inputs', 'outputs'] as const).map((side) =>
                def?.[side].map((port, i) => (
                  <span key={side + port}>
                    <span
                      className="port-label"
                      style={{ top: 34 + i * 22, [side === 'inputs' ? 'left' : 'right']: 10 }}
                    >
                      {p.t(`ports.${port}`)}
                    </span>
                    <span
                      data-node={n.id}
                      data-input={side === 'inputs' ? port : undefined}
                      data-output={side === 'outputs' ? port : undefined}
                      className={`port ${side === 'inputs' ? 'is-in' : 'is-out'} ${errors.some((e) => e.port === port) ? 'is-error' : ''}`}
                      style={{ top: 36 + i * 22 }}
                      onPointerDown={(e) => {
                        if (space.current || e.button === 1) return;
                        e.stopPropagation();
                        if (side === 'outputs') startWire(e, { node: n.id, port });
                        else {
                          const edge = p.graph.edges.find(
                            (edge) => edge.to.node === n.id && edge.to.port === port,
                          );
                          if (edge) startWire(e, edge.from, edge.to);
                        }
                      }}
                    />
                  </span>
                )),
              )}
              <div className="node-foot" style={{ marginTop: Math.max(12, height - 65) }}>
                {p.values[n.id] ?? ''}
              </div>
              {errors.map((e, i) => (
                <div className="node-error" key={i}>
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
}
