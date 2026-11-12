// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * The sandbox's selected-part panel, floating over the canvas. Drag it by its header (mouse, pen
 * or touch), double-click the header to send it back to the top-right corner. It stays inside the
 * canvas area as that area resizes, and remembers where it was left.
 */
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import {
  DEFAULT_INSPECTOR_POS,
  clampInspectorPos,
  draggedInspectorPos,
  inspectorMaxHeight,
  readInspectorPos,
  writeInspectorPos,
  type InspectorPos,
  type Size,
} from './inspector-position';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export default function FloatingInspector({
  t,
  onClose,
  children,
}: {
  t: T;
  onClose: () => void;
  children: ReactNode;
}) {
  const panel = useRef<HTMLElement>(null);
  const [pos, setPos] = useState<InspectorPos>(readInspectorPos);
  const [area, setArea] = useState<Size>();
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ id: number; x: number; y: number; start: InspectorPos }>(undefined);
  const latest = useRef(pos);
  latest.current = pos;

  const measure = () => {
    const el = panel.current;
    const parent = el?.parentElement;
    if (!el || !parent) return undefined;
    // Natural height: the header plus everything in the body, scrolled or not.
    const head = el.firstElementChild as HTMLElement | null;
    const body = el.lastElementChild as HTMLElement | null;
    const h =
      (head?.offsetHeight ?? 0) + (body?.scrollHeight ?? 0) + el.offsetHeight - el.clientHeight;
    return {
      panel: { w: el.offsetWidth, h },
      area: { w: parent.clientWidth, h: parent.clientHeight },
    };
  };
  const place = (next: InspectorPos) => {
    const m = measure();
    if (!m) return next;
    const clamped = clampInspectorPos(next, m.panel, m.area);
    setArea(m.area);
    setPos(clamped);
    return clamped;
  };

  // Re-clamp whenever the canvas area (window, chart strip) or the panel's own content resizes.
  useLayoutEffect(() => {
    const el = panel.current;
    const parent = el?.parentElement;
    if (!el || !parent) return;
    place(latest.current);
    const observer = new ResizeObserver(() => {
      if (!drag.current) place(latest.current);
    });
    observer.observe(parent);
    observer.observe(el);
    const content = el.querySelector('.sb-inspector-content');
    if (content) observer.observe(content);
    return () => observer.disconnect();
  }, []);

  const end = (e: React.PointerEvent) => {
    if (drag.current?.id !== e.pointerId) return;
    drag.current = undefined;
    setDragging(false);
    writeInspectorPos(latest.current);
  };

  return (
    <aside
      ref={panel}
      className={`sb-inspector${dragging ? ' is-dragging' : ''}`}
      aria-label={t('sandbox.inspector.title')}
      style={{
        right: pos.right,
        top: pos.top,
        maxHeight: area ? inspectorMaxHeight(pos, area) : undefined,
      }}
    >
      <header
        className="sb-inspector-head"
        title={t('sandbox.inspector.dragHelp')}
        onPointerDown={(e) => {
          if (e.button !== 0 || (e.target as Element).closest('button')) return;
          e.preventDefault();
          e.stopPropagation();
          e.currentTarget.setPointerCapture(e.pointerId);
          drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, start: latest.current };
          setDragging(true);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (d?.id !== e.pointerId) return;
          latest.current = place(draggedInspectorPos(d.start, e.clientX - d.x, e.clientY - d.y));
        }}
        onPointerUp={end}
        onPointerCancel={end}
        onDoubleClick={(e) => {
          if ((e.target as Element).closest('button')) return;
          writeInspectorPos(undefined);
          place(DEFAULT_INSPECTOR_POS);
        }}
      >
        <h2 className="label">{t('sandbox.inspector.title')}</h2>
        <button
          className="icon-btn sb-inspector-close"
          aria-label={t('sandbox.close')}
          title={t('sandbox.inspector.close')}
          onClick={onClose}
        >
          ×
        </button>
      </header>
      <div className="sb-inspector-body">
        <div className="sb-inspector-content">{children}</div>
      </div>
    </aside>
  );
}
