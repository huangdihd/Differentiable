// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { BackwardGraph, BackwardTrace } from '../engine/backward-graph';
import type { GraphJSON } from '../engine/graph';
import { translateValueName } from '../i18n';
import { SYMBOLS } from './graph-geometry';
import {
  backwardTeachingSegments,
  gradientEdgeKey,
  gradientLayout,
  namedMultiplicationRule,
} from './gradient-layout';
import { gradientParameterLabel } from './parameter-label';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const W = 136,
  H = 60,
  PITCH_X = 250,
  PITCH_Y = 78,
  PAD = 28,
  GAP = 92;

/**
 * "Where autodiff comes from": the forward graph on top (values flow →), its mirror below
 * (gradients flow ←). Each backward wire carries the local derivative of the node it leaves;
 * a value used twice collects two gradients through a "+".
 */
export default function GradientView({
  t,
  graph,
  backward,
  trace,
  wrt,
  output,
  playerBackwardStart,
  playerBackwardLabelKey,
  engineBackwardLabelKey,
  composite,
  format,
  intro,
  parameterState,
  attachedLoss,
  onClose,
}: {
  t: T;
  graph: GraphJSON;
  backward: BackwardGraph;
  trace?: BackwardTrace;
  wrt: readonly string[];
  output: string;
  playerBackwardStart?: string;
  playerBackwardLabelKey?: string;
  engineBackwardLabelKey?: string;
  composite: (component: string) => boolean;
  format: (n: number) => string;
  intro?: string;
  parameterState?: 'initial' | 'current';
  attachedLoss?: boolean;
  onClose: () => void;
}) {
  const layout = gradientLayout(graph, backward, wrt);
  const outputLabel = translateValueName(t, output);
  const teaching = playerBackwardStart
    ? backwardTeachingSegments(layout, playerBackwardStart)
    : undefined;
  const node = new Map(graph.nodes.map((n) => [n.id, n]));
  const label = (id: string) => {
    const n = node.get(id)!;
    if (n.attrs?.engineLoss === true)
      return t(n.component === 'const' ? 'gradView.targets' : 'components.cross_entropy');
    if (typeof n.attrs?.labelKey === 'string') return t(n.attrs.labelKey);
    if (n.component === 'param') return gradientParameterLabel(n, graph);
    if (n.component === 'output') return translateValueName(t, String(n.attrs?.name ?? id));
    if (n.component === 'input') return String(n.attrs?.name ?? id);
    if (n.component === 'const') {
      // A constant can be a whole tensor (e.g. 4-2's class ids); label it by length, as the canvas does.
      const v = n.attrs?.value;
      if (Array.isArray(v)) return `[${v.length}]`;
      return String(Number(Number(v).toPrecision(4)));
    }
    return SYMBOLS[n.component] ?? t(`components.${n.component}`);
  };
  const source = (id: string) => graph.edges.find((e) => e.to.node === id)?.from.node;
  const value = (kind: 'forward' | 'gradient', id: string) => {
    const n = node.get(id)!;
    if (kind === 'gradient' && n.component === 'output') return '1';
    const v = trace?.[kind][n.component === 'output' ? (source(id) ?? id) : id];
    if (!v) return '';
    const shown = v.values.slice(0, 3).map(format).join(', ') + (v.values.length > 3 ? ', …' : '');
    return v.shape.length ? `[${v.shape.join('×')}] ${shown}` : shown;
  };

  const top = PAD + 22;
  const bottom = top + layout.rows * PITCH_Y + GAP;
  const x = (column: number) => PAD + column * PITCH_X;
  const yF = (row: number) => top + row * PITCH_Y;
  const yB = (row: number) => bottom + row * PITCH_Y;
  const at = new Map(layout.nodes.map((n) => [n.id, n]));
  const width = PAD * 2 + (layout.columns - 1) * PITCH_X + W;
  const height = bottom + layout.rows * PITCH_Y + PAD;
  const backwardSegment = (from: string, to: string, port: string) => {
    const key = gradientEdgeKey(from, to, port);
    if (teaching?.playerEdges.has(key)) return ' is-player-step';
    if (teaching?.engineEdges.has(key)) return ' is-engine-step';
    return '';
  };

  const ruleText = (edge: (typeof layout.edges)[number]) => {
    const consumer = node.get(edge.to)!;
    if (consumer.component === 'output') return '';
    if (composite(consumer.component))
      return t('gradView.expanded', {
        name: SYMBOLS[consumer.component] ?? t(`components.${consumer.component}`),
      });
    if (
      edge.rule === 'backward.rule.mul' &&
      teaching?.playerEdges.has(gradientEdgeKey(edge.from, edge.to, edge.port))
    ) {
      const named = namedMultiplicationRule(graph, edge, label);
      if (named) return t('gradView.ruleMulNamed', named);
    }
    // Generic rules are written for the first operand; the second one sees the other side.
    if (edge.rule === 'backward.rule.mul' && edge.port === 'b') return t('gradView.ruleMulB');
    return edge.rule ? t(edge.rule) : '';
  };
  const curve = (x1: number, y1: number, x2: number, y2: number) => {
    const dx = Math.max(30, Math.abs(x2 - x1) / 2) * Math.sign(x2 - x1 || 1);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  };

  return (
    <div className="gv-back" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <section className="gv" role="dialog" aria-modal="true" aria-label={t('gradView.title')}>
        <header className="gv-head">
          <div>
            <h2>{t('gradView.title')}</h2>
            <p className="hint">
              {intro ?? t('gradView.intro')}
              {parameterState && (
                <>
                  {' '}
                  {t(`gradView.parameters.${parameterState}`)} {t('gradView.trainingData')}
                </>
              )}
              {attachedLoss && <> {t('gradView.attachedLoss')}</>}
            </p>
          </div>
          <button className="btn" onClick={onClose} autoFocus>
            {t('ui.close')}
          </button>
        </header>
        <div className="gv-legend">
          <span className="gv-key is-forward">{t('gradView.forward')}</span>
          <span className="gv-key is-backward">{t('gradView.backward', { y: outputLabel })}</span>
          {teaching && (
            <span className="gv-key is-player-step">{t('gradView.playerBackwardLegend')}</span>
          )}
          <span className="gv-key is-sum">{t('gradView.sum')}</span>
        </div>
        <div className="gv-scroll">
          <div className="gv-stage" style={{ width, height }}>
            <span className="gv-band" style={{ top: PAD - 4 }}>
              {t('gradView.forwardBand')}
            </span>
            <span className="gv-band is-backward" style={{ top: bottom - (teaching ? 54 : 26) }}>
              {t('gradView.backwardBand', { y: outputLabel })}
            </span>
            {teaching && playerBackwardLabelKey && (
              <span className="gv-region is-player" style={{ left: PAD, top: bottom - 29 }}>
                {t(playerBackwardLabelKey)}
              </span>
            )}
            {teaching && engineBackwardLabelKey && (
              <span className="gv-region is-engine" style={{ right: PAD, top: bottom - 29 }}>
                {t(engineBackwardLabelKey)}
              </span>
            )}
            <svg width={width} height={height} className="gv-wires">
              <defs>
                {(['forward', 'backward', 'player'] as const).map((k) => (
                  <marker
                    key={k}
                    id={`gv-arrow-${k}`}
                    viewBox="0 0 8 8"
                    refX="7"
                    refY="4"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto"
                  >
                    <path d="M0,0 L8,4 L0,8 z" className={`gv-head-${k}`} />
                  </marker>
                ))}
              </defs>
              {layout.nodes
                .filter((n) => n.carries)
                .map((n) => (
                  <line
                    key={`tie-${n.id}`}
                    className="gv-tie"
                    x1={x(n.column) + W / 2}
                    y1={yF(n.row) + H}
                    x2={x(n.column) + W / 2}
                    y2={yB(n.row)}
                  />
                ))}
              {layout.edges.map((e, i) => {
                const a = at.get(e.from)!,
                  b = at.get(e.to)!;
                return (
                  <path
                    key={`f${i}`}
                    className="gv-wire is-forward"
                    markerEnd="url(#gv-arrow-forward)"
                    d={curve(
                      x(a.column) + W,
                      yF(a.row) + H / 2,
                      x(b.column) - 2,
                      yF(b.row) + H / 2,
                    )}
                  />
                );
              })}
              {layout.edges
                .filter((e) => e.backward)
                .map((e, i) => {
                  const a = at.get(e.from)!,
                    b = at.get(e.to)!;
                  const sum = layout.sums[e.from];
                  const endX = x(a.column) + W + (sum ? 22 : 2);
                  const endY = yB(a.row) + H / 2;
                  const startX = x(b.column),
                    startY = yB(b.row) + H / 2;
                  const text = ruleText(e);
                  return (
                    <g key={`b${i}`}>
                      <path
                        className={`gv-wire is-backward${backwardSegment(e.from, e.to, e.port)}`}
                        markerEnd={
                          sum
                            ? undefined
                            : teaching?.playerEdges.has(gradientEdgeKey(e.from, e.to, e.port))
                              ? 'url(#gv-arrow-player)'
                              : 'url(#gv-arrow-backward)'
                        }
                        d={curve(startX, startY, endX, endY)}
                      />
                      {text && (
                        <text
                          className={`gv-rule${backwardSegment(e.from, e.to, e.port)}`}
                          x={(startX + endX) / 2}
                          y={(startY + endY) / 2 - 7}
                          textAnchor="middle"
                        >
                          {text}
                        </text>
                      )}
                    </g>
                  );
                })}
              {Object.keys(layout.sums).map((id) => {
                const n = at.get(id)!;
                const cx = x(n.column) + W + 22,
                  cy = yB(n.row) + H / 2;
                return (
                  <g key={`sum-${id}`} className="gv-sum">
                    <path
                      className="gv-wire is-backward"
                      markerEnd="url(#gv-arrow-backward)"
                      d={`M ${cx - 10} ${cy} L ${cx - 20} ${cy}`}
                    />
                    <circle cx={cx} cy={cy} r={10} />
                    <text x={cx} y={cy + 4} textAnchor="middle">
                      +
                    </text>
                  </g>
                );
              })}
            </svg>
            {layout.nodes.map((n) => (
              <div
                key={`fc-${n.id}`}
                className={`gv-card is-forward${n.carries ? '' : ' is-idle'}`}
                style={{ left: x(n.column), top: yF(n.row), width: W, height: H }}
              >
                <span className="gv-card-label">{label(n.id)}</span>
                <span className="gv-card-value">{value('forward', n.id)}</span>
              </div>
            ))}
            {layout.nodes
              .filter((n) => n.carries)
              .map((n) => (
                <div
                  key={`bc-${n.id}`}
                  className={`gv-card is-backward${
                    ['input', 'param'].includes(node.get(n.id)!.component) ? ' is-target' : ''
                  }${teaching?.start === n.id ? ' is-handoff' : ''}${
                    teaching?.playerNodes.has(n.id) ? ' is-player-step' : ''
                  }${
                    teaching?.engineNodes.has(n.id) && teaching.start !== n.id
                      ? ' is-engine-step'
                      : ''
                  }`}
                  style={{ left: x(n.column), top: yB(n.row), width: W, height: H }}
                >
                  <span className="gv-card-label">
                    {teaching?.start === n.id ? (
                      t('gradView.playerBackwardInput', { y: outputLabel, x: n.id })
                    ) : (
                      <>
                        ∂{outputLabel}/∂
                        {['input', 'param'].includes(node.get(n.id)!.component)
                          ? label(n.id)
                          : `(${label(n.id)})`}
                      </>
                    )}
                  </span>
                  <span className="gv-card-value">{value('gradient', n.id)}</span>
                  {teaching?.start === n.id && (
                    <span className="gv-card-note">{t('gradView.playerBackwardStart')}</span>
                  )}
                </div>
              ))}
          </div>
        </div>
      </section>
    </div>
  );
}
