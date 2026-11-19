// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { MathText } from '../MathText';
import { ExperimentSteps } from '../TemporalExperimentParts';

export type Translator = (
  key: string,
  params?: Readonly<Record<string, string | number>>,
) => string;

export interface PrimerLessonProps {
  levelId: string;
  code: string;
  t: Translator;
  onContinue: () => void;
}

/**
 * The "go on" button of a step (next / finish / continue / try again). It takes focus when it
 * appears, and Enter anywhere on the page presses it unless focus sits in a field or on another
 * button, which keep their own Enter.
 */
export function NextButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    button.current?.focus({ preventScroll: true });
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Enter' || event.repeat || event.isComposing || event.defaultPrevented)
        return;
      const active = document.activeElement;
      // The focused button handles Enter natively; fields and other buttons keep theirs.
      if (
        active === button.current ||
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        active instanceof HTMLSelectElement ||
        active instanceof HTMLButtonElement
      )
        return;
      event.preventDefault();
      button.current?.click();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return (
    <button ref={button} type="button" className="btn btn-primary" onClick={onClick}>
      {children}
    </button>
  );
}

/** One primer lesson: header, round tracker, the active task, and the takeaway once passed. */
export function PrimerShell({
  levelId,
  code,
  t,
  steps,
  current,
  done,
  onContinue,
  children,
  extra,
  badgeKey = 'primer.ui.badge',
}: PrimerLessonProps & {
  /** Header tag: the math warm-up by default; later-chapter experiments override it. */
  badgeKey?: string;
  steps: readonly string[];
  current: number;
  done: boolean;
  children: ReactNode;
  /** Shown under the takeaway once passed, e.g. a free-play toy. */
  extra?: ReactNode;
}) {
  return (
    <main className="experiment blueprint-bg primer">
      <div className="experiment-inner">
        <header className="experiment-head">
          <span className="level-id">
            LEVEL {code} · {t(badgeKey)}
          </span>
          <h1 className="level-title">{t(`levels.${levelId}.title`)}</h1>
          <div className="experiment-lede">
            <MathText text={t(`levels.${levelId}.description`)} />
          </div>
        </header>
        <ExperimentSteps label={t('primer.ui.progress')} steps={steps} current={current} />
        {!done && children}
        {done && (
          <section className="experiment-takeaway" aria-live="polite">
            <h2>{t('primer.ui.takeawayTitle')}</h2>
            <p>{t(`primer.ui.${levelId}.takeaway`)}</p>
            {extra}
            <div className="experiment-prompt">
              <p className="experiment-meta">{t('primer.ui.next')}</p>
              <NextButton onClick={onContinue}>{t('primer.ui.continue')} →</NextButton>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

/** A panel holding one round's task. */
export function PrimerPanel({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="experiment-panel primer-panel">
      <div className="experiment-panel-head">
        <h2 className="label">{title}</h2>
        {meta && <span className="experiment-meta">{meta}</span>}
      </div>
      <div className="primer-body">{children}</div>
    </section>
  );
}

export type Verdict = 'good' | 'bad' | undefined;

/** One line of feedback under a task; colour follows the verdict. */
export function Feedback({ verdict, children }: { verdict: Verdict; children: ReactNode }) {
  if (!children) return null;
  return (
    <p className={`primer-feedback ${verdict ? `is-${verdict}` : ''}`} aria-live="polite">
      {children}
    </p>
  );
}

/** Parses what a player types: accepts a Unicode minus and a comma decimal. */
export function parseNumber(text: string): number | undefined {
  const normalized = text.trim().replace(/[−–]/g, '-').replace(',', '.');
  if (!normalized) return undefined;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : undefined;
}

/** Short, stable display of a value: at most `digits` decimals, no trailing zeros. */
export function formatNumber(value: number, digits = 3): string {
  const rounded = Number(value.toFixed(digits));
  const text = String(Object.is(rounded, -0) ? 0 : rounded);
  return text.replace('-', '−');
}

/** A numeric answer box with a submit button; Enter submits. */
export function NumberAnswer({
  label,
  submitLabel,
  disabled,
  invalid,
  onSubmit,
}: {
  label: string;
  submitLabel: string;
  disabled?: boolean;
  invalid?: boolean;
  onSubmit: (value: number) => void;
}) {
  const [text, setText] = useState('');
  const value = parseNumber(text);
  const submit = () => {
    if (value === undefined || disabled) return;
    onSubmit(value);
  };
  return (
    <form
      className="primer-answer"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <label>
        <span className="label">{label}</span>
        <input
          className="field"
          inputMode="decimal"
          autoComplete="off"
          value={text}
          disabled={disabled}
          aria-invalid={invalid || undefined}
          onChange={(event) => setText(event.target.value)}
        />
      </label>
      <button type="submit" className="btn btn-primary" disabled={disabled || value === undefined}>
        {submitLabel}
      </button>
    </form>
  );
}

/** A compact answer box that sits inside a figure or equation; Enter or ✓ submits. */
export function InlineAnswer({
  label,
  submitLabel,
  invalid,
  onSubmit,
}: {
  label: string;
  submitLabel: string;
  invalid: boolean;
  onSubmit: (value: number) => void;
}) {
  const [text, setText] = useState('');
  const value = parseNumber(text);
  return (
    <form
      className="primer-inline-answer"
      onSubmit={(event) => {
        event.preventDefault();
        if (value !== undefined) onSubmit(value);
      }}
    >
      <input
        className="field"
        inputMode="decimal"
        autoComplete="off"
        autoFocus
        aria-label={label}
        placeholder="?"
        value={text}
        aria-invalid={invalid || undefined}
        onChange={(event) => setText(event.target.value)}
      />
      <button
        type="submit"
        className="btn btn-primary"
        aria-label={submitLabel}
        title={submitLabel}
        disabled={value === undefined}
      >
        ✓
      </button>
    </form>
  );
}

/** Maps data coordinates into an SVG viewBox. */
export interface Viewport {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  width: number;
  height: number;
}
export const sx = (v: Viewport, x: number) => ((x - v.x0) / (v.x1 - v.x0)) * v.width;
export const sy = (v: Viewport, y: number) => v.height - ((y - v.y0) / (v.y1 - v.y0)) * v.height;
export const dataX = (v: Viewport, px: number) => v.x0 + (px / v.width) * (v.x1 - v.x0);
export const dataY = (v: Viewport, py: number) =>
  v.y0 + ((v.height - py) / v.height) * (v.y1 - v.y0);

/** An SVG polyline path for f over [from, to], sampled densely enough to look smooth. */
export function curvePath(
  v: Viewport,
  f: (x: number) => number,
  from = v.x0,
  to = v.x1,
  samples = 240,
): string {
  const span = v.y1 - v.y0;
  const points: string[] = [];
  for (let i = 0; i <= samples; i += 1) {
    const x = from + ((to - from) * i) / samples;
    // Clamp far-off values so the path stays finite; the clip box hides the excess.
    const y = Math.min(v.y1 + span, Math.max(v.y0 - span, f(x)));
    points.push(`${i ? 'L' : 'M'}${sx(v, x).toFixed(2)},${sy(v, y).toFixed(2)}`);
  }
  return points.join('');
}

/** Light background grid and axes for a plot, with the zero lines when they are in view. */
export function PlotGrid({ v, step, yStep = step }: { v: Viewport; step: number; yStep?: number }) {
  const lines: ReactNode[] = [];
  const first = (start: number, size = step) => Math.ceil(start / size) * size;
  for (let x = first(v.x0); x <= v.x1; x += step)
    lines.push(
      <line
        key={`x${x}`}
        className={Math.abs(x) < step / 2 ? 'primer-axis' : 'primer-gridline'}
        x1={sx(v, x)}
        x2={sx(v, x)}
        y1={0}
        y2={v.height}
      />,
    );
  for (let y = first(v.y0, yStep); y <= v.y1; y += yStep)
    lines.push(
      <line
        key={`y${y}`}
        className={Math.abs(y) < yStep / 2 ? 'primer-axis' : 'primer-gridline'}
        x1={0}
        x2={v.width}
        y1={sy(v, y)}
        y2={sy(v, y)}
      />,
    );
  return <g aria-hidden="true">{lines}</g>;
}

/** A grid step that gives roughly 6–12 lines across `span`. */
export function niceStep(span: number): number {
  const raw = span / 8;
  const power = 10 ** Math.floor(Math.log10(raw));
  const scaled = raw / power;
  return (scaled < 1.5 ? 1 : scaled < 3.5 ? 2 : scaled < 7.5 ? 5 : 10) * power;
}
