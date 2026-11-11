// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * The sandbox's training tab: transport controls, the live picture of what the model does to
 * the data (fit curve or decision boundary), the loss curve and the knobs, top to bottom.
 * No pass line and no verdict: training simply runs for the chosen number of steps.
 */
import { useEffect, useMemo, useRef } from 'react';
import { formatNumber, type Locale } from '../i18n';
import DataPlot, { type PlotData } from './DataPlot';
import { lossSeries, plotAccuracy, type LossSeries } from '../sandbox/train-view';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

export const STEP_PRESETS = [100, 500, 2000, 10000] as const;

export interface SandboxTrainProps {
  t: T;
  locale: Locale;
  backend: string;
  training: 'idle' | 'running' | 'paused';
  /** Stepping or editing a component's insides: the transport waits. */
  locked: boolean;
  losses: number[];
  report?: { step: number; loss: number };
  steps: number;
  stepsRange: { min: number; max: number };
  learningRate: number;
  learningRateRange: { min: number; max: number };
  optimizer: 'sgd' | 'adam';
  plot?: PlotData;
  /** What the plot can show for this data. */
  view: 'fit' | 'boundary' | 'none';
  stepPhase?: string;
  format: (value: number) => string;
  onRun: () => void;
  onPauseResume: () => void;
  onStop: () => void;
  onStep: () => void;
  onReset: () => void;
  onSteps: (steps: number) => void;
  onLearningRate: (lr: number) => void;
  onOptimizer: (kind: 'sgd' | 'adam') => void;
}

export default function SandboxTrain(props: SandboxTrainProps) {
  const { t, locale, training, locked, losses, report, steps, plot, view, format } = props;
  const busy = training !== 'idle';
  const series = useMemo(() => lossSeries(losses), [losses]);
  const accuracy = view === 'boundary' ? plotAccuracy(plot) : undefined;
  const done = report ? Math.min(report.step, steps) : 0;
  return (
    <div className="sb-train">
      <div className="sb-transport" role="group" aria-label={t('sandbox.train.controls')}>
        {busy ? (
          <button className="btn btn-primary sb-run" onClick={props.onPauseResume}>
            {training === 'running' ? `❚❚ ${t('ui.pause')}` : `▶ ${t('ui.resume')}`}
          </button>
        ) : (
          <button className="btn btn-primary sb-run" disabled={locked} onClick={props.onRun}>
            ▶ {t('sandbox.train.run', { n: formatNumber(locale, steps) })}
          </button>
        )}
        {busy ? (
          <button className="btn btn-danger" onClick={props.onStop}>
            ■ {t('ui.stop')}
          </button>
        ) : (
          <button className="btn" disabled={locked} title={t('ui.stepHelp')} onClick={props.onStep}>
            ⏭ {t('sandbox.train.step')}
          </button>
        )}
        <button
          className="btn"
          disabled={busy || !losses.length}
          title={t('sandbox.train.resetHelp')}
          onClick={props.onReset}
        >
          ↺ {t('sandbox.train.reset')}
        </button>
      </div>
      <div className="sb-progress" aria-hidden={!report}>
        <div className="sb-progress-bar">
          <span style={{ width: `${steps ? (100 * done) / steps : 0}%` }} />
        </div>
        <span className="sb-progress-text">
          {report
            ? t('sandbox.train.progress', {
                step: formatNumber(locale, report.step),
                total: formatNumber(locale, steps),
              })
            : t('sandbox.train.idle')}
        </span>
      </div>
      {props.stepPhase && <p className="sb-phase">{props.stepPhase}</p>}

      <section className="sb-card" aria-label={t(`sandbox.train.view.${view}`)}>
        <header className="sb-card-head">
          <h3>{t(`sandbox.train.view.${view}`)}</h3>
          {accuracy !== undefined && (
            <span className="sb-metric">
              {t('ui.accuracy')}{' '}
              <b>
                {formatNumber(locale, accuracy, { style: 'percent', maximumFractionDigits: 1 })}
              </b>
            </span>
          )}
        </header>
        {view === 'none' || !plot ? (
          <p className="sb-empty-note">{t(`sandbox.train.view.${view}.empty`)}</p>
        ) : (
          <>
            <DataPlot data={plot} label={t(`sandbox.train.view.${view}`)} />
            <p className="sb-legend">{t(`sandbox.train.view.${view}.legend`)}</p>
          </>
        )}
      </section>

      <section className="sb-card" aria-label={t('ui.loss')}>
        <header className="sb-card-head">
          <h3>{t('ui.loss')}</h3>
          <span className={`sb-metric${series.diverged ? ' is-bad' : ''}`}>
            <b>{report ? format(report.loss) : '—'}</b>
          </span>
        </header>
        {losses.length ? (
          <SandboxLossChart series={series} label={t('ui.loss')} steps={steps} />
        ) : (
          <p className="sb-loss-empty">{t('sandbox.train.lossEmpty')}</p>
        )}
        {series.diverged && (
          <p className="sb-warn" role="status">
            {t('sandbox.train.diverged')}
          </p>
        )}
      </section>

      <section className="sb-settings" aria-label={t('sandbox.train.settings')}>
        <div className="sb-setting">
          <span className="sb-setting-name">{t('ui.optimizer')}</span>
          <div className="seg" role="radiogroup" aria-label={t('ui.optimizer')}>
            {(['adam', 'sgd'] as const).map((kind) => (
              <button
                key={kind}
                role="radio"
                aria-checked={kind === props.optimizer}
                className={`seg-btn${kind === props.optimizer ? ' is-on' : ''}`}
                disabled={busy}
                onClick={() => props.onOptimizer(kind)}
              >
                {t(`ui.optimizer.${kind}`)}
              </button>
            ))}
          </div>
        </div>
        <label className="sb-setting sb-setting-wide">
          <span className="sb-setting-name">{t('ui.learningRate')}</span>
          <output className="sb-setting-value">η = {format(props.learningRate)}</output>
          <input
            type="range"
            min={Math.log10(props.learningRateRange.min)}
            max={Math.log10(props.learningRateRange.max)}
            step={0.01}
            value={Math.log10(props.learningRate)}
            disabled={busy}
            onChange={(e) =>
              props.onLearningRate(Number(Number(10 ** Number(e.target.value)).toPrecision(2)))
            }
          />
        </label>
        <div className="sb-setting sb-setting-wide">
          <label className="sb-setting-name" htmlFor="sb-steps">
            {t('ui.trainSteps')}
          </label>
          <input
            id="sb-steps"
            className="field sb-steps-field"
            type="number"
            min={props.stepsRange.min}
            max={props.stepsRange.max}
            step={1}
            value={steps}
            disabled={busy}
            onChange={(e) => props.onSteps(Number(e.target.value))}
          />
          <div className="sb-presets">
            {STEP_PRESETS.map((n) => (
              <button
                key={n}
                className={`chip${n === steps ? ' is-on' : ''}`}
                aria-pressed={n === steps}
                disabled={busy}
                onClick={() => props.onSteps(n)}
              >
                {formatNumber(locale, n)}
              </button>
            ))}
          </div>
        </div>
        <p className="sb-backend">
          {t('ui.backend')} · {t(`backend.${props.backend}`)}
        </p>
      </section>
    </div>
  );
}

/**
 * Loss over steps. Log scale while every value is positive, linear otherwise (a hand-made loss
 * may go negative). Non-finite steps are marked along the top instead of breaking the curve.
 */
function SandboxLossChart({
  series,
  label,
  steps,
}: {
  series: LossSeries;
  label: string;
  steps: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const draw = () => {
      if (!canvas) return;
      const g = canvas.getContext('2d');
      if (!g) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const css = getComputedStyle(canvas);
      const color = (name: string) => css.getPropertyValue(name).trim();
      const pad = { l: 40, r: 10, t: 12, b: 20 };
      const { points, bad, lo, hi, log, count } = series;
      const n = Math.max(count, Math.min(steps, 50), 2);
      const sx = (i: number) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
      const f = (v: number) => (log ? Math.log10(v) : v);
      const flo = f(lo),
        fhi = f(hi);
      const sy = (v: number) =>
        pad.t + ((fhi - f(Math.min(Math.max(v, lo), hi))) / (fhi - flo || 1)) * (h - pad.t - pad.b);
      g.font = `10px ${color('--mono')}`;
      g.lineWidth = 1;
      const ticks = log
        ? Array.from({ length: Math.round(fhi - flo) + 1 }, (_, k) => 10 ** (flo + k))
        : [lo, (lo + hi) / 2, hi];
      const step = Math.ceil(ticks.length / 5);
      ticks.forEach((v, k) => {
        if (k % step) return;
        g.strokeStyle = color('--line');
        g.beginPath();
        g.moveTo(pad.l, sy(v));
        g.lineTo(w - pad.r, sy(v));
        g.stroke();
        g.fillStyle = color('--faint');
        const text = log
          ? Math.abs(Math.log10(v)) < 3
            ? String(Number(v.toPrecision(1)))
            : `1e${Math.round(Math.log10(v))}`
          : Number(v.toPrecision(2)).toString();
        g.fillText(text, 2, sy(v) + 3);
      });
      g.fillStyle = color('--faint');
      g.textAlign = 'right';
      g.fillText(String(count || 0), w - pad.r, h - 5);
      g.textAlign = 'left';
      g.fillText('0', pad.l, h - 5);
      if (bad.length) {
        g.fillStyle = color('--bad');
        for (const i of bad) g.fillRect(sx(i) - 1, pad.t - 8, 2, 5);
      }
      if (!points.length) return;
      g.strokeStyle = color('--accent');
      g.lineWidth = 1.8;
      g.beginPath();
      points.forEach(([i, v], k) => (k ? g.lineTo(sx(i), sy(v)) : g.moveTo(sx(i), sy(v))));
      g.stroke();
      const [li, lv] = points[points.length - 1];
      g.fillStyle = color('--accent');
      g.fillRect(sx(li) - 2.5, sy(lv) - 2.5, 5, 5);
    };
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas) observer.observe(canvas);
    return () => observer.disconnect();
  }, [series, steps]);
  return <canvas className="loss-chart sb-loss-chart" ref={ref} aria-label={label} role="img" />;
}
