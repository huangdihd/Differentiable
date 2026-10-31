// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { vocabulary } from '../levels/corpus';
import type { FixedWindowPrediction } from '../worker/fixed-window-protocol';
import type { FixedWindowExperimentState } from './use-fixed-window-experiment';

/** Shared building blocks for the chapter-5 character experiments. */
export const characterLabeler =
  (text: (key: string) => string) =>
  (character: string): string => {
    if (character === ' ') return text('space');
    if (character === '\n') return text('newline');
    return character;
  };

export const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

export function ExperimentSteps({
  label,
  steps,
  current,
}: {
  label: string;
  steps: readonly string[];
  current: number;
}) {
  return (
    <ol className="experiment-steps" aria-label={label}>
      {steps.map((step, index) => {
        const state = index < current ? 'is-done' : index === current ? 'is-current' : '';
        return (
          <li key={step} className={state} aria-current={index === current ? 'step' : undefined}>
            <span className="experiment-step-n">{index < current ? '✓' : index + 1}</span>
            {step}
          </li>
        );
      })}
    </ol>
  );
}

export function ProbabilityBars({
  label,
  probabilities,
  target,
  characterLabel,
  count = 3,
}: {
  label: string;
  probabilities: readonly number[];
  target: string;
  characterLabel: (character: string) => string;
  count?: number;
}) {
  const ranked = probabilities
    .map((value, token) => ({ character: vocabulary[token], value }))
    .sort((left, right) => right.value - left.value)
    .slice(0, count);
  return (
    <ul className="experiment-bars" aria-label={label}>
      {ranked.map(({ character, value }, index) => (
        <li
          key={character}
          className={`${index === 0 ? 'is-top' : ''} ${character === target ? 'is-target' : ''}`}
        >
          <code>{characterLabel(character)}</code>
          <span className="experiment-bar" aria-hidden="true">
            <span style={{ width: `${value * 100}%` }} />
          </span>
          <span className="experiment-bar-value">{percent(value)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Calm live loss curve. Points are the async loss reports; the x axis spans the whole run. */
export function LossSparkline({
  label,
  history,
  totalSteps,
}: {
  label: string;
  history: readonly { step: number; loss: number }[];
  totalSteps: number;
}) {
  const width = 240;
  const height = 56;
  const pad = 3;
  const points = history.filter((p) => Number.isFinite(p.loss));
  const max = Math.max(...points.map((p) => p.loss), 1e-6);
  const min = Math.min(...points.map((p) => p.loss), max);
  const span = Math.max(max - min, max * 0.05, 1e-6);
  const x = (step: number) => pad + ((width - 2 * pad) * step) / Math.max(1, totalSteps);
  const y = (loss: number) => pad + ((height - 2 * pad) * (max - loss)) / span;
  const path = points.map((p) => `${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(' ');
  const last = points.at(-1);
  return (
    <figure className="experiment-loss" aria-label={label}>
      <figcaption>
        <span className="label">{label}</span>
        <span className="experiment-bar-value">{last ? last.loss.toFixed(3) : '…'}</span>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
        <line
          className="experiment-loss-axis"
          x1={pad}
          x2={width - pad}
          y1={height - pad}
          y2={height - pad}
        />
        {points.length > 1 && <polyline className="experiment-loss-line" points={path} />}
        {last && (
          <circle className="experiment-loss-dot" cx={x(last.step)} cy={y(last.loss)} r={2.5} />
        )}
      </svg>
    </figure>
  );
}

/** Small "what it guesses right now" line for a sentence, from the latest mid-training peek. */
export function LiveGuess({
  label,
  prediction,
  target,
  characterLabel,
}: {
  label: string;
  prediction: FixedWindowPrediction;
  target: string;
  characterLabel: (character: string) => string;
}) {
  const probability = prediction.probabilities[vocabulary.indexOf(prediction.prediction)] ?? 0;
  const verdict = prediction.prediction === target ? 'is-correct' : 'is-wrong';
  return (
    <div className={`experiment-live ${verdict}`}>
      <span className="experiment-meta">{label}</span>
      <span className={`experiment-token ${verdict}`}>{characterLabel(prediction.prediction)}</span>
      <span className="comparison-prob">
        <span className="comparison-prob-track">
          <span className="comparison-prob-fill" style={{ width: `${probability * 100}%` }} />
        </span>
        <span className="experiment-bar-value">{(probability * 100).toFixed(1)}%</span>
      </span>
    </div>
  );
}

export function TrainingProgress({
  label,
  message,
  progress,
  lossLabel,
  lossHistory,
}: {
  label: string;
  message: string;
  progress: FixedWindowExperimentState['progress'];
  lossLabel?: string;
  lossHistory?: FixedWindowExperimentState['lossHistory'];
}) {
  return (
    <div className="experiment-training" role="status">
      <p>{message}</p>
      {lossLabel && progress && (
        <LossSparkline
          label={lossLabel}
          history={lossHistory ?? []}
          totalSteps={progress.totalSteps}
        />
      )}
      <div
        className={progress ? 'experiment-bar' : 'experiment-bar is-indeterminate'}
        role="progressbar"
        aria-label={label}
        {...(progress
          ? {
              'aria-valuemin': 0,
              'aria-valuemax': progress.totalSteps,
              'aria-valuenow': progress.step,
            }
          : {})}
      >
        <span
          style={
            progress
              ? {
                  width: `${Math.min(100, (100 * progress.step) / Math.max(1, progress.totalSteps))}%`,
                }
              : undefined
          }
        />
      </div>
    </div>
  );
}

export function RunPicker({
  label,
  runs,
  selected,
  disabled,
  onSelect,
}: {
  label: string;
  runs: readonly { step: number; title: string; detail?: string }[];
  selected?: number;
  disabled: boolean;
  onSelect: (step: number) => void;
}) {
  return (
    <div className="experiment-runs" role="group" aria-label={label}>
      <span className="label">{label}</span>
      {runs.map((run) => (
        <button
          key={run.step}
          type="button"
          className="experiment-run"
          disabled={disabled}
          aria-pressed={selected === run.step}
          onClick={() => onSelect(run.step)}
        >
          <strong>{run.title}</strong>
          {run.detail && <span>{run.detail}</span>}
        </button>
      ))}
    </div>
  );
}
