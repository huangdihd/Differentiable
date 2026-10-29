import { vocabulary } from '../levels/corpus';
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

export function TrainingProgress({
  label,
  message,
  progress,
}: {
  label: string;
  message: string;
  progress: FixedWindowExperimentState['progress'];
}) {
  return (
    <div className="experiment-training" role="status">
      <p>{message}</p>
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
