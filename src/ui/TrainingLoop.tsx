// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const STAGES = ['data', 'forward', 'loss', 'backward', 'update'] as const;

/**
 * The whole training loop on one card: what the engine repeats each step, and which
 * level the player built each stage in. While training runs, the stages light up in turn.
 */
export function TrainingLoop({
  t,
  optimizer,
  learningRate,
  steps,
  resample,
  running,
}: {
  t: T;
  optimizer: 'sgd' | 'adam';
  learningRate: string;
  steps: string;
  resample: boolean;
  running: boolean;
}) {
  const note = (stage: (typeof STAGES)[number]) =>
    stage === 'data'
      ? t(resample ? 'loop.data.resample' : 'loop.data.fixed')
      : stage === 'update'
        ? t(`loop.update.${optimizer}`, { lr: learningRate })
        : t(`loop.${stage}.note`);
  return (
    <details className={`loop${running ? ' is-running' : ''}`} open>
      <summary className="label">{t('loop.title')}</summary>
      <ol className="loop-stages">
        {STAGES.map((stage, i) => (
          <li key={stage} className={`loop-stage loop-${stage}`} style={{ '--i': i } as never}>
            <span className="loop-n">{i + 1}</span>
            <span className="loop-name">{t(`loop.${stage}.name`)}</span>
            <span className="loop-note">{note(stage)}</span>
          </li>
        ))}
      </ol>
      <p className="loop-repeat">↺ {t('loop.repeat', { steps })}</p>
    </details>
  );
}
