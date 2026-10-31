// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useState } from 'react';
import type { Progress } from '../engine/levels';
import { translateDiagnostic, type Locale } from '../i18n';
import { vocabulary } from '../levels/corpus';
import { fixedWindowContext, fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowPrediction } from '../worker/fixed-window-protocol';
import { loadFixedWindowResult, saveFixedWindowResult } from './fixed-window-cache';
import {
  ExperimentSteps,
  RunPicker,
  TrainingProgress,
  characterLabeler,
} from './TemporalExperimentParts';
import {
  useFixedWindowExperiment,
  type FixedWindowExperimentState,
} from './use-fixed-window-experiment';

type Translator = (key: string, params?: Readonly<Record<string, string | number>>) => string;
interface PlayerRnnComparisonProps {
  code: string;
  locale: Locale;
  t: Translator;
  progress: Progress;
  backend: 'cpu' | 'webgpu';
  onContinue: () => void;
  onBack: () => void;
}

export function PlayerRnnComparison({ progress, backend, ...props }: PlayerRnnComparisonProps) {
  // 5-1 already trained this exact circuit with these settings: reuse it, don't retrain.
  const [cached] = useState(() => loadFixedWindowResult(progress));
  const fixed = useFixedWindowExperiment('fixed-window', progress, backend, {
    initialSnapshots: cached,
    onSnapshots: (snapshots) => {
      if (snapshots.length === 2) saveFixedWindowResult(progress, snapshots);
    },
  });
  const recurrent = useFixedWindowExperiment('rnn', progress, backend);
  return (
    <PlayerRnnComparisonView
      {...props}
      fixedState={fixed.state}
      recurrentState={recurrent.state}
      onTrainFixed={fixed.train}
      onTrainRecurrent={recurrent.train}
    />
  );
}

export function PlayerRnnComparisonView({
  code,
  locale,
  t,
  onContinue,
  onBack,
  fixedState,
  recurrentState,
  onTrainFixed,
  onTrainRecurrent,
}: Omit<PlayerRnnComparisonProps, 'progress' | 'backend'> & {
  fixedState: FixedWindowExperimentState;
  recurrentState: FixedWindowExperimentState;
  onTrainFixed: () => void;
  onTrainRecurrent: () => void;
}) {
  const [selectedStep, setSelectedStep] = useState<number>();
  const [resetHistory, setResetHistory] = useState(false);
  const [skipped, setSkipped] = useState(false);
  const text = (key: string, params?: Readonly<Record<string, string | number>>) =>
    t(`levels.time-player-comparison.${key}`, params);
  const hasBothRuns = (state: FixedWindowExperimentState) =>
    state.status === 'ready' &&
    [320, 640].every((step) => state.snapshots.some((run) => run.step === step));
  const fixedComplete = hasBothRuns(fixedState);
  const firstRecurrent =
    recurrentState.status === 'ready' && recurrentState.snapshots.length === 1
      ? recurrentState.snapshots[0]
      : undefined;
  // Skipping is offered only when the first RNN run already gets both sentences right.
  const canSkip =
    fixedComplete &&
    !!firstRecurrent &&
    fixedWindowExamples.every(
      (example) =>
        firstRecurrent.predictions.find((p) => p.history === example.history)?.prediction ===
        example.target,
    );
  const completed = fixedComplete && (hasBothRuns(recurrentState) || (skipped && canSkip));
  const activeState = fixedComplete ? recurrentState : fixedState;
  const training = activeState.status === 'training';
  // Once RNN results exist, only offer budgets measured for both player graphs.
  const runs = recurrentState.snapshots.length
    ? fixedState.snapshots.filter((run) =>
        recurrentState.snapshots.some((other) => other.step === run.step),
      )
    : fixedState.snapshots;
  const selected = runs.find((run) => run.step === selectedStep);
  const fixedSnapshot =
    selected ?? (fixedComplete && !recurrentState.snapshots.length ? runs[0] : runs.at(-1));
  const recurrentSnapshot = recurrentState.snapshots.find(
    (run) => run.step === fixedSnapshot?.step,
  );
  const beginTraining = () => {
    setSelectedStep(undefined);
    setResetHistory(false);
    if (fixedComplete) onTrainRecurrent();
    else onTrainFixed();
  };
  const activeModelKey = fixedComplete ? 'recurrentModel' : 'fixedModel';
  let promptKey = fixedComplete ? 'comparePrompt' : 'fixedPrompt';
  let actionKey = fixedState.snapshots.length ? 'trainFixedMore' : 'runFixed';
  if (fixedComplete) {
    actionKey = recurrentState.snapshots.length ? 'trainRecurrentMore' : 'runRecurrent';
  }
  const characterLabel = characterLabeler(text);
  const stage = completed ? 2 : fixedComplete ? 1 : 0;
  const livePeek = training ? activeState.peek : undefined;
  const recurrentPredictions =
    livePeek && fixedComplete
      ? livePeek.predictions
      : resetHistory
        ? recurrentSnapshot?.resetPredictions
        : recurrentSnapshot?.predictions;
  const fixedPredictions =
    livePeek && !fixedComplete ? livePeek.predictions : fixedSnapshot?.predictions;
  const pct1 = (value: number) => `${(value * 100).toFixed(1)}%`;
  const modelRow = (
    result: FixedWindowPrediction | undefined,
    target: string,
    model: string,
    kind: 'fixed' | 'recurrent',
    seenKey: string,
    liveStep?: number,
  ) => {
    const correct = result?.prediction === target;
    const verdict = result ? (correct ? 'is-correct' : 'is-wrong') : '';
    const probability = result ? result.probabilities[vocabulary.indexOf(result.prediction)] : 0;
    return (
      <div className={`comparison-guess is-${kind} ${verdict}`}>
        <div className="comparison-guess-head">
          <span className="experiment-model-name">
            <i className={`comparison-swatch is-${kind}`} aria-hidden="true" />
            {model}
          </span>
          <span className="experiment-meta">
            {liveStep !== undefined ? text('liveStep', { step: liveStep }) : text(seenKey)}
          </span>
        </div>
        {result ? (
          <div className="comparison-guess-body">
            <span className={`experiment-token ${verdict}`}>
              {characterLabel(result.prediction)}
            </span>
            <span className="comparison-prob">
              <span className="comparison-prob-track">
                <span className="comparison-prob-fill" style={{ width: pct1(probability) }} />
              </span>
              <span className="experiment-bar-value">{pct1(probability)}</span>
            </span>
            {liveStep === undefined && (
              <b className="experiment-verdict">{text(correct ? 'correct' : 'wrong')}</b>
            )}
          </div>
        ) : (
          <span className="experiment-empty">{text('awaiting')}</span>
        )}
      </div>
    );
  };
  // Conclusion derived from the actual predictions (full history), never assumed.
  const outcome = (predictions: readonly FixedWindowPrediction[] | undefined) =>
    fixedWindowExamples.map((example) => {
      const p = predictions?.find((x) => x.history === example.history);
      return { example, p, correct: p?.prediction === example.target };
    });
  const fixedOutcome = outcome(fixedSnapshot?.predictions);
  const recurrentOutcome = outcome(recurrentSnapshot?.predictions);
  const fixedRight = fixedOutcome.filter((o) => o.correct).length;
  const recurrentRight = recurrentOutcome.filter((o) => o.correct).length;
  const fixedSame =
    fixedOutcome.every((o) => o.p) && new Set(fixedOutcome.map((o) => o.p!.prediction)).size === 1;
  const fixedGuess = fixedOutcome[0]?.p ? characterLabel(fixedOutcome[0].p.prediction) : '';
  const total = fixedWindowExamples.length;
  const verdictLines: string[] = [];
  if (recurrentSnapshot) {
    verdictLines.push(
      fixedSame
        ? text('verdictFixedSame', { guess: fixedGuess, right: fixedRight, total })
        : text('verdictFixedCount', { right: fixedRight, total }),
    );
    verdictLines.push(
      recurrentRight === total
        ? text('verdictRecurrentAll')
        : text('verdictRecurrentSome', { right: recurrentRight, total }),
    );
  }
  const headlineKey = !recurrentSnapshot
    ? undefined
    : recurrentRight === total && fixedRight < total
      ? 'headlineWin'
      : recurrentRight > fixedRight
        ? 'headlineBetter'
        : 'headlineNoGain';
  const resetOutcome = outcome(recurrentSnapshot?.resetPredictions);
  const resetSame =
    resetOutcome.every((o) => o.p) && new Set(resetOutcome.map((o) => o.p!.prediction)).size === 1;
  const highlighted = (history: string, recurrentSeesAll: boolean) => {
    const cut = history.length - fixedWindowContext.length;
    return (
      <div className="experiment-history comparison-sentence" aria-label={history}>
        <span className={recurrentSeesAll ? 'seen-recurrent' : undefined}>
          {history.slice(0, cut)}
        </span>
        <span className={`seen-fixed${recurrentSeesAll ? ' seen-recurrent' : ' seen-both'}`}>
          {history.slice(cut)}
        </span>
        <span className="experiment-token">?</span>
      </div>
    );
  };

  return (
    <main className="experiment blueprint-bg is-comparing">
      <div className="experiment-inner">
        <header className="experiment-head">
          <div className="experiment-head-row">
            <span className="level-id">
              LEVEL {code} · {text('experiment')}
            </span>
            <button type="button" className="btn" onClick={onBack}>
              ← {text('back')}
            </button>
          </div>
          <h1 className="level-title">{text('title')}</h1>
          <p className="experiment-lede">{text('description')}</p>
        </header>

        <ExperimentSteps
          label={text('experiment')}
          steps={[text('fixedStage'), text('recurrentStage'), text('compareStage')]}
          current={stage}
        />

        {completed && headlineKey && (
          <section className="experiment-takeaway comparison-verdict" aria-live="polite">
            <h2>{text(headlineKey)}</h2>
            {verdictLines.map((line) => (
              <p key={line}>{line}</p>
            ))}
            {recurrentSnapshot?.resetPredictions && (
              <label className="experiment-toggle comparison-try">
                <input
                  type="checkbox"
                  checked={resetHistory}
                  disabled={training}
                  onChange={(event) => setResetHistory(event.target.checked)}
                />
                <span>
                  <strong>{text('resetHistory')}</strong>
                  <span className="experiment-meta">
                    {resetHistory
                      ? resetSame
                        ? text('resetResultSame', {
                            guess: characterLabel(resetOutcome[0].p!.prediction),
                          })
                        : text('resetResultDiffers')
                      : text('resetExpectation')}
                  </span>
                </span>
              </label>
            )}
            <div className="experiment-prompt">
              <p className="experiment-meta">{text('next')}</p>
              <button type="button" className="btn btn-primary" onClick={onContinue}>
                {text('continue')} →
              </button>
            </div>
          </section>
        )}

        <section className="experiment-panel" aria-label={text('samples')}>
          <div className="experiment-panel-head">
            <h2 className="label">{text('samples')}</h2>
            {fixedSnapshot && (
              <span className="experiment-meta">
                {text(recurrentSnapshot ? 'matchedSnapshot' : 'fixedSnapshot', {
                  step: fixedSnapshot.step,
                })}
              </span>
            )}
          </div>
          <div className="experiment-cases">
            {fixedWindowExamples.map((example, index) => (
              <article className="experiment-case" key={example.history}>
                <div className="experiment-case-head">
                  <span className="label">{text('case', { n: index + 1 })}</span>
                  <span className="experiment-meta">
                    {text('correctTarget')} <code>{characterLabel(example.target)}</code>
                  </span>
                </div>
                {highlighted(example.history, !resetHistory)}
                <div className="comparison-guesses" aria-label={text('modelComparison')}>
                  {modelRow(
                    fixedPredictions?.find((p) => p.history === example.history),
                    example.target,
                    text('fixedModel'),
                    'fixed',
                    'seenWindow',
                    livePeek && !fixedComplete ? livePeek.step : undefined,
                  )}
                  {modelRow(
                    recurrentPredictions?.find((p) => p.history === example.history),
                    example.target,
                    text('recurrentModel'),
                    'recurrent',
                    resetHistory ? 'seenWindow' : 'seenWhole',
                    livePeek && fixedComplete ? livePeek.step : undefined,
                  )}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="experiment-console" aria-label={text('training')}>
          {activeState.status === 'error' ? (
            <div className="experiment-error" role="alert">
              <p>
                {activeState.error?.key === 'training.nonfinite'
                  ? text('failedNonfinite')
                  : activeState.error
                    ? `${text(activeModelKey)}: ${translateDiagnostic(locale, activeState.error)}`
                    : text('failed')}
              </p>
              <button type="button" className="btn" onClick={beginTraining}>
                {text('retry')}
              </button>
            </div>
          ) : training ? (
            <TrainingProgress
              label={text('training')}
              message={text('trainingProgress', {
                model: text(activeModelKey),
                step: activeState.progress?.step ?? activeState.snapshots.at(-1)?.step ?? 0,
                total: activeState.progress?.totalSteps ?? '…',
              })}
              progress={activeState.progress}
              lossLabel={text('lossCurve', { model: text(activeModelKey) })}
              lossHistory={activeState.lossHistory}
            />
          ) : completed ? null : canSkip ? (
            <div className="experiment-prompt" aria-live="polite">
              <p>{text('skipPrompt')}</p>
              <div className="experiment-actions">
                <button type="button" className="btn btn-primary" onClick={() => setSkipped(true)}>
                  {text('skipToResults')} →
                </button>
                <button type="button" className="btn" onClick={beginTraining}>
                  {text(actionKey)}
                </button>
              </div>
            </div>
          ) : (
            <div className="experiment-prompt" aria-live="polite">
              {fixedState.cached && fixedComplete && !recurrentState.snapshots.length && (
                <p className="experiment-meta">{text('fixedFromLesson')}</p>
              )}
              <p>{text(promptKey)}</p>
              <button type="button" className="btn btn-primary" onClick={beginTraining}>
                {text(actionKey)} →
              </button>
            </div>
          )}
          {runs.length > 0 && (
            <RunPicker
              label={text('compareRuns')}
              runs={runs.map((run) => ({
                step: run.step,
                title: text('steps', { step: run.step }),
              }))}
              selected={fixedSnapshot?.step}
              disabled={training}
              onSelect={setSelectedStep}
            />
          )}
        </section>

        <details className="experiment-details">
          <summary>{text('comparisonMethod')}</summary>
          <p>{text('freshWeights')}</p>
          <p>{text('comparisonNote')}</p>
          {(fixedSnapshot || recurrentSnapshot) && (
            <table className="table experiment-metrics">
              <thead>
                <tr>
                  <th>{text('metricModel')}</th>
                  <th>{text('metricParameters')}</th>
                  <th>{text('metricSamples')}</th>
                  <th>loss</th>
                  <th>{text('metricOptimizer')}</th>
                </tr>
              </thead>
              <tbody>
                {[fixedSnapshot, recurrentSnapshot].map(
                  (snapshot) =>
                    snapshot && (
                      <tr key={snapshot.model}>
                        <th scope="row">
                          {text(snapshot.model === 'rnn' ? 'recurrentModel' : 'fixedModel')}
                        </th>
                        <td>{snapshot.parameterCount.toLocaleString(locale)}</td>
                        <td>{snapshot.sampleCount.toLocaleString(locale)}</td>
                        <td>
                          <strong>{snapshot.loss.toFixed(3)}</strong>
                        </td>
                        <td>
                          Adam · {snapshot.learningRate} · seed {snapshot.seed}
                        </td>
                      </tr>
                    ),
                )}
              </tbody>
            </table>
          )}
          <p className="experiment-meta">{text('temporaryResults')}</p>
        </details>
      </div>
    </main>
  );
}
