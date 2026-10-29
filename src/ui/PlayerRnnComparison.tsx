import { useState } from 'react';
import type { Progress } from '../engine/levels';
import { translateDiagnostic, type Locale } from '../i18n';
import { vocabulary } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowPrediction } from '../worker/fixed-window-protocol';
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
  const fixed = useFixedWindowExperiment('fixed-window', progress, backend);
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
  const text = (key: string, params?: Readonly<Record<string, string | number>>) =>
    t(`levels.time-player-comparison.${key}`, params);
  const hasBothRuns = (state: FixedWindowExperimentState) =>
    state.status === 'ready' &&
    [320, 640].every((step) => state.snapshots.some((run) => run.step === step));
  const fixedComplete = hasBothRuns(fixedState);
  const completed = fixedComplete && hasBothRuns(recurrentState);
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
  if (completed) promptKey = 'conclusion';
  let actionKey = fixedState.snapshots.length ? 'trainFixedMore' : 'runFixed';
  if (fixedComplete) {
    actionKey = recurrentState.snapshots.length ? 'trainRecurrentMore' : 'runRecurrent';
  }
  const characterLabel = (character: string) => {
    if (character === ' ') return text('space');
    if (character === '\n') return text('newline');
    return character;
  };
  const comparisonResult = (
    result: FixedWindowPrediction | undefined,
    target: string,
    model: string,
    step?: number,
  ) => (
    <div className="window-limit-comparison-result">
      <span>{step === undefined ? model : text('modelAtStep', { model, step })}</span>
      {result ? (
        <>
          <strong className={result.prediction === target ? 'is-correct' : 'is-wrong'}>
            <code>{characterLabel(result.prediction)}</code>{' '}
            {(result.probabilities[vocabulary.indexOf(result.prediction)] * 100).toFixed(2)}%
            <small>{text(result.prediction === target ? 'correct' : 'wrong')}</small>
          </strong>
          <small>
            {text('modelInput')} <code>{result.input}</code>
          </small>
        </>
      ) : (
        <span>{text('awaiting')}</span>
      )}
    </div>
  );

  return (
    <main className="window-limit blueprint-bg is-comparing">
      <div className="window-limit-inner">
        <header className="window-limit-head">
          <div className="window-limit-actions">
            <span className="level-id">
              LEVEL {code} / {text('experiment')}
            </span>
            <button type="button" className="btn" onClick={onBack}>
              ← {text('back')}
            </button>
          </div>
          <h1>{text('title')}</h1>
          <p>{text('description')}</p>
          <p className="window-limit-note">{text('freshWeights')}</p>
        </header>
        <section className="window-limit-bench" aria-label={text('samples')}>
          <div className="window-limit-bench-head">
            <span className="label">{text('samples')}</span>
            {fixedSnapshot && (
              <span>
                {text(recurrentSnapshot ? 'matchedSnapshot' : 'fixedSnapshot', {
                  step: fixedSnapshot.step,
                })}
              </span>
            )}
          </div>
          <div className="window-limit-input-mode">
            <label>
              <input
                type="checkbox"
                checked={resetHistory}
                disabled={!recurrentSnapshot?.resetPredictions}
                onChange={(event) => setResetHistory(event.target.checked)}
              />
              {text('resetHistory')}
            </label>
            <span>{text(resetHistory ? 'resetNote' : 'fullHistory')}</span>
          </div>
          <div className="window-limit-examples">
            {fixedWindowExamples.map((example, index) => {
              const fixedResult = fixedSnapshot?.predictions.find(
                (prediction) => prediction.history === example.history,
              );
              const recurrentPredictions = resetHistory
                ? recurrentSnapshot?.resetPredictions
                : recurrentSnapshot?.predictions;
              const recurrentResult = recurrentPredictions?.find(
                (prediction) => prediction.history === example.history,
              );
              return (
                <article className="window-limit-example" key={example.history}>
                  <div className="window-limit-case">
                    <span>{text('case', { n: index + 1 })}</span>
                    <span>
                      {text('correctTarget')} <strong>{characterLabel(example.target)}</strong>
                    </span>
                  </div>
                  <div className="window-limit-history" aria-label={example.history}>
                    {example.history}
                  </div>
                  <div className="window-limit-comparison" aria-label={text('modelComparison')}>
                    {comparisonResult(
                      fixedResult,
                      example.target,
                      text('fixedModel'),
                      fixedSnapshot?.step,
                    )}
                    {comparisonResult(
                      recurrentResult,
                      example.target,
                      text('recurrentModel'),
                      recurrentSnapshot?.step,
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </section>
        <section className="window-limit-console" aria-label={text('training')}>
          <div className="window-limit-console-head">
            <span className="label">{text(completed ? 'modelComparison' : activeModelKey)}</span>
            <span>{text('trainingPlan')}</span>
          </div>
          {runs.length > 0 && (
            <div className="window-limit-runs" role="group" aria-label={text('compareRuns')}>
              {runs.map((run) => (
                <button
                  key={run.step}
                  type="button"
                  className="window-limit-run"
                  disabled={training}
                  aria-pressed={fixedSnapshot === run}
                  onClick={() => setSelectedStep(run.step)}
                >
                  {text('steps', { step: run.step })}
                </button>
              ))}
            </div>
          )}
          <div className="window-limit-metrics">
            {[fixedSnapshot, recurrentSnapshot].map(
              (snapshot) =>
                snapshot && (
                  <span key={snapshot.model}>
                    {text('modelMetrics', {
                      model: text(snapshot.model === 'rnn' ? 'recurrentModel' : 'fixedModel'),
                      parameters: snapshot.parameterCount,
                      samples: snapshot.sampleCount,
                    })}
                    {' · '}loss <strong>{snapshot.loss.toFixed(3)}</strong>
                    {' · '}Adam · {snapshot.learningRate} · seed {snapshot.seed}
                  </span>
                ),
            )}
          </div>
          {activeState.status === 'error' ? (
            <div className="window-limit-error" role="alert">
              <p>{text('failed')}</p>
              {activeState.error && (
                <p>
                  {text(activeModelKey)}：{translateDiagnostic(locale, activeState.error)}{' '}
                  <code>{activeState.error.key}</code>
                  {activeState.error.node && (
                    <>
                      {' '}
                      ·{' '}
                      <code>
                        {activeState.error.node}
                        {activeState.error.port && `:${activeState.error.port}`}
                      </code>
                    </>
                  )}
                </p>
              )}
              <button type="button" className="btn" onClick={beginTraining}>
                {text('retry')}
              </button>
            </div>
          ) : training ? (
            <div className="window-limit-training" role="status">
              <p>
                {text('trainingProgress', {
                  model: text(activeModelKey),
                  step: activeState.progress?.step ?? activeState.snapshots.at(-1)?.step ?? 0,
                  total: activeState.progress?.totalSteps ?? '…',
                })}
              </p>
              <progress
                aria-label={text('training')}
                {...(activeState.progress
                  ? { value: activeState.progress.step, max: activeState.progress.totalSteps }
                  : {})}
              />
            </div>
          ) : (
            <div className="window-limit-prompt" aria-live="polite">
              <p>{text(promptKey)}</p>
              {!completed && (
                <button type="button" className="btn btn-primary" onClick={beginTraining}>
                  {text(actionKey)} →
                </button>
              )}
            </div>
          )}
          <p className="window-limit-note">{text('temporaryResults')}</p>
        </section>
        <details className="window-limit-details">
          <summary>{text('comparisonMethod')}</summary>
          <p>{text('comparisonNote')}</p>
        </details>
        {completed && (
          <footer className="window-limit-actions">
            <p>{text('next')}</p>
            <button type="button" className="btn btn-primary" onClick={onContinue}>
              {text('continue')} →
            </button>
          </footer>
        )}
      </div>
    </main>
  );
}
