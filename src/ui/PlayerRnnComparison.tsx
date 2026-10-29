import { useState } from 'react';
import type { Progress } from '../engine/levels';
import { translateDiagnostic, type Locale } from '../i18n';
import { vocabulary } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowPrediction } from '../worker/fixed-window-protocol';
import {
  ExperimentSteps,
  RunPicker,
  TrainingProgress,
  characterLabeler,
  percent,
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
  const characterLabel = characterLabeler(text);
  const stage = completed ? 2 : fixedComplete ? 1 : 0;
  const recurrentPredictions = resetHistory
    ? recurrentSnapshot?.resetPredictions
    : recurrentSnapshot?.predictions;
  const modelRow = (
    result: FixedWindowPrediction | undefined,
    target: string,
    model: string,
    kind: 'fixed' | 'recurrent',
  ) => {
    const correct = result?.prediction === target;
    const verdict = result ? (correct ? 'is-correct' : 'is-wrong') : '';
    return (
      <div className={`window-limit-comparison-result experiment-model is-${kind} ${verdict}`}>
        <span className="experiment-model-name">{model}</span>
        {result ? (
          <>
            <span className={`experiment-token ${verdict}`}>
              {characterLabel(result.prediction)}
            </span>
            <span className="experiment-bar-value">
              {percent(result.probabilities[vocabulary.indexOf(result.prediction)])}
            </span>
            <b className="experiment-verdict">{text(correct ? 'correct' : 'wrong')}</b>
            <small className="experiment-meta">
              {text('modelInput')} <code>{result.input}</code>
            </small>
          </>
        ) : (
          <span className="experiment-empty">{text('awaiting')}</span>
        )}
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
                <div className="experiment-history" aria-label={example.history}>
                  {example.history}
                  <span className="experiment-token">?</span>
                </div>
                <div className="window-limit-comparison" aria-label={text('modelComparison')}>
                  {modelRow(
                    fixedSnapshot?.predictions.find((p) => p.history === example.history),
                    example.target,
                    text('fixedModel'),
                    'fixed',
                  )}
                  {modelRow(
                    recurrentPredictions?.find((p) => p.history === example.history),
                    example.target,
                    text('recurrentModel'),
                    'recurrent',
                  )}
                </div>
              </article>
            ))}
          </div>
          {recurrentSnapshot?.resetPredictions && (
            <label className="experiment-toggle">
              <input
                type="checkbox"
                checked={resetHistory}
                disabled={false}
                onChange={(event) => setResetHistory(event.target.checked)}
              />
              <span>
                <strong>{text('resetHistory')}</strong>
                <span className="experiment-meta">
                  {text(resetHistory ? 'resetNote' : 'fullHistory')}
                </span>
              </span>
            </label>
          )}
        </section>

        <section className="experiment-console" aria-label={text('training')}>
          {activeState.status === 'error' ? (
            <div className="experiment-error" role="alert">
              <div>
                <p>{text('failed')}</p>
                {activeState.error && (
                  <p className="experiment-meta">
                    {text(activeModelKey)}: {translateDiagnostic(locale, activeState.error)}{' '}
                    <code>
                      {activeState.error.key}
                      {activeState.error.node && ` · ${activeState.error.node}`}
                      {activeState.error.port && `:${activeState.error.port}`}
                    </code>
                  </p>
                )}
              </div>
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
            />
          ) : completed ? null : (
            <div className="experiment-prompt" aria-live="polite">
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

        {completed && (
          <section className="experiment-takeaway" aria-live="polite">
            <h2>{text('conclusionTitle')}</h2>
            <p>{text(promptKey)}</p>
            <div className="experiment-prompt">
              <p className="experiment-meta">{text('next')}</p>
              <button type="button" className="btn btn-primary" onClick={onContinue}>
                {text('continue')} →
              </button>
            </div>
          </section>
        )}

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
