import { useState } from 'react';
import type { Progress } from '../engine/levels';
import { saveFixedWindowResult } from './fixed-window-cache';
import { vocabulary } from '../levels/corpus';
import { fixedWindowContext, fixedWindowExamples } from '../levels/fixed-window-evidence';
import {
  useFixedWindowExperiment,
  type FixedWindowExperimentState,
} from './use-fixed-window-experiment';
import {
  ExperimentSteps,
  LiveGuess,
  ProbabilityBars,
  RunPicker,
  TrainingProgress,
  characterLabeler,
} from './TemporalExperimentParts';

type Translator = (key: string, params?: Readonly<Record<string, string | number>>) => string;
interface LessonProps {
  code: string;
  t: Translator;
  backend?: 'cpu' | 'webgpu';
  /** The player's saved circuits: 5-1 trains their own 4-6 fixed window, same as 5-2. */
  progress?: Progress;
  onContinue: () => void;
}

export function FixedWindowLimitLesson(props: LessonProps) {
  const experiment = useFixedWindowExperiment('fixed-window', props.progress, props.backend, {
    // 5-2 reuses this measured result when the circuit and settings are unchanged.
    onSnapshots: (snapshots) => {
      if (snapshots.length === 2) saveFixedWindowResult(props.progress, snapshots);
    },
  });
  return <FixedWindowLimitView {...props} state={experiment.state} onTrain={experiment.train} />;
}

const STEP_KEYS = ['runStage', 'trainStage', 'inspectStage'] as const;
const PROMPT_KEYS = ['beforeTraining', 'afterTraining', 'afterMoreTraining'] as const;
const ACTION_KEYS = ['run', 'trainMore', 'inspect'] as const;

export function FixedWindowLimitView({
  code,
  t,
  onContinue,
  state,
  onTrain,
}: LessonProps & {
  state: FixedWindowExperimentState;
  onTrain: () => void;
}) {
  const [revealed, setRevealed] = useState(false);
  const [selectedStep, setSelectedStep] = useState<number>();
  const snapshot =
    state.snapshots.find((run) => run.step === selectedStep) ?? state.snapshots.at(-1);
  const compared = state.snapshots.length === 2;
  const livePredictions = state.status === 'training' ? state.peek?.predictions : undefined;
  const training = state.status === 'training';
  const text = (key: string, params?: Readonly<Record<string, string | number>>) =>
    t(`levels.time-window-limit.${key}`, params);
  const characterLabel = characterLabeler(text);
  const beginTraining = () => {
    setSelectedStep(undefined);
    setRevealed(false);
    onTrain();
  };
  // 0: untrained, 1: one run, 2: both runs, 3: input revealed.
  const stage = revealed ? 3 : Math.min(state.snapshots.length, 2);

  return (
    <main className="experiment blueprint-bg">
      <div className="experiment-inner">
        <header className="experiment-head">
          <span className="level-id">
            LEVEL {code} · {text('experiment')}
          </span>
          <h1 className="level-title">{text('title')}</h1>
          <p className="experiment-lede">{text('description')}</p>
        </header>

        <ExperimentSteps
          label={text('experiment')}
          steps={STEP_KEYS.map((key) => text(key))}
          current={stage}
        />

        <section className="experiment-panel" aria-label={text('samples')}>
          <div className="experiment-panel-head">
            <h2 className="label">{text('samples')}</h2>
            {snapshot && (
              <span className="experiment-meta">{text('snapshot', { step: snapshot.step })}</span>
            )}
          </div>
          <div className="experiment-cases">
            {fixedWindowExamples.map((example, index) => {
              // While training, the latest live guess replaces the previous run's result.
              const result = livePredictions?.length
                ? undefined
                : snapshot?.predictions.find(
                    (prediction) => prediction.history === example.history,
                  );
              const correct = result?.prediction === example.target;
              const verdict = result && (correct ? 'is-correct' : 'is-wrong');
              return (
                <article className="experiment-case" key={example.history}>
                  <div className="experiment-case-head">
                    <span className="label">{text('case', { n: index + 1 })}</span>
                    <span className="experiment-meta">
                      {text('correctTarget')} <code>{characterLabel(example.target)}</code>
                    </span>
                  </div>
                  <div className="experiment-history" aria-label={example.history}>
                    {revealed ? (
                      <>
                        <span className="is-excluded">
                          {example.history.slice(0, -fixedWindowContext.length)}
                        </span>
                        <mark>{result?.input}</mark>
                      </>
                    ) : (
                      example.history
                    )}
                    <span className={`experiment-token ${verdict ?? ''}`}>
                      {result ? characterLabel(result.prediction) : '?'}
                    </span>
                  </div>
                  {result ? (
                    <>
                      <div className={`window-limit-result ${verdict}`}>
                        <span>{text('prediction')}</span>
                        <b>{text(correct ? 'correct' : 'wrong')}</b>
                      </div>
                      <ProbabilityBars
                        label={text('probabilities')}
                        probabilities={result.probabilities}
                        target={example.target}
                        characterLabel={characterLabel}
                      />
                    </>
                  ) : training && livePredictions?.find((p) => p.history === example.history) ? (
                    <LiveGuess
                      label={text('liveGuess', { step: state.peek!.step })}
                      prediction={livePredictions.find((p) => p.history === example.history)!}
                      target={example.target}
                      characterLabel={characterLabel}
                    />
                  ) : (
                    <p className="experiment-empty">{text(training ? 'computing' : 'awaiting')}</p>
                  )}
                  {revealed && (
                    <p className="experiment-meta">
                      {text('modelInput')} <code>{result?.input}</code>
                    </p>
                  )}
                </article>
              );
            })}
          </div>
        </section>

        <section className="experiment-console" aria-label={text('training')}>
          {state.status === 'error' ? (
            <div className="experiment-error" role="alert">
              <p>
                {state.error?.key === 'training.nonfinite'
                  ? text('failedNonfinite')
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
                step: state.progress?.step ?? snapshot?.step ?? 0,
                total: state.progress?.totalSteps ?? '…',
              })}
              progress={state.progress}
              lossLabel={text('lossCurve')}
              lossHistory={state.lossHistory}
            />
          ) : (
            !revealed && (
              <div className="experiment-prompt" aria-live="polite">
                <p>{text(PROMPT_KEYS[stage])}</p>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={compared ? () => setRevealed(true) : beginTraining}
                >
                  {text(ACTION_KEYS[stage])} →
                </button>
              </div>
            )
          )}
          {state.snapshots.length > 0 && (
            <RunPicker
              label={text('compareRuns')}
              runs={state.snapshots.map((run) => ({
                step: run.step,
                title: text('steps', { step: run.step }),
                detail: `loss ${run.loss.toFixed(3)}`,
              }))}
              selected={snapshot?.step}
              disabled={training}
              onSelect={setSelectedStep}
            />
          )}
        </section>

        {revealed && (
          <section className="experiment-takeaway window-limit-conclusion" aria-live="polite">
            <h2>{text('conclusion')}</h2>
            <p>{text('sameOutput')}</p>
            <div className="experiment-prompt">
              <p className="experiment-meta">{text('next')}</p>
              <button type="button" className="btn btn-primary" onClick={onContinue}>
                {text('continue')} →
              </button>
            </div>
          </section>
        )}

        <details className="experiment-details">
          <summary>{text(revealed ? 'why' : 'aboutExperiment')}</summary>
          {revealed && <p>{text('explanation')}</p>}
          <p>{text('demonstrationNote')}</p>
          {snapshot && (
            <p className="experiment-meta">
              {text('referenceModel')} · Adam · lr {snapshot.learningRate} · seed {snapshot.seed}
            </p>
          )}
        </details>
      </div>
    </main>
  );
}
