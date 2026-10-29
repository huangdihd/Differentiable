import { useState } from 'react';
import { vocabulary } from '../levels/corpus';
import { fixedWindowContext, fixedWindowExamples } from '../levels/fixed-window-evidence';
import {
  useFixedWindowExperiment,
  type FixedWindowExperimentState,
} from './use-fixed-window-experiment';

type Translator = (key: string, params?: Readonly<Record<string, string | number>>) => string;
interface LessonProps {
  code: string;
  t: Translator;
  backend?: 'cpu' | 'webgpu';
  onContinue: () => void;
}

export function FixedWindowLimitLesson(props: LessonProps) {
  const experiment = useFixedWindowExperiment('fixed-window', undefined, props.backend);
  return <FixedWindowLimitView {...props} state={experiment.state} onTrain={experiment.train} />;
}

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
  const training = state.status === 'training';
  const text = (key: string, params?: Readonly<Record<string, string | number>>) =>
    t(`levels.time-window-limit.${key}`, params);
  const characterLabel = (character: string) => {
    if (character === ' ') return text('space');
    if (character === '\n') return text('newline');
    return character;
  };
  const beginTraining = () => {
    setSelectedStep(undefined);
    setRevealed(false);
    onTrain();
  };
  const stage = revealed ? 3 : state.snapshots.length;
  const promptKeys = ['beforeTraining', 'afterTraining', 'afterMoreTraining', 'sameOutput'];
  const actionKeys = ['run', 'trainMore', 'inspect'];

  return (
    <main className="window-limit blueprint-bg">
      <div className="window-limit-inner">
        <header className="window-limit-head">
          <span className="level-id">
            LEVEL {code} / {text('experiment')}
          </span>
          <h1>{text('title')}</h1>
          <p>{text('description')}</p>
        </header>
        <ol className="window-limit-stages" aria-label={text('experiment')}>
          {['runStage', 'trainStage', 'inspectStage', 'rememberStage'].map((key, index) => (
            <li
              key={key}
              className={index <= stage ? 'is-active' : ''}
              aria-current={index === stage ? 'step' : undefined}
            >
              <span>0{index + 1}</span> {text(key)}
            </li>
          ))}
        </ol>
        <section className="window-limit-bench" aria-label={text('samples')}>
          <div className="window-limit-bench-head">
            <span className="label">{text('samples')}</span>
            {snapshot && <span>{text('snapshot', { step: snapshot.step })}</span>}
          </div>
          <div className="window-limit-examples">
            {fixedWindowExamples.map((example, index) => {
              const result = snapshot?.predictions.find(
                (prediction) => prediction.history === example.history,
              );
              const correct = result?.prediction === example.target;
              const resultClass = result && (correct ? 'is-correct' : 'is-wrong');
              const ranked = result?.probabilities
                .map((value, token) => ({ character: vocabulary[token], value }))
                .sort((left, right) => right.value - left.value)
                .slice(0, 4);
              return (
                <article className="window-limit-example" key={example.history}>
                  <div className="window-limit-case">
                    <span>{text('case', { n: index + 1 })}</span>
                    {revealed && (
                      <span>
                        {text('modelInput')} <code>{fixedWindowContext}</code>
                      </span>
                    )}
                  </div>
                  <div className="window-limit-history" aria-label={example.history}>
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
                    <span className={`window-limit-token ${resultClass ?? ''}`}>
                      {result ? characterLabel(result.prediction) : '?'}
                    </span>
                  </div>
                  <div className={`window-limit-result ${resultClass ?? ''}`}>
                    <span>
                      {text('correctTarget')} <strong>{characterLabel(example.target)}</strong>
                    </span>
                    {result && <b>{text(correct ? 'correct' : 'wrong')}</b>}
                  </div>
                  {ranked ? (
                    <div className="window-limit-probabilities">
                      <div className="window-limit-probability-label">{text('topPrediction')}</div>
                      <ul aria-label={text('probabilities')}>
                        {ranked.map(({ character, value }) => (
                          <li key={character}>
                            <code>{characterLabel(character)}</code>
                            <span className="window-limit-bar">
                              <span style={{ width: `${value * 100}%` }} />
                            </span>
                            <span>{(value * 100).toFixed(2)}%</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <p className="window-limit-awaiting">
                      {text(training ? 'computing' : 'awaiting')}
                    </p>
                  )}
                </article>
              );
            })}
          </div>
        </section>
        <section className="window-limit-console" aria-label={text('training')}>
          <div className="window-limit-console-head">
            <span className="label">{text('referenceModel')}</span>
            {snapshot && (
              <code>
                Adam · {snapshot.learningRate} · seed {snapshot.seed}
              </code>
            )}
          </div>
          {state.snapshots.length > 0 && (
            <div className="window-limit-runs" role="group" aria-label={text('compareRuns')}>
              {state.snapshots.map((run) => (
                <button
                  key={run.step}
                  type="button"
                  className="window-limit-run"
                  disabled={training}
                  aria-pressed={snapshot === run}
                  onClick={() => setSelectedStep(run.step)}
                >
                  <span>{text('steps', { step: run.step })}</span>
                  <span>
                    loss <strong>{run.loss.toFixed(3)}</strong>
                  </span>
                </button>
              ))}
            </div>
          )}
          {state.status === 'error' ? (
            <div className="window-limit-error" role="alert">
              <p>{text('failed')}</p>
              <button type="button" className="btn" onClick={beginTraining}>
                {text('retry')}
              </button>
            </div>
          ) : training ? (
            <div className="window-limit-training" role="status">
              <p>
                {text('trainingProgress', {
                  step: state.progress?.step ?? state.snapshots.at(-1)?.step ?? 0,
                  total: state.progress?.totalSteps ?? '…',
                })}
              </p>
              <progress
                aria-label={text('training')}
                {...(state.progress
                  ? { value: state.progress.step, max: state.progress.totalSteps }
                  : {})}
              />
            </div>
          ) : (
            <div className="window-limit-prompt" aria-live="polite">
              <p>{text(promptKeys[stage])}</p>
              {!revealed && (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={compared ? () => setRevealed(true) : beginTraining}
                >
                  {text(actionKeys[stage])} →
                </button>
              )}
            </div>
          )}
          <p className="window-limit-note">{text('demonstrationNote')}</p>
        </section>
        {revealed && (
          <section className="window-limit-conclusion" aria-live="polite">
            <h2>{text('conclusion')}</h2>
            <details className="window-limit-details">
              <summary>{text('why')}</summary>
              <p>{text('explanation')}</p>
            </details>
            <footer className="window-limit-actions">
              <p>{text('next')}</p>
              <button type="button" className="btn btn-primary" onClick={onContinue}>
                {text('continue')} →
              </button>
            </footer>
          </section>
        )}
      </div>
    </main>
  );
}
