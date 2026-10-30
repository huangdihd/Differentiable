import { useEffect, useRef, useState } from 'react';
import { sequenceExample, sequenceRule, type SequenceProbeKind } from '../levels/sequence-probe';

type Translate = (key: string, params?: Readonly<Record<string, string | number>>) => string;
export type Predict = (
  rows: number[][],
) => Promise<{ values: number[]; state: 'initial' | 'current' | 'trained' }>;

const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));

/** "Try it": the player's own window, the model's prediction, and what it should be. */
export default function SequenceTry({
  t,
  kind,
  length,
  predict,
  refreshKey,
}: {
  t: Translate;
  kind: SequenceProbeKind;
  length: number;
  predict: Predict;
  /** Changes when the weights may have changed (a training report), to re-run the window. */
  refreshKey: unknown;
}) {
  const [seed, setSeed] = useState(1);
  const [example, setExample] = useState(() => sequenceExample(kind, length, 1));
  const [texts, setTexts] = useState(() => example.row.map(fmt));
  const [edited, setEdited] = useState(false);
  const [answer, setAnswer] = useState<{ value: number; state: string }>();
  const [failed, setFailed] = useState(false);
  const values = texts.map((s) => Number(s));
  const valid = texts.every((s) => s.trim() !== '' && Number.isFinite(Number(s)));
  const outOfRange = valid && values.some((v) => Math.abs(v) > 2);
  const expected = edited ? sequenceRule(kind, values) : example.expected;
  const busy = useRef(false);
  const pending = useRef(false);
  const key = JSON.stringify([valid ? values : null, refreshKey]);
  useEffect(() => {
    if (!valid) return;
    let cancelled = false;
    const go = async () => {
      if (busy.current) {
        pending.current = true;
        return;
      }
      busy.current = true;
      try {
        const r = await predict([values]);
        if (!cancelled) {
          setAnswer({ value: r.values[0], state: r.state });
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        busy.current = false;
        if (pending.current && !cancelled) {
          pending.current = false;
          void go();
        }
      }
    };
    const timer = setTimeout(() => void go(), 120);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key]);
  const fresh = () => {
    const next = seed + 1;
    const e = sequenceExample(kind, length, next * 7919);
    setSeed(next);
    setExample(e);
    setTexts(e.row.map(fmt));
    setEdited(false);
  };
  const error = answer && expected !== undefined ? Math.abs(answer.value - expected) : undefined;
  return (
    <div className="seq-try">
      <div className="section-head">
        <span className="label">{t('ui.try.title')}</span>
        <button className="btn" onClick={fresh}>
          ↻ {t(`ui.try.${kind}.fresh`)}
        </button>
      </div>
      <div className="seq-try-row" role="group" aria-label={t('ui.try.input')}>
        {texts.map((text, i) => (
          <label key={i} className="seq-try-cell">
            <span className="seq-try-step">{kind === 'delay' && i === 0 ? '★' : `t${i + 1}`}</span>
            <input
              className="field"
              inputMode="decimal"
              value={text}
              aria-label={t('ui.try.step', { step: i + 1 })}
              onChange={(e) => {
                const next = [...texts];
                next[i] = e.target.value;
                setTexts(next);
                setEdited(true);
              }}
            />
          </label>
        ))}
        <span className="seq-try-arrow">→</span>
        <div className="seq-try-out" aria-live="polite">
          <span className="label">{t('ui.try.prediction')}</span>
          <output
            className={`seq-try-value${error !== undefined && error > 0.15 ? ' is-off' : ''}`}
          >
            {valid && answer && !failed ? fmt(answer.value) : '—'}
          </output>
          <span className="seq-try-expected">
            {expected !== undefined
              ? t('ui.try.expected', { value: fmt(expected) })
              : t('ui.try.noAnswer')}
          </span>
        </div>
      </div>
      <p className="hint">
        {!valid
          ? t('ui.try.invalid')
          : outOfRange
            ? t('ui.try.outOfRange')
            : t(`ui.try.${kind}.help`)}{' '}
        {answer && t(`ui.try.state.${answer.state}`)}
      </p>
    </div>
  );
}
