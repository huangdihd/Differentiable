import { useEffect } from 'react';
import type { GraphJSON } from '../engine/graph';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

/** Each step names the UI it points at; `<html data-coach>` lets CSS light that spot up. */
const STEPS = ['mul', 'wireX', 'const', 'wireY', 'run'] as const;
type Step = (typeof STEPS)[number];

/** How far the player has got in 0-1, read off the graph alone. */
export function coachStep(graph: GraphJSON): Step {
  const kind = new Map(graph.nodes.map((n) => [n.id, n]));
  const into = (id: string) => graph.edges.filter((e) => e.to.node === id);
  const muls = graph.nodes.filter((n) => n.component === 'mul');
  if (!muls.length) return 'mul';
  const fromX = (id: string) =>
    into(id).some((e) => {
      const n = kind.get(e.from.node);
      return n?.component === 'input' && n.attrs?.name === 'x';
    });
  if (!muls.some((m) => fromX(m.id))) return 'wireX';
  const fromMinusOne = (id: string) =>
    into(id).some((e) => {
      const n = kind.get(e.from.node);
      return n?.component === 'const' && Number(n.attrs?.value) === -1;
    });
  if (!muls.some((m) => fromX(m.id) && fromMinusOne(m.id))) return 'const';
  const output = graph.nodes.find((n) => n.component === 'output');
  if (!output || !into(output.id).length) return 'wireY';
  return 'run';
}

/** First-visit walkthrough on 0-1: placing parts, wiring, editing a constant, running tests. */
export function Coach({ t, graph, onSkip }: { t: T; graph: GraphJSON; onSkip: () => void }) {
  const step = coachStep(graph);
  const index = STEPS.indexOf(step);
  useEffect(() => {
    document.documentElement.dataset.coach = step;
    return () => {
      delete document.documentElement.dataset.coach;
    };
  }, [step]);
  return (
    <div className="coach" role="status" aria-live="polite">
      <div className="coach-head">
        <span className="label">
          {t('coach.title')} · {index + 1}/{STEPS.length}
        </span>
        <button type="button" className="coach-skip" onClick={onSkip}>
          {t('coach.skip')}
        </button>
      </div>
      <ol className="coach-dots" aria-hidden="true">
        {STEPS.map((s, i) => (
          <li key={s} className={i < index ? 'is-done' : i === index ? 'is-now' : ''} />
        ))}
      </ol>
      <p className="coach-text">{t(`coach.${step}`)}</p>
    </div>
  );
}
