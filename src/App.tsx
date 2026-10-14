import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import type { GraphJSON, GraphNode } from './engine/graph';
import { applyResult, type Level, type LevelResult, type Progress } from './engine/levels';
import { parseExpression } from './engine/expression';
import { exampleLevels, createExampleRegistry } from './levels/examples';
import { createEditorState, editorReducer, type EditorCommand } from './editor/state';
import {
  createTranslator,
  translateDiagnostic,
  formatNumber,
  locales,
  localeNames,
  type Locale,
} from './i18n';
import { WorkerClient, WorkerClientError } from './worker/client';
import type { WorkerError } from './worker/protocol';
import type { TrainingReport } from './engine/training';
import Canvas, { position } from './ui/Canvas';
import Properties, { defaults } from './ui/Properties';
import LossChart from './ui/LossChart';
import { loadProgress, loadLanguage, saveSave, createThrottledSave } from './storage';
import './ui/theme.css';
import './ui/layout.css';
const registry = createExampleRegistry();
export default function App() {
  const [save, setSave] = useState(() => ({
    progress: loadProgress(),
    settings: { language: loadLanguage() },
  }));
  const saver = useMemo(
    () =>
      createThrottledSave(
        (value: { progress: Progress; settings: { language: Locale } }) =>
          saveSave({ version: 1, ...value }),
        400,
      ),
    [],
  );
  const [level, setLevel] = useState<Level>();
  const [backend, setBackend] = useState('pending');
  const [failure, setFailure] = useState<WorkerError>();
  const [fallback, setFallback] = useState(false);
  const client = useRef<WorkerClient>(null);
  const revision = useRef(0);
  const t = createTranslator(save.settings.language);
  useEffect(() => {
    const c = new WorkerClient();
    client.current = c;
    c.init()
      .then((r) => {
        setBackend(r.backend);
        setFallback(!!r.fallback);
      })
      .catch((e) => setFailure(e.detail ?? { key: 'worker.transport_error', params: {} }));
    return () => c.dispose();
  }, []);
  useEffect(() => {
    document.documentElement.lang = save.settings.language;
    document.title = t('app.title');
    saver.schedule(save);
  }, [save, saver]);
  useEffect(() => {
    const flush = () => saver.flush();
    window.addEventListener('pagehide', flush);
    return () => {
      flush();
      window.removeEventListener('pagehide', flush);
    };
  }, [saver]);
  const progress = (p: Progress) => setSave((s) => ({ ...s, progress: p }));
  return (
    <>
      <header className="topbar">
        <button className="brand btn" onClick={() => setLevel(undefined)}>
          {t('app.title')}
        </button>
        <span className="crumb">{level ? t(level.titleKey) : t('ui.levels')}</span>
        <span className="spacer" />
        <span className="tag">
          {t(`backend.${backend}`)}
          {fallback ? ` · ${t('ui.backendFallback')}` : ''}
        </span>
        <select
          className="field"
          style={{ width: 130 }}
          aria-label={t('ui.language')}
          value={save.settings.language}
          onChange={(e) =>
            setSave((s) => ({ ...s, settings: { language: e.target.value as Locale } }))
          }
        >
          {Object.keys(locales).map((l) => (
            <option key={l} value={l}>
              {localeNames[l as Locale]}
            </option>
          ))}
        </select>
      </header>
      {failure && (
        <p className="status status-fail">{translateDiagnostic(save.settings.language, failure)}</p>
      )}
      {level ? (
        <Editor
          key={level.id}
          level={level}
          progress={save.progress}
          onProgress={progress}
          locale={save.settings.language}
          client={client}
          revision={revision}
        />
      ) : (
        <main className="level-list">
          {exampleLevels.map((l, i) => (
            <section key={l.id}>
              <h2>{t(`chapter.${[0, 1, 3][i]}`)}</h2>
              <button
                className="btn"
                disabled={backend === 'pending' || !save.progress.unlocked.includes(l.id)}
                onClick={() => setLevel(l)}
              >
                <span>{t(l.titleKey)}</span>
                <span>
                  {t(
                    save.progress.passed.includes(l.id)
                      ? 'ui.passed'
                      : save.progress.unlocked.includes(l.id)
                        ? 'ui.playable'
                        : 'ui.locked',
                  )}
                </span>
              </button>
              {save.progress.bestScores[l.id] && (
                <Metrics metrics={save.progress.bestScores[l.id]} t={t} />
              )}
            </section>
          ))}
        </main>
      )}
    </>
  );
}
type EditorProps = {
  level: Level;
  progress: Progress;
  onProgress: (p: Progress) => void;
  locale: Locale;
  client: React.RefObject<WorkerClient | null>;
  revision: React.RefObject<number>;
};
function Metrics({
  metrics,
  t,
}: {
  metrics: LevelResult['metrics'];
  t: ReturnType<typeof createTranslator>;
}) {
  return (
    <dl className="kv">
      {Object.entries(metrics).map(([k, v]) => (
        <Fragment key={k}>
          <dt>{t(`metrics.${k}`)}</dt>
          <dd>{t('ui.number', { value: v })}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
function Editor({ level, progress, onProgress, locale, client, revision }: EditorProps) {
  const [state, setState] = useState(() =>
    createEditorState(level, progress.savedGraphs[level.id]),
  );
  const [diagnostics, setDiagnostics] = useState<WorkerError[]>([]);
  const [error, setError] = useState<WorkerError>();
  const [result, setResult] = useState<LevelResult>();
  const [modal, setModal] = useState(false);
  const [previousBest, setPreviousBest] = useState<LevelResult['metrics']>();
  const [training, setTraining] = useState<'idle' | 'running' | 'paused'>('idle');
  const [report, setReport] = useState<TrainingReport>();
  const [losses, setLosses] = useState<number[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [expression, setExpression] = useState('');
  const [inside, setInside] = useState<{ name: string; graph: GraphJSON }[]>([]);
  const [inputs, setInputs] = useState(() =>
    Object.fromEntries(
      Object.entries(level.interface.inputs).map(([name, shape]) => [
        name,
        JSON.stringify(
          level.kind !== 'train'
            ? (level.fixed[0]?.inputs[name] ?? Array(shape.reduce((a, b) => a * b, 1)).fill(0))
            : Array(shape.reduce((a, b) => a * b, 1)).fill(0),
        ),
      ]),
    ),
  );
  const t = createTranslator(locale);
  const latest = useRef({ progress, onProgress });
  latest.current = { progress, onProgress };
  const busy = training !== 'idle';
  const command = (cmd: EditorCommand) => {
    if (busy || inside.length) return;
    setState((s) => editorReducer(s, cmd, registry));
  };
  const fail = (e: unknown) => {
    const detail =
      e instanceof WorkerClientError
        ? e.detail
        : typeof e === 'object' && e && 'key' in e
          ? (e as WorkerError)
          : { key: 'errors.unknown', params: {} };
    if (detail.key !== 'worker.stale_revision') setError(detail);
  };
  const finish = (r: LevelResult) => {
    setResult(r);
    setTraining('idle');
    if (r.passed) {
      setPreviousBest(latest.current.progress.bestScores[level.id]);
      latest.current.onProgress(applyResult(latest.current.progress, level, r));
      setModal(true);
    }
    if (r.errorKey) setError(r.error ?? { key: r.errorKey, params: {} });
  };
  useEffect(() => {
    const c = client.current;
    if (!c) return;
    return c.subscribe((event) => {
      if (event.type === 'progress') {
        setReport(event.report);
        setLosses((a) => [...a, event.report.loss]);
      } else finish(event.result);
    });
  }, [level, client]);
  useEffect(() => {
    const c = client.current;
    revision.current++;
    c?.setRevision(revision.current);
    setValues({});
    setDiagnostics([]);
    setError(undefined);
    setResult(undefined);
    latest.current.onProgress({
      ...latest.current.progress,
      savedGraphs: { ...latest.current.progress.savedGraphs, [level.id]: state.graph },
    });
    const timer = setTimeout(
      () =>
        c
          ?.compile(level.id, state.graph)
          .then((r) => setDiagnostics(r.diagnostics))
          .catch(fail),
      250,
    );
    return () => clearTimeout(timer);
  }, [state.graph]);
  useEffect(
    () => () => {
      void client.current?.stop().catch(() => {});
      revision.current++;
      client.current?.setRevision(revision.current);
    },
    [],
  );
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input,textarea,select') || inside.length) return;
      const mod = e.metaKey || e.ctrlKey;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        command({ type: 'removeNodes' });
      }
      if (mod && ['z', 'c', 'v'].includes(e.key.toLowerCase())) {
        e.preventDefault();
        command({
          type:
            e.key.toLowerCase() === 'z'
              ? e.shiftKey
                ? 'redo'
                : 'undo'
              : e.key.toLowerCase() === 'c'
                ? 'copy'
                : 'paste',
        });
      }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [busy, inside.length]);
  const selected = state.graph.nodes.find((n) => n.id === state.selection[0]);
  const display = inside.at(-1)?.graph ?? state.graph;
  const errors = diagnostics.map((e) => ({ ...e, message: translateDiagnostic(locale, e) }));
  const inspect = (node: GraphNode) => {
    const def = registry.get(node.component);
    if (!def?.subgraph) return;
    try {
      const graph =
        typeof def.subgraph === 'function' ? def.subgraph(node.attrs ?? {}) : def.subgraph;
      setInside((s) => [
        ...s,
        {
          name: node.component,
          graph: {
            ...graph,
            nodes: graph.nodes.map((n, i) => ({
              ...n,
              ui: { x: 50 + (i % 4) * 210, y: 70 + Math.floor(i / 4) * 150 },
            })),
          },
        },
      ]);
    } catch (e) {
      fail(e);
    }
  };
  async function probe() {
    const probeRevision = revision.current;
    try {
      const data = Object.fromEntries(
        Object.entries(inputs).map(([key, value]) => {
          const a = JSON.parse(value);
          if (!Array.isArray(a) || !a.every((v) => typeof v === 'number' && Number.isFinite(v)))
            throw new Error();
          return [key, a];
        }),
      );
      const c = client.current;
      if (!c) return;
      const next: Record<string, string> = {};
      for (const node of state.graph.nodes) {
        const port = registry.get(node.component)?.outputs[0];
        if (!port) continue;
        if (revision.current !== probeRevision) return;
        const r = await c.probe(level.id, state.graph, { node: node.id, port }, data);
        if (revision.current !== probeRevision) return;
        next[node.id] = `[${r.shape.join('×')}] ${r.values
          .slice(0, 5)
          .map((n) => formatNumber(locale, n, { maximumSignificantDigits: 4 }))
          .join(', ')}${r.values.length > 5 ? '…' : ''}`;
      }
      setValues(next);
    } catch (e) {
      fail(e);
    }
  }
  const run = async () => {
    setError(undefined);
    setResult(undefined);
    try {
      if (level.kind === 'train') {
        setLosses([]);
        setReport(undefined);
        setTraining('running');
        await client.current?.start(level.id, state.graph);
      } else {
        const r = await client.current?.evaluate(level.id, state.graph);
        if (r) finish(r);
      }
    } catch (e) {
      setTraining('idle');
      fail(e);
    }
  };
  return (
    <main className="workspace">
      <aside className="panel panel-left">
        <h1 className="level-title">{t(level.titleKey)}</h1>
        <p>{t(level.descriptionKey)}</p>
        <p className="muted">{t('ui.help')}</p>
        {['primitive', 'player'].map((group) => (
          <section className="section" key={group}>
            <h2 className="label">{t(group === 'primitive' ? 'ui.primitives' : 'ui.player')}</h2>
            <div className="palette">
              {level.available
                .filter((name) => !!registry.get(name)?.subgraph === (group === 'player'))
                .map((name) => (
                  <button
                    className={`palette-item ${group === 'player' ? 'is-player' : ''}`}
                    key={name}
                    draggable={!busy && !inside.length}
                    onDragStart={(e) => e.dataTransfer.setData('application/x-component', name)}
                    disabled={busy || !!inside.length}
                    onClick={() =>
                      command({
                        type: 'addNode',
                        component: name,
                        attrs: defaults(name),
                        position: { x: 260, y: 80 + state.graph.nodes.length * 30 },
                      })
                    }
                  >
                    {t(`components.${name}`)}
                  </button>
                ))}
            </div>
          </section>
        ))}
        {(level.unlockComponents ?? []).map((name) => (
          <div className="palette-item is-locked" key={name}>
            {t(`components.${name}`)} · {t('ui.locked')}
          </div>
        ))}
        <div className="btn-row">
          <button
            className="btn"
            disabled={busy || !!inside.length || !state.past.length}
            onClick={() => command({ type: 'undo' })}
          >
            {t('ui.undo')}
          </button>
          <button
            className="btn"
            disabled={busy || !!inside.length || !state.future.length}
            onClick={() => command({ type: 'redo' })}
          >
            {t('ui.redo')}
          </button>
        </div>
        {!inside.length && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              try {
                const variables = Object.fromEntries(
                  state.graph.nodes
                    .filter((n) => n.component === 'input')
                    .map((n) => [String(n.attrs?.name), { node: n.id, port: 'out' }]),
                );
                // Every available single-output component can be called by name, e.g. exp(x).
                const functions = Object.fromEntries(
                  level.available.flatMap((name) => {
                    const def = registry.get(name);
                    return def && def.inputs.length > 0 && def.outputs.length === 1
                      ? [[name, { component: name, inputs: def.inputs, output: def.outputs[0] }]]
                      : [];
                  }),
                );
                const fragment = parseExpression(expression, {
                  variables,
                  functions,
                  availableComponents: level.available,
                });
                const y = Math.max(...state.graph.nodes.map((n) => position(n).y), 0) + 160;
                command({
                  type: 'insertExpression',
                  expression: fragment,
                  position: { x: 60, y },
                  to: {
                    node: state.graph.nodes.find((n) => n.component === 'output')!.id,
                    port: 'x',
                  },
                });
                setExpression('');
              } catch (e) {
                fail(e);
              }
            }}
          >
            <label>
              {t('ui.expression')}
              <input
                className="field"
                value={expression}
                disabled={busy}
                onChange={(e) => setExpression(e.target.value)}
              />
            </label>
            <button className="btn" disabled={busy}>
              {t('ui.insert')}
            </button>
          </form>
        )}
        {inside.length > 0 && (
          <button className="btn" onClick={() => setInside((s) => s.slice(0, -1))}>
            {t('ui.back')} / {inside.map((v) => t(`components.${v.name}`)).join(' / ')} (
            {t('ui.readonly')})
          </button>
        )}
      </aside>
      <Canvas
        viewport={state.viewport}
        onViewport={(viewport) =>
          setState((s) => editorReducer(s, { type: 'setViewport', viewport }, registry))
        }
        graph={display}
        selected={inside.length ? [] : state.selection}
        readonly={busy || !!inside.length}
        t={t}
        errors={inside.length ? [] : errors}
        values={inside.length ? {} : values}
        onSelect={(ids, shift) => command({ type: 'select', ids, shift })}
        onMove={(positions) => command({ type: 'moveNodes', positions })}
        onAdd={(component, x, y) =>
          command({ type: 'addNode', component, attrs: defaults(component), position: { x, y } })
        }
        onDeleteEdge={(index) => command({ type: 'disconnect', edge: state.graph.edges[index] })}
        onConnect={(from, to, previous) => {
          if (previous) {
            const edge = state.graph.edges.find(
              (e) => e.to.node === previous.node && e.to.port === previous.port,
            );
            if (edge) command({ type: 'rewire', edge, from, to });
          } else command({ type: 'connect', from, to });
        }}
        onInspect={inspect}
      />
      <aside className="panel panel-right">
        {(error || state.error) && (
          <p role="alert" className="status status-fail">
            {translateDiagnostic(locale, error ?? { key: state.error! })}
          </p>
        )}
        {errors
          .filter((e) => !e.node)
          .map((e, i) => (
            <p className="err" key={i}>
              {e.message}
            </p>
          ))}
        {selected && !inside.length && !busy && (
          <Properties
            key={selected.id}
            node={selected}
            t={t}
            onChange={(attrs) =>
              command({ type: 'setAttrs', id: selected.id, attrs, replace: true })
            }
          />
        )}
        {level.kind !== 'train' && (
          <section className="section">
            <h2 className="label">{t('ui.probe')}</h2>
            {Object.entries(inputs).map(([name, value]) => (
              <label key={name}>
                {t(`ports.${name}`)}
                <input
                  className="field"
                  value={value}
                  onChange={(e) => setInputs((s) => ({ ...s, [name]: e.target.value }))}
                />
                {level.interface.inputs[name].length === 0 && (
                  <input
                    type="range"
                    min={-10}
                    max={10}
                    step={0.1}
                    value={(() => {
                      try {
                        return JSON.parse(value)[0];
                      } catch {
                        return 0;
                      }
                    })()}
                    onChange={(e) =>
                      setInputs((s) => ({ ...s, [name]: JSON.stringify([Number(e.target.value)]) }))
                    }
                  />
                )}
              </label>
            ))}
            <button className="btn" onClick={() => void probe()}>
              {t('ui.probe')}
            </button>
          </section>
        )}
        <section>
          <h2 className="label">{t(level.kind === 'train' ? 'ui.training' : 'ui.tests')}</h2>
          <div className="btn-row">
            <button className="btn btn-primary" disabled={busy} onClick={() => void run()}>
              {t(level.kind === 'train' ? 'ui.start' : 'ui.runTests')}
            </button>
            {busy && (
              <>
                <button
                  className="btn"
                  onClick={() => {
                    const next = training === 'running' ? 'paused' : 'running';
                    void (next === 'paused' ? client.current?.pause() : client.current?.resume())
                      ?.then(() => setTraining(next))
                      .catch(fail);
                  }}
                >
                  {t(training === 'running' ? 'ui.pause' : 'ui.resume')}
                </button>
                <button
                  className="btn"
                  onClick={() =>
                    void client.current
                      ?.stop()
                      .then(() => setTraining('idle'))
                      .catch(fail)
                  }
                >
                  {t('ui.stop')}
                </button>
              </>
            )}
          </div>
          {level.kind === 'train' && (
            <>
              <LossChart losses={losses} label={t('ui.loss')} />
              {report && (
                <p>
                  {t('ui.report', {
                    step: report.step,
                    loss: report.loss,
                    accuracy: report.accuracy ?? 0,
                  })}
                </p>
              )}
            </>
          )}
          {result && (
            <>
              <p className={`status ${result.passed ? 'status-pass' : 'status-fail'}`}>
                {t(result.passed ? 'ui.passed' : 'ui.failed')}
              </p>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t('ui.case')}</th>
                    <th>{t('ui.expected')}</th>
                    <th>{t('ui.actual')}</th>
                    <th>{t('ui.result')}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.tests.map((r, i) => (
                    <tr key={r.id}>
                      <td>{t('ui.number', { value: i + 1 })}</td>
                      <td>{r.expected.map((n) => formatNumber(locale, n)).join(', ')}</td>
                      <td>{r.actual.map((n) => formatNumber(locale, n)).join(', ')}</td>
                      <td className={r.passed ? 'ok' : 'err'}>
                        {t(r.passed ? 'ui.passed' : 'ui.failed')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
      </aside>
      {modal && result && (
        <div className="modal-back">
          <section className="modal" role="dialog" aria-modal="true" aria-label={t('ui.passed')}>
            <h2>{t('ui.passed')}</h2>
            <p>
              {t('ui.unlocked', {
                components:
                  (level.unlockComponents ?? []).map((c) => t(`components.${c}`)).join(', ') ||
                  t('ui.none'),
              })}
            </p>
            <Metrics metrics={result.metrics} t={t} />
            {previousBest && (
              <>
                <h3>{t('ui.previousBest')}</h3>
                <Metrics metrics={previousBest} t={t} />
              </>
            )}
            <button className="btn btn-primary" onClick={() => setModal(false)}>
              {t('ui.continue')}
            </button>
          </section>
        </div>
      )}
    </main>
  );
}
