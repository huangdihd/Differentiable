import { useEffect, useMemo, useRef, useState } from 'react';
import type { GraphJSON, GraphNode } from './engine/graph';
import { applyResult, type Level, type LevelResult, type Progress } from './engine/levels';
import { parseExpression } from './engine/expression';
import { buildPlayerRegistry, availableComponents } from './levels/registry';
import { CHAPTERS, catalog, entryFor, gameLevels } from './levels/catalog';
import {
  connectionError,
  createEditorState,
  editorReducer,
  type EditorCommand,
} from './editor/state';
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
import Canvas, { position, shapeText, type CanvasHandle, type NodeValue } from './ui/Canvas';
import { Random } from './engine/random';
import Properties, { defaults } from './ui/Properties';
import LossChart from './ui/LossChart';
import { caseInputs, fullValues } from './ui/tensor-values';
import TensorGrid from './ui/TensorGrid';
import { expectedOutput, matchesExpected, offIndices } from './ui/expected-output';
import { NODE_W, SYMBOLS } from './ui/graph-geometry';
import {
  loadProgress,
  isExpressionUnlocked,
  loadProgressWithMigration,
  loadLanguage,
  saveSave,
  createThrottledSave,
} from './storage';
import './ui/theme.css';
import './ui/layout.css';

type T = ReturnType<typeof createTranslator>;

export default function App() {
  const [save, setSave] = useState(() => ({
    progress: loadProgress(),
    settings: { language: loadLanguage() },
  }));
  const saver = useMemo(
    () =>
      createThrottledSave(
        (value: { progress: Progress; settings: { language: Locale } }) =>
          saveSave({ version: 2, ...value }),
        400,
      ),
    [],
  );
  const [loaded, setLoaded] = useState(false);
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
    let active = true;
    loadProgressWithMigration(gameLevels)
      .then(async (progress) => {
        if (!active) return undefined;
        setSave((s) => ({ ...s, progress }));
        setLoaded(true);
        return c.init();
      })
      .then((r) => {
        if (!active || !r) return;
        setBackend(r.backend);
        setFallback(!!r.fallback);
      })
      .catch((e) => setFailure(e.detail ?? { key: 'worker.transport_error', params: {} }));
    return () => {
      active = false;
      c.dispose();
    };
  }, []);
  useEffect(() => {
    document.documentElement.lang = save.settings.language;
    document.title = level ? `${t(level.titleKey)} · ${t('app.title')}` : t('app.title');
    if (loaded) saver.schedule(save);
  }, [save, saver, level, loaded]);
  useEffect(() => {
    const flush = () => saver.flush();
    window.addEventListener('pagehide', flush);
    return () => {
      flush();
      window.removeEventListener('pagehide', flush);
    };
  }, [saver]);
  const progress = (p: Progress) => setSave((s) => ({ ...s, progress: p }));
  const entry = level && entryFor(level);
  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => setLevel(undefined)}>
          <span className="brand-mark">∂</span>
          {t('app.title')}
        </button>
        <span className="crumb">
          {entry ? (
            <>
              {t('ui.chapterNamed', { n: entry.chapter, name: t(`chapter.${entry.chapter}`) })} /{' '}
              <b>
                {entry.code} {t(entry.level.titleKey)}
              </b>
            </>
          ) : (
            t('ui.levels')
          )}
        </span>
        <span className="spacer" />
        <span
          className={`tag ${backend === 'webgpu' ? 'tag-accent' : ''}`}
          title={fallback ? t('ui.backendFallback') : undefined}
        >
          {t(`backend.${backend}`)}
        </span>
        <select
          className="field lang-select"
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
        <p className="status status-fail banner">
          {translateDiagnostic(save.settings.language, failure)}
        </p>
      )}
      {level ? (
        <Editor
          key={level.id}
          level={level}
          progress={save.progress}
          onProgress={progress}
          onExit={() => setLevel(undefined)}
          locale={save.settings.language}
          backend={backend}
          client={client}
          revision={revision}
        />
      ) : (
        <Home t={t} progress={save.progress} ready={backend !== 'pending'} onOpen={setLevel} />
      )}
    </div>
  );
}

function Home({
  t,
  progress,
  ready,
  onOpen,
}: {
  t: T;
  progress: Progress;
  ready: boolean;
  onOpen: (level: Level) => void;
}) {
  return (
    <main className="home blueprint-bg">
      <div className="home-inner">
        <section className="hero">
          <h1>
            {t('home.titleA')}
            <span className="hero-accent">a + b</span>
            {t('home.titleB')}
            <br />
            {t('home.titleC')}
          </h1>
          <p className="muted">{t('home.lead')}</p>
        </section>
        <ol className="road">
          {CHAPTERS.map((chapter) => {
            const entries = catalog.filter((e) => e.chapter === chapter);
            return (
              <li
                key={chapter}
                className={`road-chapter ${entries.length ? 'has-levels' : ''} ${chapter === 6 ? 'is-goal' : ''}`}
              >
                <div className="road-head">
                  <span className="road-n">CH {chapter}</span>
                  <span className="road-name">{t(`chapter.${chapter}`)}</span>
                  <span className="road-desc">{t(`chapter.${chapter}.desc`)}</span>
                </div>
                <div className="road-levels">
                  {entries.length === 0 && <span className="road-soon">{t('ui.comingSoon')}</span>}
                  {entries.map(({ level, code }) => {
                    const passed = progress.passed.includes(level.id);
                    const unlocked = progress.unlocked.includes(level.id);
                    const best = progress.bestScores[level.id];
                    return (
                      <button
                        key={level.id}
                        className={`level-card ${passed ? 'is-passed' : ''}`}
                        disabled={!ready || !unlocked}
                        onClick={() => onOpen(level)}
                      >
                        <span className="level-card-top">
                          <span className="level-id">{code}</span>
                          <span className="level-state">
                            {passed ? '✓ ' : ''}
                            {t(passed ? 'ui.passed' : unlocked ? 'ui.playable' : 'ui.locked')}
                          </span>
                        </span>
                        <span className="level-card-title">{t(level.titleKey)}</span>
                        {best && (
                          <span className="level-card-best">
                            {t('ui.bestShort', {
                              nodes: best.expandedPrimitives,
                              params: best.parameterCount,
                            })}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </main>
  );
}

type EditorProps = {
  level: Level;
  progress: Progress;
  onProgress: (p: Progress) => void;
  onExit: () => void;
  locale: Locale;
  backend: string;
  client: React.RefObject<WorkerClient | null>;
  revision: React.RefObject<number>;
};

/** Time each test case stays on the canvas during step-through verification. */
const STEP_MS = 420;

function compact(locale: Locale, n: number) {
  if (!Number.isFinite(n)) return Number.isNaN(n) ? 'NaN' : n > 0 ? '∞' : '−∞';
  if (n !== 0 && (Math.abs(n) >= 1e5 || Math.abs(n) < 1e-3)) return n.toExponential(2);
  // -0 (e.g. 0 · −1.5) is the same number; showing a sign there only invites questions.
  return formatNumber(locale, n === 0 ? 0 : n, { maximumSignificantDigits: 4 });
}
function listText(locale: Locale, values: number[], max = 5) {
  return (
    values
      .slice(0, max)
      .map((n) => compact(locale, n))
      .join(', ') + (values.length > max ? ', …' : '')
  );
}

function Editor({
  level,
  progress,
  onProgress,
  onExit,
  locale,
  backend,
  client,
  revision,
}: EditorProps) {
  const entry = entryFor(level);
  const registry = useMemo(
    () => buildPlayerRegistry(gameLevels, progress),
    [progress.passedSolutions, progress.unlockedComponents],
  );
  const available = availableComponents(level, progress);
  const [state, setState] = useState(() =>
    createEditorState(level, progress.savedGraphs[level.id], available),
  );
  const [diagnostics, setDiagnostics] = useState<WorkerError[]>([]);
  const [error, setError] = useState<WorkerError>();
  const [result, setResult] = useState<LevelResult>();
  const [running, setRunning] = useState(false);
  const [modal, setModal] = useState(false);
  const [previousBest, setPreviousBest] = useState<LevelResult['metrics']>();
  const [training, setTraining] = useState<'idle' | 'running' | 'paused'>('idle');
  const [report, setReport] = useState<TrainingReport>();
  const [losses, setLosses] = useState<number[]>([]);
  const [values, setValues] = useState<Record<string, NodeValue>>({});
  const [expression, setExpression] = useState('');
  const [inside, setInside] = useState<{ name: string; graph: GraphJSON }[]>([]);
  const [shake, setShake] = useState<{ node: string; tick: number }>();
  const [ghost, setGhost] = useState<{ component: string; x: number; y: number }>();
  const canvas = useRef<CanvasHandle>(null);
  // Every level probes values: training levels show what the initial (seeded) network computes.
  const probing = true;
  // Step-through verification, Turing Complete style: one test case at a time on the canvas.
  const [hintsShown, setHintsShown] = useState(0);
  const [verify, setVerify] = useState<{ index: number; result: LevelResult } | null>(null);
  const skip = useRef(false);
  const testCases = useMemo(() => {
    if (level.kind === 'train') return [];
    const generated = level.generated;
    const random = generated ? new Random(generated.seed) : undefined;
    return [
      ...level.fixed,
      ...(generated
        ? Array.from({ length: generated.count }, (_, i) => generated.generate(random!, i))
        : []),
    ];
  }, [level]);
  const [inputCase, setInputCase] = useState('0');
  const hasTensorInputs = Object.values(level.interface.inputs).some((shape) => shape.length > 0);
  const selectInputCase = (index: string) => {
    setInputCase(index);
    if (index !== '') setInputs(caseInputs(testCases[Number(index)].inputs));
  };
  const [inputs, setInputs] = useState(() =>
    Object.fromEntries(
      Object.entries(level.interface.inputs).map(([name, shape]) => [
        name,
        (level.kind !== 'train'
          ? (level.fixed[0]?.inputs[name] ?? Array(shape.reduce((a, b) => a * b, 1)).fill(0))
          : Array(shape.reduce((a, b) => a * b, 1)).fill(0)
        ).join(', '),
      ]),
    ),
  );
  useEffect(() => {
    if (level.kind !== 'train') return;
    let alive = true;
    const batch = level.dataset(new Random(level.seed));
    void Promise.all(
      Object.entries(batch.inputs).map(async ([k, v]) => [k, Array.from(await v.read())] as const),
    )
      .then((entries) => {
        if (alive) setInputs(Object.fromEntries(entries.map(([k, v]) => [k, v.join(', ')])));
      })
      .finally(() => {
        Object.values(batch.inputs).forEach((v) => v.dispose());
        batch.targets.dispose();
      });
    return () => {
      alive = false;
    };
  }, [level]);
  const t = createTranslator(locale);
  const latest = useRef({ progress, onProgress, graph: state.graph });
  latest.current = { progress, onProgress, graph: state.graph };
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
  const submittedGraph = useRef(state.graph);
  const finish = (r: LevelResult) => {
    setResult(r);
    setTraining('idle');
    if (r.passed) {
      setPreviousBest(latest.current.progress.bestScores[level.id]);
      latest.current.onProgress(
        applyResult(latest.current.progress, level, r, submittedGraph.current),
      );
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
  const lastGraph = useRef(state.graph);
  useEffect(() => {
    const c = client.current;
    if (lastGraph.current !== state.graph) {
      setResult(undefined);
      setModal(false);
    }
    lastGraph.current = state.graph;
    revision.current++;
    void c?.syncProgress(latest.current.progress, revision.current).catch(fail);
    setDiagnostics([]);
    setError(undefined);
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
  }, [state.graph, registry]);
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
      if (e.key === 'Escape') {
        setModal(false);
        return;
      }
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

  const parsedInputs = useMemo(() => {
    try {
      return Object.fromEntries(
        Object.entries(inputs).map(([key, text]) => {
          const a = text
            .split(/[\s,]+/)
            .filter(Boolean)
            .map(Number);
          const size = level.interface.inputs[key].reduce((x, y) => x * y, 1);
          if (a.length !== size || !a.every(Number.isFinite)) throw new Error(key);
          return [key, a];
        }),
      ) as Record<string, number[]>;
    } catch (e) {
      return { invalid: (e as Error).message };
    }
  }, [inputs, level]);

  // The answer for the same inputs, so the player can compare without running the tests.
  const [expected, setExpected] = useState<{ shape: number[]; values: number[] }>();
  useEffect(() => {
    setExpected(undefined);
    if (level.kind === 'train' || 'invalid' in parsedInputs) return;
    let cancelled = false;
    expectedOutput(level, parsedInputs as Record<string, number[]>).then(
      (r) => !cancelled && setExpected(r),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, parsedInputs]);

  // Live probe: every node shows its output for the probe inputs, refreshed as the graph changes.
  useEffect(() => {
    if (!probing || busy || 'invalid' in parsedInputs) {
      setValues({});
      return;
    }
    const probeRevision = revision.current;
    let cancelled = false;
    const timer = setTimeout(
      async () => {
        const c = client.current;
        if (!c) return;
        const next: Record<string, NodeValue> = {};
        for (const node of state.graph.nodes) {
          const port =
            node.component === 'output'
              ? null
              : node.component === 'input'
                ? 'out'
                : registry.get(node.component)?.outputs[0];
          const source =
            node.component === 'output'
              ? state.graph.edges.find((e) => e.to.node === node.id)?.from
              : port
                ? { node: node.id, port }
                : undefined;
          if (!source) continue;
          if (diagnostics.some((d) => d.node === source.node)) continue;
          try {
            const r = await c.probe(
              level.id,
              state.graph,
              source,
              parsedInputs as Record<string, number[]>,
            );
            if (cancelled || revision.current !== probeRevision) return;
            const shown =
              r.values.length === 1 ? compact(locale, r.values[0]) : listText(locale, r.values, 2);
            next[node.id] = {
              text: shown,
              title: `[${r.shape.join('×') || '·'}] ${fullValues(r.values)}`,
              ok: r.values.every(Number.isFinite),
              shape: r.shape,
            };
          } catch {
            if (cancelled || revision.current !== probeRevision) return;
          }
        }
        setValues(next);
      },
      verify ? 10 : 300,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [state.graph, parsedInputs, diagnostics, busy, locale]);

  const selected = state.graph.nodes.find((n) => n.id === state.selection[0]);
  const display = inside.at(-1)?.graph ?? state.graph;
  // "Not wired yet" is unfinished work, not a mistake: shown as a quiet note instead of in red.
  const SOFT = new Set(['missing_port', 'missing_input']);
  const errors = diagnostics.map((e) => ({
    ...e,
    soft: SOFT.has(e.key),
    message: translateDiagnostic(locale, e),
  }));
  // Component internals get their own framing; leaving restores the player's view.
  const outerView = useRef(state.viewport);
  useEffect(() => {
    if (inside.length) requestAnimationFrame(() => canvas.current?.fit());
  }, [inside.length]);
  const leaveInside = () => {
    const next = inside.slice(0, -1);
    setInside(next);
    if (!next.length)
      setState((st) =>
        editorReducer(st, { type: 'setViewport', viewport: outerView.current }, registry),
      );
  };
  const inspect = (node: GraphNode) => {
    if (!inside.length) outerView.current = state.viewport;
    const def = registry.get(node.component);
    if (!def?.subgraph) return;
    try {
      const graph =
        typeof def.subgraph === 'function' ? def.subgraph(node.attrs ?? {}) : def.subgraph;
      setInside((s) => [...s, { name: node.component, graph }]);
    } catch (e) {
      fail(e);
    }
  };
  const run = async () => {
    submittedGraph.current = structuredClone(state.graph);
    setError(undefined);
    setResult(undefined);
    try {
      if (level.kind === 'train') {
        setLosses([]);
        setReport(undefined);
        setTraining('running');
        await client.current?.start(level.id, state.graph);
      } else {
        setRunning(true);
        const r = await client.current?.evaluate(level.id, state.graph);
        if (!r) return;
        if (r.errorKey || !r.tests.length) {
          finish(r);
          return;
        }
        skip.current = false;
        for (let i = 0; i < r.tests.length; i++) {
          setVerify({ index: i, result: r });
          const c = testCases[i];
          if (c) {
            setInputCase(String(i));
            setInputs(
              Object.fromEntries(Object.entries(c.inputs).map(([k, v]) => [k, v.join(', ')])),
            );
          }
          // Stop on the first failing case and leave its values on the canvas to debug.
          if (!r.tests[i].passed) break;
          if (!skip.current) await new Promise((done) => setTimeout(done, STEP_MS));
        }
        setVerify(null);
        finish(r);
      }
    } catch (e) {
      setTraining('idle');
      fail(e);
    } finally {
      setRunning(false);
    }
  };

  // Palette: press on an item and drag it onto the canvas; a plain click drops it near the middle.
  const startPaletteDrag = (component: string, e: React.PointerEvent) => {
    if (busy || inside.length || e.button !== 0) return;
    e.preventDefault();
    const sx = e.clientX,
      sy = e.clientY;
    let moved = false;
    setGhost({ component, x: sx, y: sy });
    const move = (ev: PointerEvent) => {
      if (Math.hypot(ev.clientX - sx, ev.clientY - sy) > 3) moved = true;
      setGhost({ component, x: ev.clientX, y: ev.clientY });
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setGhost(undefined);
      const at = moved
        ? canvas.current?.toGraph(ev.clientX, ev.clientY)
        : (() => {
            const r = document.querySelector('.canvas')?.getBoundingClientRect();
            return r
              ? canvas.current?.toGraph(r.left + r.width * 0.45, r.top + r.height * 0.45)
              : null;
          })();
      if (!at) return;
      command({
        type: 'addNode',
        component,
        attrs: defaults(component),
        position: { x: at.x - NODE_W / 2, y: at.y - 13 },
      });
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const insertExpression = () => {
    try {
      const variables = Object.fromEntries(
        state.graph.nodes
          .filter((n) => n.component === 'input')
          .map((n) => [String(n.attrs?.name), { node: n.id, port: 'out' }]),
      );
      // Every available single-output component can be called by name, e.g. exp(x).
      const functions = Object.fromEntries(
        available.flatMap((name) => {
          const def = registry.get(name);
          return def && def.inputs.length > 0 && def.outputs.length === 1
            ? [[name, { component: name, inputs: def.inputs, output: def.outputs[0] }]]
            : [];
        }),
      );
      const fragment = parseExpression(expression, {
        variables,
        functions,
        availableComponents: available,
      });
      const y = Math.max(...state.graph.nodes.map((n) => position(n).y), 0) + 160;
      command({
        type: 'insertExpression',
        expression: fragment,
        position: { x: 60, y },
        to: { node: state.graph.nodes.find((n) => n.component === 'output')!.id, port: 'x' },
      });
      setExpression('');
    } catch (e) {
      fail(e);
    }
  };

  const playerComponents = available.filter((name) => progress.unlockedComponents.includes(name));
  const presets = available.filter(
    (name) => !playerComponents.includes(name) && (level.presets ?? []).includes(name),
  );
  const primitives = available.filter(
    (name) => !playerComponents.includes(name) && !presets.includes(name),
  );
  const outputValue = values[state.graph.nodes.find((n) => n.component === 'output')?.id ?? ''];
  const outputActual = outputValue?.shape && {
    shape: outputValue.shape,
    values: outputValue.title
      .replace(/^\[[^\]]*\]\s*/, '')
      .split(', ')
      .map(Number),
  };
  const canEdit = !busy && !inside.length;
  // The empty-canvas hint is for the very first look only: any edit (even wiring x straight to y)
  // or any history at all hides it for good.
  const untouched =
    !state.graph.edges.length &&
    !state.past.length &&
    state.graph.nodes.every((n) => ['input', 'output'].includes(n.component));
  const threshold = level.kind === 'train' ? level.threshold.loss : undefined;
  // The expression box is a tool you earn at the end of chapter 1; early levels are about wiring.
  const expressionUnlocked = isExpressionUnlocked(progress, gameLevels);
  const ioShapes = Object.fromEntries(
    state.graph.nodes.flatMap((n) =>
      n.component === 'input'
        ? [[n.id, level.interface.inputs[String(n.attrs?.name)] ?? []]]
        : n.component === 'output'
          ? [[n.id, level.interface.outputShape]]
          : [],
    ),
  );
  const shown = verify ? verify.result : result;
  const shownTests = verify ? verify.result.tests.slice(0, verify.index + 1) : result?.tests;

  return (
    <main className="workspace">
      <aside className="panel panel-left">
        <div className="section">
          <div className="level-id">LEVEL {entry.code}</div>
          <h1 className="level-title">{t(level.titleKey)}</h1>
          <p className="level-desc">{t(level.descriptionKey)}</p>
          {level.par !== undefined && <p className="hint">{t('ui.par', { n: level.par })}</p>}
          {level.hintsKeys?.slice(0, hintsShown).map((key, i) => (
            <p className="hint hint-open" key={key}>
              <span className="hint-n">{i + 1}</span>
              {t(key)}
            </p>
          ))}
          {hintsShown < (level.hintsKeys?.length ?? 0) && (
            <button className="hint-toggle" onClick={() => setHintsShown((n) => n + 1)}>
              {t('ui.showHint', { n: hintsShown + 1 })}
            </button>
          )}
        </div>
        <section className="section">
          <h2 className="label">{t('ui.primitives')}</h2>
          <div className="palette">
            {primitives.map((name) => (
              <PaletteItem
                key={name}
                name={name}
                t={t}
                disabled={!canEdit}
                onPress={startPaletteDrag}
              />
            ))}
          </div>
        </section>
        {(playerComponents.length > 0 || (level.unlockComponents ?? []).length > 0) && (
          <section className="section">
            <h2 className="label">{t('ui.player')}</h2>
            <div className="palette">
              {playerComponents.map((name) => (
                <PaletteItem
                  key={name}
                  name={name}
                  t={t}
                  player
                  disabled={!canEdit}
                  onPress={startPaletteDrag}
                />
              ))}
              {(level.unlockComponents ?? []).map((name) => (
                <div className="palette-item is-locked" key={name} title={t('ui.producedHere')}>
                  <span className="sym">{SYMBOLS[name] ?? '?'}</span>
                  <span className="name">
                    {t(`components.${name}`)} · {t('ui.producedHere')}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}
        {presets.length > 0 && (
          <section className="section">
            <h2 className="label">{t('ui.presets')}</h2>
            <div className="palette">
              {presets.map((name) => (
                <PaletteItem
                  key={name}
                  name={name}
                  t={t}
                  disabled={!canEdit}
                  onPress={startPaletteDrag}
                />
              ))}
            </div>
          </section>
        )}
        {expressionUnlocked && (
          <form
            className="section expression"
            onSubmit={(e) => {
              e.preventDefault();
              insertExpression();
            }}
          >
            <label className="label" htmlFor="expression">
              {t('ui.expression')}
            </label>
            <div className="expression-row">
              <input
                id="expression"
                className="field"
                value={expression}
                placeholder={t('ui.expressionPlaceholder')}
                disabled={!canEdit}
                autoComplete="off"
                spellCheck={false}
                onChange={(e) => setExpression(e.target.value)}
              />
              <button className="btn" disabled={!canEdit || !expression.trim()}>
                ↵
              </button>
            </div>
            <p className="hint">{t('ui.expressionHelp')}</p>
          </form>
        )}
        <p className="hint help">{t('ui.help')}</p>
      </aside>

      <section className="canvas-wrap">
        <Canvas
          ref={canvas}
          registry={registry}
          viewport={state.viewport}
          onViewport={(viewport) =>
            setState((s) => editorReducer(s, { type: 'setViewport', viewport }, registry))
          }
          graph={display}
          selected={inside.length ? [] : state.selection}
          readonly={!canEdit}
          t={t}
          errors={inside.length ? [] : errors}
          values={inside.length ? null : values}
          ioShapes={inside.length ? undefined : ioShapes}
          onSelect={(ids, shift) => command({ type: 'select', ids, shift })}
          onMove={(positions) => command({ type: 'moveNodes', positions })}
          onDeleteEdge={(index) => command({ type: 'disconnect', edge: state.graph.edges[index] })}
          onDeleteNode={(id) => {
            command({ type: 'select', ids: [id], shift: false });
            command({ type: 'removeNodes' });
          }}
          onConnect={(from, to, previous) => {
            const base = {
              ...state.graph,
              edges: state.graph.edges.filter(
                (e) =>
                  !(e.to.node === to.node && e.to.port === to.port) &&
                  !(previous && e.to.node === previous.node && e.to.port === previous.port),
              ),
            };
            if (connectionError(base, { from, to }, registry) === 'cycle') {
              setShake((s) => ({ node: to.node, tick: (s?.tick ?? 0) + 1 }));
              setError({ key: 'cycle', params: {} });
              return;
            }
            if (previous) {
              const edge = state.graph.edges.find(
                (e) => e.to.node === previous.node && e.to.port === previous.port,
              );
              if (edge) command({ type: 'rewire', edge, from, to });
            } else command({ type: 'connect', from, to });
          }}
          onInspect={inspect}
          shake={shake}
        />
        <div className="canvas-toolbar">
          <button
            className="tool-btn"
            title={t('ui.undo')}
            aria-label={t('ui.undo')}
            disabled={!canEdit || !state.past.length}
            onClick={() => command({ type: 'undo' })}
          >
            ↶
          </button>
          <button
            className="tool-btn"
            title={t('ui.redo')}
            aria-label={t('ui.redo')}
            disabled={!canEdit || !state.future.length}
            onClick={() => command({ type: 'redo' })}
          >
            ↷
          </button>
          <span className="tool-sep" />
          <button
            className="tool-btn"
            title={t('ui.zoomOut')}
            aria-label={t('ui.zoomOut')}
            onClick={() =>
              setState((s) =>
                editorReducer(
                  s,
                  {
                    type: 'setViewport',
                    viewport: { ...s.viewport, zoom: Math.max(0.25, s.viewport.zoom / 1.2) },
                  },
                  registry,
                ),
              )
            }
          >
            −
          </button>
          <span className="tool-zoom">{Math.round(state.viewport.zoom * 100)}%</span>
          <button
            className="tool-btn"
            title={t('ui.zoomIn')}
            aria-label={t('ui.zoomIn')}
            onClick={() =>
              setState((s) =>
                editorReducer(
                  s,
                  {
                    type: 'setViewport',
                    viewport: { ...s.viewport, zoom: Math.min(2, s.viewport.zoom * 1.2) },
                  },
                  registry,
                ),
              )
            }
          >
            +
          </button>
          <button
            className="tool-btn tool-wide"
            title={t('ui.fit')}
            onClick={() => canvas.current?.fit()}
          >
            {t('ui.fit')}
          </button>
        </div>
        {inside.length > 0 && (
          <div className="canvas-crumb">
            <button className="btn" onClick={leaveInside}>
              ← {t('ui.back')}
            </button>
            <span>
              {inside.map((v) => t(`components.${v.name}`)).join(' / ')} · {t('ui.readonly')}
            </span>
          </div>
        )}
        {untouched && !inside.length && <div className="canvas-hint">{t('ui.emptyCanvas')}</div>}
      </section>

      <aside className="panel panel-right">
        {(error || state.error) && (
          <p role="alert" className="status status-fail">
            {translateDiagnostic(locale, error ?? { key: state.error! })}
          </p>
        )}
        {errors
          .filter((e) => !e.node && !e.soft)
          .map((e, i) => (
            <p className="status status-fail" key={i}>
              {e.message}
            </p>
          ))}
        {selected && canEdit && (
          <Properties
            key={selected.id}
            node={selected}
            t={t}
            onChange={(attrs) =>
              command({ type: 'setAttrs', id: selected.id, attrs, replace: true })
            }
          />
        )}
        {probing && (
          <section className="section">
            <h2 className="label">{t('ui.liveValues')}</h2>
            {testCases.length > 0 && (
              <div className="case-switch">
                <span className="label-inline">{t('ui.inputCase')}</span>
                {hasTensorInputs && (
                  <button
                    className="btn case-step"
                    aria-label={t('ui.previousCase')}
                    title={t('ui.previousCase')}
                    disabled={running || busy || Number(inputCase) <= 0}
                    onClick={() => selectInputCase(String(Number(inputCase) - 1))}
                  >
                    ‹
                  </button>
                )}
                <select
                  className="field"
                  aria-label={t('ui.inputCase')}
                  value={inputCase}
                  disabled={running || busy}
                  onChange={(e) => selectInputCase(e.target.value)}
                >
                  {!hasTensorInputs && <option value="">{t('ui.customInputs')}</option>}
                  {testCases.map((c, i) => (
                    <option key={c.id} value={i}>
                      {t('ui.inputCaseNumber', { n: i + 1 })}
                    </option>
                  ))}
                </select>
                {hasTensorInputs && (
                  <button
                    className="btn case-step"
                    aria-label={t('ui.nextCase')}
                    title={t('ui.nextCase')}
                    disabled={running || busy || Number(inputCase) >= testCases.length - 1}
                    onClick={() => selectInputCase(String(Number(inputCase) + 1))}
                  >
                    ›
                  </button>
                )}
              </div>
            )}
            {Object.entries(inputs).map(([name, value]) => (
              <div
                className={`probe-field${level.interface.inputs[name].length > 0 ? ' is-tensor' : ''}`}
                key={name}
              >
                <label htmlFor={`probe-${name}`} className="probe-name">
                  {name} =
                </label>
                {level.interface.inputs[name].length > 0 ? (
                  <TensorGrid
                    label={name}
                    shape={level.interface.inputs[name]}
                    values={value.split(',').map(Number)}
                    format={(n) => compact(locale, n)}
                  />
                ) : (
                  <input
                    id={`probe-${name}`}
                    className="field"
                    value={value}
                    aria-invalid={'invalid' in parsedInputs && parsedInputs.invalid === name}
                    spellCheck={false}
                    disabled={!!verify}
                    onChange={(e) => {
                      setInputCase('');
                      setInputs((s) => ({ ...s, [name]: e.target.value }));
                    }}
                  />
                )}
                {level.interface.inputs[name].length > 0 && (
                  <span className="probe-shape">{shapeText(level.interface.inputs[name], t)}</span>
                )}
                {level.interface.inputs[name].length === 0 && (
                  <input
                    type="range"
                    className="probe-slider"
                    aria-label={name}
                    min={-5}
                    max={5}
                    step={0.05}
                    disabled={!!verify}
                    value={Number(value) || 0}
                    onChange={(e) => {
                      setInputCase('');
                      setInputs((s) => ({ ...s, [name]: e.target.value }));
                    }}
                  />
                )}
              </div>
            ))}
            <div className={`probe-out${outputValue?.shape?.length ? ' is-tensor' : ''}`}>
              <span className="probe-name">{level.interface.output} =</span>
              {outputValue?.shape?.length ? (
                <TensorGrid
                  label={level.interface.output}
                  shape={outputValue.shape}
                  values={outputActual!.values}
                  off={
                    expected && outputActual ? offIndices(level, outputActual, expected) : undefined
                  }
                  format={(n) => compact(locale, n)}
                />
              ) : (
                <span className="mono">
                  {outputValue?.title.replace(/^\[[^\]]*\]\s*/, '') ?? '—'}
                </span>
              )}
            </div>
            {expected && (
              <div
                className={`probe-out probe-expected${expected.shape.length ? ' is-tensor' : ''}`}
              >
                <span className="probe-name">
                  {t('ui.expected')} {level.interface.output} =
                </span>
                {expected.shape.length ? (
                  <TensorGrid
                    label={t('ui.expected')}
                    shape={expected.shape}
                    values={expected.values}
                    format={(n) => compact(locale, n)}
                  />
                ) : (
                  <span className="mono" title={fullValues(expected.values)}>
                    {compact(locale, expected.values[0])}
                  </span>
                )}
                {outputActual && (
                  <span
                    className={`probe-verdict ${matchesExpected(level, outputActual, expected) ? 'ok' : 'off'}`}
                  >
                    {matchesExpected(level, outputActual, expected) ? '✓' : '≠'}
                  </span>
                )}
              </div>
            )}
            <p className="hint">{t(hasTensorInputs ? 'ui.tensorProbeHelp' : 'ui.probeHelp')}</p>
          </section>
        )}
        <section className="section">
          <div className="section-head">
            <h2 className="label">{t(level.kind === 'train' ? 'ui.training' : 'ui.tests')}</h2>
            {level.kind !== 'train' && (
              <span className="label">
                {t('ui.partCount', {
                  count: state.graph.nodes.filter((n) => !['input', 'output'].includes(n.component))
                    .length,
                })}
              </span>
            )}
          </div>
          {level.kind === 'train' && (
            <>
              <LossChart
                losses={losses}
                threshold={threshold}
                label={t('ui.loss')}
                thresholdLabel={t('ui.threshold')}
              />
              <dl className="kv train-stats">
                <dt>{t('ui.steps')}</dt>
                <dd>{report ? formatNumber(locale, report.step) : '—'}</dd>
                <dt>{t('ui.loss')}</dt>
                <dd>{report ? compact(locale, report.loss) : '—'}</dd>
                <dt>{t('ui.accuracy')}</dt>
                <dd>
                  {report?.accuracy !== undefined
                    ? formatNumber(locale, report.accuracy, {
                        style: 'percent',
                        maximumFractionDigits: 1,
                      })
                    : '—'}
                </dd>
                <dt>{t('ui.backend')}</dt>
                <dd>{t(`backend.${backend}`)}</dd>
              </dl>
            </>
          )}
          <div className="btn-row">
            {verify ? (
              <button className="btn btn-primary" onClick={() => (skip.current = true)}>
                ⏭ {t('ui.skip')}
              </button>
            ) : (
              <button
                className="btn btn-primary"
                disabled={busy || running || !!inside.length}
                onClick={() => void run()}
              >
                ▶ {t(level.kind === 'train' ? 'ui.start' : 'ui.runTests')}
              </button>
            )}
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
                  className="btn btn-danger"
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
          {shown && shownTests && (
            <div className="results">
              {verify ? (
                <p className="status">
                  {t('ui.verifying', { n: verify.index + 1, count: verify.result.tests.length })}
                </p>
              ) : (
                <p className={`status ${shown.passed ? 'status-pass' : 'status-fail'}`}>
                  {shown.passed ? '✓ ' : '✗ '}
                  {t(shown.passed ? 'ui.passedCount' : 'ui.failedCount', {
                    passed: shown.tests.filter((r) => r.passed).length,
                    count: shown.tests.length,
                  })}
                </p>
              )}
              {level.kind !== 'train' && (
                <div className="table-scroll">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>{t('ui.expected')}</th>
                        <th>{t('ui.actual')}</th>
                        <th />
                      </tr>
                    </thead>
                    <tbody>
                      {shownTests.map((r, i) => (
                        <tr
                          key={r.id}
                          className={verify && i === verify.index ? 'is-current' : undefined}
                        >
                          <td>{i + 1}</td>
                          <td title={fullValues(r.expected)}>{listText(locale, r.expected, 2)}</td>
                          <td title={fullValues(r.actual)}>
                            {r.errorKey
                              ? translateDiagnostic(locale, {
                                  key: r.errorKey,
                                  params: r.errorParams,
                                })
                              : listText(locale, r.actual, 2)}
                          </td>
                          <td className={r.passed ? 'ok' : 'err'}>
                            {verify && i === verify.index && r.passed ? '·' : r.passed ? '✓' : '✗'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </section>
      </aside>

      {ghost && (
        <div className="palette-ghost" style={{ left: ghost.x, top: ghost.y }}>
          <span className="sym">{SYMBOLS[ghost.component] ?? ''}</span>
          {t(`components.${ghost.component}`)}
        </div>
      )}

      {modal && result && (
        <WinModal
          t={t}
          locale={locale}
          code={entry.code}
          title={t(level.titleKey)}
          unlocked={level.unlockComponents ?? []}
          par={level.par}
          features={level.unlockFeatures ?? []}
          metrics={result.metrics}
          previous={previousBest}
          onClose={() => setModal(false)}
          onExit={onExit}
        />
      )}
    </main>
  );
}

function PaletteItem({
  name,
  t,
  player,
  disabled,
  onPress,
}: {
  name: string;
  t: T;
  player?: boolean;
  disabled: boolean;
  onPress: (component: string, e: React.PointerEvent) => void;
}) {
  return (
    <button
      type="button"
      className={`palette-item ${player ? 'is-player' : ''}`}
      disabled={disabled}
      onPointerDown={(e) => onPress(name, e)}
    >
      <span className="sym">{SYMBOLS[name] ?? '?'}</span>
      <span className="name">{t(`components.${name}`)}</span>
    </button>
  );
}

function WinModal({
  t,
  locale,
  code,
  title,
  unlocked,
  par,
  features,
  metrics,
  previous,
  onClose,
  onExit,
}: {
  t: T;
  locale: Locale;
  code: string;
  title: string;
  unlocked: readonly string[];
  par?: number;
  features: readonly string[];
  metrics: LevelResult['metrics'];
  previous?: LevelResult['metrics'];
  onClose: () => void;
  onExit: () => void;
}) {
  const rows = ['topNodes', 'expandedPrimitives', 'parameterCount'] as const;
  return (
    <div className="modal-back" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <section className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <div className="level-id">
          LEVEL {code} · {t('ui.complete')}
        </div>
        <h2>{t('ui.levelComplete', { title })}</h2>
        {unlocked.length ? (
          unlocked.map((c) => (
            <div className="unlock" key={c}>
              <span className="unlock-sym">{SYMBOLS[c] ?? '?'}</span>
              <div>
                <b>{t('ui.addedToLibrary', { name: t(`components.${c}`) })}</b>
                <p className="hint">{t('ui.addedHint')}</p>
              </div>
            </div>
          ))
        ) : (
          <p className="muted">{t('ui.noUnlock')}</p>
        )}
        {par !== undefined && (
          <p className="hint">
            {t('ui.par', { n: par })} ·{' '}
            {t(metrics.expandedPrimitives <= par ? 'ui.parMet' : 'ui.parNotMet')}
          </p>
        )}
        {features.map((key) => (
          <p className="status status-pass" key={key}>
            {t(key)}
          </p>
        ))}
        <table className="table score-table">
          <thead>
            <tr>
              <th />
              <th>{t('ui.thisRun')}</th>
              <th>{t('ui.best')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((k) => {
              const better = previous && metrics[k] < previous[k];
              return (
                <tr key={k}>
                  <td className="score-name">{t(`metrics.${k}`)}</td>
                  <td className={better ? 'ok' : ''}>
                    {formatNumber(locale, metrics[k])}
                    {better ? ' ↓' : ''}
                  </td>
                  <td className="muted">{previous ? formatNumber(locale, previous[k]) : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="hint">{t('ui.scoreHint')}</p>
        <div className="btn-row modal-actions">
          <button className="btn" onClick={onClose}>
            {t('ui.keepImproving')}
          </button>
          <button className="btn btn-primary" onClick={onExit} autoFocus>
            {t('ui.backToLevels')} →
          </button>
        </div>
      </section>
    </div>
  );
}
