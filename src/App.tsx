import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  computationGraph,
  expandComponentSubgraph,
  type GraphJSON,
  type GraphNode,
} from './engine/graph';
import {
  applyResult,
  acknowledgeObservation,
  type CircuitLevel,
  levelOutputShapes,
  type Level,
  type LevelResult,
  type Progress,
} from './engine/levels';
import { componentOperators, parseExpression } from './engine/expression';
import { levelTracker } from './analytics';
import { buildPlayerRegistry, availableComponents } from './levels/registry';
import { CHAPTERS, catalog, entryFor, gameLevels, resumeEntry } from './levels/catalog';
import {
  connectionError,
  createEditorState,
  editorReducer,
  type EditorCommand,
} from './editor/state';
import {
  createTranslator,
  translate,
  translateDiagnostic,
  translateValueName,
  formatNumber,
  locales,
  localeNames,
  type Locale,
} from './i18n';
import { WorkerClient, WorkerClientError } from './worker/client';
import type { WorkerError } from './worker/protocol';
import type { TrainingReport, TrainingStepTrace } from './engine/training';
import Canvas, {
  nonfiniteKind,
  position,
  shapeText,
  signal,
  type CanvasHandle,
  type NodeValue,
} from './ui/Canvas';
import { Random } from './engine/random';
import Properties, { defaults } from './ui/Properties';
import LossChart from './ui/LossChart';
import DataPlot, { type PlotData } from './ui/DataPlot';
import SequenceTry from './ui/SequenceTry';
import { sequenceProbe } from './levels/sequence-probe';
import DatasetCard from './ui/DatasetCard';
import { VOCAB } from './levels/corpus';
import { caseInputs, fullValues } from './ui/tensor-values';
import { diagnosticNode, graphDiagnostic } from './ui/diagnostic';
import { formatValue } from './ui/format-value';
import TensorGrid from './ui/TensorGrid';
import { FixedWindowLimitLesson } from './ui/FixedWindowLimitLesson';
import { PlayerRnnComparison } from './ui/PlayerRnnComparison';
import GradientView from './ui/GradientView';
import type { BackwardGraph, BackwardTrace } from './engine/backward-graph';
import {
  expectedOutputs,
  matchesExpected,
  offIndices,
  type ExpectedValue,
} from './ui/expected-output';
import FunctionPlot from './ui/FunctionPlot';
import GradientSurfacePlot from './ui/GradientSurfacePlot';
import LossLandscapePlot from './ui/LossLandscapePlot';
import SelectedNodeIO, { selectedNodePorts, type SelectedPortProbe } from './ui/SelectedNodeIO';
import { gradientSurfaceInputs, plottedInput } from './ui/function-plot';
import {
  NODE_W,
  SYMBOLS,
  categoryOf,
  freeSpot,
  layoutGraph,
  nodeHeight,
} from './ui/graph-geometry';
import { TrainingLoop } from './ui/TrainingLoop';
import { ComponentPalette } from './ui/ComponentPalette';
import TensorTransformView, { tensorTransformKind } from './ui/TensorTransformView';
import { Coach } from './ui/Coach';
import {
  STORAGE_VERSION,
  createInitialProgress,
  isAutodiffUnlocked,
  isExpressionUnlocked,
  loadProgressWithMigration,
  loadLanguage,
  saveSave,
  createThrottledSave,
} from './storage';
import {
  dismissBackendFallbackDialog,
  isBackendFallbackDialogDismissed,
} from './backend-fallback-dialog';
import './ui/theme.css';
import { MusicToggle } from './ui/MusicToggle';
import { SfxToggle } from './ui/SfxToggle';
import { sfx, type SfxName } from './audio/sfx';
import { MathText } from './ui/MathText';
import { TemporalBenchmarkEvidence } from './ui/TemporalBenchmarkEvidence';
import { LevelFigure } from './ui/LevelFigure';
import { levelPapers } from './levels/papers';
import { levelLearningResources } from './levels/learning-resources';
import { testCaseLocation } from './ui/test-case-location';
import {
  componentSeenKey,
  initializeSeenComponents,
  isNewComponent,
  newlyIntroducedComponents,
} from './ui/new-components';
import './ui/layout.css';

type T = ReturnType<typeof createTranslator>;

export default function App() {
  const [save, setSave] = useState(() => ({
    progress: createInitialProgress(),
    settings: { language: loadLanguage() },
  }));
  const storageReady = useRef(false);
  const [saveFailed, setSaveFailed] = useState(false);
  const saver = useMemo(
    () =>
      createThrottledSave((value: { progress: Progress; settings: { language: Locale } }) => {
        if (!storageReady.current) return false;
        const saved = saveSave({ version: STORAGE_VERSION, ...value });
        setSaveFailed(!saved);
        return saved;
      }, 400),
    [],
  );
  const [loaded, setLoaded] = useState(false);
  const [level, setLevel] = useState<Level>();
  const [showPlayerComparison, setShowPlayerComparison] = useState(false);
  const openLevel = (next?: Level) => {
    setShowPlayerComparison(false);
    setLevel(next);
  };
  const [backend, setBackend] = useState<'pending' | 'cpu' | 'webgpu'>('pending');
  const [failure, setFailure] = useState<WorkerError>();
  const [fallback, setFallback] = useState(false);
  const [fallbackDialog, setFallbackDialog] = useState(false);
  const [revoked, setRevoked] = useState<string[]>([]);
  const client = useRef<WorkerClient>(null);
  const revision = useRef(0);
  const t = createTranslator(save.settings.language);
  useEffect(() => {
    const c = new WorkerClient();
    client.current = c;
    let active = true;
    storageReady.current = false;
    setLoaded(false);
    saver.cancel();
    loadProgressWithMigration(gameLevels, undefined, setRevoked, () => active)
      .catch(() => {
        if (active) {
          saver.cancel();
          setFailure({ key: 'storage.progress_load_failed', params: {} });
        }
        return undefined;
      })
      .then(async (progress) => {
        if (!active || !progress) return undefined;
        setSave((s) => ({ ...s, progress }));
        storageReady.current = true;
        setLoaded(true);
        return c.init();
      })
      .then((r) => {
        if (!active || !r) return;
        setBackend(r.backend);
        const fellBack = !!r.fallback;
        setFallback(fellBack);
        if (fellBack && !isBackendFallbackDialogDismissed()) setFallbackDialog(true);
      })
      .catch((e) => setFailure(e.detail ?? { key: 'worker.transport_error', params: {} }));
    return () => {
      active = false;
      saver.flush();
      storageReady.current = false;
      saver.cancel();
      c.dispose();
    };
  }, [saver]);
  useEffect(() => {
    document.documentElement.lang = save.settings.language;
    document.title = level ? `${t(level.titleKey)} · ${t('app.title')}` : t('app.title');
    if (loaded) saver.schedule(save);
  }, [save, saver, level, loaded]);
  useEffect(() => {
    const flush = () => saver.flush();
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (saver.flush()) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', beforeUnload);
    window.addEventListener('pagehide', flush);
    return () => {
      flush();
      window.removeEventListener('beforeunload', beforeUnload);
      window.removeEventListener('pagehide', flush);
    };
  }, [saver]);
  const [phone, setPhone] = useState(isPhone);
  const progress = (p: Progress) => setSave((s) => ({ ...s, progress: p }));
  const entry = level && entryFor(level);
  if (phone)
    return (
      <main className="phone-gate blueprint-bg">
        <div className="phone-gate-card">
          <span className="brand-mark">∂</span>
          <h1>{t('phone.title')}</h1>
          <p>{t('phone.body')}</p>
          <p className="phone-gate-url">{window.location.host}</p>
          <button onClick={() => setPhone(false)}>{t('phone.anyway')}</button>
        </div>
      </main>
    );
  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => openLevel()}>
          <span className="brand-mark">∂</span>
          {t('app.title')}
        </button>
        <span className="tag demo-tag" title={t('home.demoNote')}>
          DEMO
        </span>
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
        <SfxToggle t={t} />
        <MusicToggle t={t} chapter={entry?.chapter ?? 0} />
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
      {(failure || saveFailed) && (
        <div className="status status-fail banner" role="alert">
          {failure && <p>{translateDiagnostic(save.settings.language, failure)}</p>}
          {saveFailed && (
            <p>
              {t('storage.progress_save_failed')}{' '}
              <button type="button" className="btn" onClick={() => saver.flush()}>
                {t('storage.retry_save')}
              </button>
            </p>
          )}
        </div>
      )}
      {fallbackDialog && (
        <BackendFallbackDialog
          t={t}
          onContinue={() => setFallbackDialog(false)}
          onDontShowAgain={() => {
            dismissBackendFallbackDialog();
            setFallbackDialog(false);
          }}
        />
      )}
      {!loaded ? (
        <main className="home blueprint-bg">
          <div className="home-inner" role="status">
            {t(
              failure?.key === 'storage.progress_load_failed'
                ? 'storage.progress_load_failed'
                : 'storage.progress_loading',
            )}
          </div>
        </main>
      ) : showPlayerComparison &&
        level?.id === 'time-rnn' &&
        (backend === 'cpu' || backend === 'webgpu') ? (
        <PlayerRnnComparison
          locale={save.settings.language}
          code={entry!.code}
          t={t}
          progress={save.progress}
          backend={backend}
          onBack={() => setShowPlayerComparison(false)}
          onContinue={() => openLevel(nextOf(level))}
        />
      ) : level?.kind === 'observation' ? (
        <FixedWindowLimitLesson
          code={entry!.code}
          t={t}
          backend={backend === 'cpu' || backend === 'webgpu' ? backend : undefined}
          onContinue={() => {
            sfx.play('pass');
            setSave((current) => ({
              ...current,
              progress: acknowledgeObservation(current.progress, level),
            }));
            const next = nextOf(level);
            if (next) openLevel(next);
          }}
          progress={save.progress}
        />
      ) : level ? (
        <Editor
          key={level.id}
          level={level}
          progress={save.progress}
          onProgress={progress}
          onExit={() => openLevel()}
          onNext={
            nextOf(level)
              ? () => {
                  if (level.id === 'time-rnn') setShowPlayerComparison(true);
                  else openLevel(nextOf(level));
                }
              : undefined
          }
          locale={save.settings.language}
          backend={backend}
          client={client}
          revision={revision}
        />
      ) : (
        <Home
          t={t}
          progress={save.progress}
          ready={backend !== 'pending'}
          onOpen={openLevel}
          revoked={revoked}
          onDismissRevoked={() => setRevoked([])}
        />
      )}
    </div>
  );
}

/** Phones get a notice instead of the editor: wiring needs a pointer and room for three panels. */
function isPhone() {
  if (typeof navigator === 'undefined') return false;
  if (/iPhone|iPod|Android.*Mobile|Windows Phone|Mobile.*Firefox/i.test(navigator.userAgent))
    return true;
  return (
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(pointer: coarse) and (max-width: 760px)').matches
  );
}

/** Tools earned mid-game wear a NEW badge until first used (a per-browser nicety). */
const SEEN_FEATURES_KEY = 'differentiable.seenFeatures';
function readSeenFeatures(): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(SEEN_FEATURES_KEY) ?? '[]');
    return Array.isArray(value) ? value.filter((v) => typeof v === 'string') : [];
  } catch {
    return [];
  }
}
const FEATURE_SYMBOLS: Record<string, string> = {
  'levels.features.autodiff': '∂',
  'levels.features.expression': 'f(x)',
};

function Home({
  t,
  progress,
  ready,
  onOpen,
  revoked,
  onDismissRevoked,
}: {
  t: T;
  progress: Progress;
  ready: boolean;
  onOpen: (level: Level) => void;
  revoked: string[];
  onDismissRevoked: () => void;
}) {
  const fresh = progress.passed.length === 0;
  // Where the player left off: the first unlocked, unpassed level in road order.
  const next = resumeEntry(catalog, progress);
  const [chapter, setChapter] = useState<number>(next?.chapter ?? catalog.at(-1)!.chapter);
  const entries = catalog.filter((e) => e.chapter === chapter);
  return (
    <main className="home blueprint-bg">
      <div className="home-inner">
        <section className={`hero ${fresh ? '' : 'is-compact'}`}>
          <h1>
            {t('home.titleA')}
            <span className="hero-accent">a + b</span>
            {t('home.titleB')}
            {fresh && <br />}
            {t('home.titleC')}
          </h1>
          {fresh && <p className="muted">{t('home.lead')}</p>}
          {next ? (
            <button className="continue" disabled={!ready} onClick={() => onOpen(next.level)}>
              <span className="continue-label">{t(fresh ? 'home.start' : 'home.continue')}</span>
              <span className="continue-level">
                <span className="level-id">{next.code}</span> {t(next.level.titleKey)}
              </span>
              <span aria-hidden="true">→</span>
            </button>
          ) : (
            <span className="home-done-badge" role="status">
              {t('home.allDone')}
            </span>
          )}
        </section>
        {revoked.length > 0 && (
          <section className="revoked" role="status">
            <div>
              <b>{t('home.revokedTitle')}</b>
              <p>{t('home.revokedBody')}</p>
              <p className="revoked-levels">
                {catalog
                  .filter((e) => revoked.includes(e.level.id))
                  .map((e) => (
                    <button key={e.level.id} disabled={!ready} onClick={() => onOpen(e.level)}>
                      <span className="level-id">{e.code}</span> {t(e.level.titleKey)} →
                    </button>
                  ))}
              </p>
            </div>
            <button
              className="revoked-close"
              aria-label={t('home.revokedDismiss')}
              onClick={onDismissRevoked}
            >
              ×
            </button>
          </section>
        )}
        <ol className="road">
          {CHAPTERS.map((n) => {
            const levels = catalog.filter((e) => e.chapter === n);
            const passed = levels.filter((e) => progress.passed.includes(e.level.id)).length;
            const open = levels.some((e) => progress.unlocked.includes(e.level.id));
            return (
              <li key={n}>
                <button
                  className={`road-chapter ${open ? 'is-open' : ''} ${n === chapter ? 'is-selected' : ''} ${passed > 0 && passed === levels.length ? 'is-done' : ''}`}
                  aria-pressed={n === chapter}
                  onClick={() => setChapter(n)}
                >
                  <span
                    className="road-track"
                    style={
                      { '--done': levels.length ? passed / levels.length : 0 } as CSSProperties
                    }
                  />
                  <span className="road-n">CH {n}</span>
                  <span className="road-name">{t(`chapter.${n}`)}</span>
                  <span className="road-count">
                    {levels.length ? `${passed}/${levels.length}` : t('ui.comingSoon')}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
        <section className="chapter">
          <header className="chapter-head">
            <span className="road-n">CH {chapter}</span>
            <h2>{t(`chapter.${chapter}`)}</h2>
            <p className="muted">{t(`chapter.${chapter}.desc`)}</p>
          </header>
          {entries.length === 0 ? (
            <p className="road-soon">{t('ui.comingSoon')}</p>
          ) : (
            <div className="chapter-levels">
              {entries.map(({ level, code }) => {
                const passed = progress.passed.includes(level.id);
                const unlocked = progress.unlocked.includes(level.id);
                const best = progress.bestScores[level.id];
                return (
                  <button
                    key={level.id}
                    className={`level-card ${passed ? 'is-passed' : unlocked ? 'is-playable' : ''}`}
                    disabled={!ready || !unlocked}
                    onClick={() => onOpen(level)}
                  >
                    <span className="level-id">{code}</span>
                    <span className="level-card-title">{t(level.titleKey)}</span>
                    <span className="level-state">
                      {passed
                        ? best
                          ? t('ui.bestCompact', {
                              nodes: best.expandedPrimitives,
                              params: best.parameterCount,
                            })
                          : '✓'
                        : t(unlocked ? 'ui.playable' : 'ui.locked')}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>
        <footer className="home-foot">
          <p>
            <span className="tag demo-tag">DEMO</span> {t('home.demoNote')}
          </p>
          <p className="home-community">
            <a href="https://www.bilibili.com/video/BV13BJt6SEVx" target="_blank" rel="noreferrer">
              {t('home.promoVideo')}
            </a>
            <span>{t('home.qqGroup', { number: '1130870192' })}</span>
          </p>
          <p>{t('home.credit')}</p>
        </footer>
      </div>
    </main>
  );
}

/** The level a pass leads to: the first one it unlocks. */
const nextOf = (level: Level) =>
  level.unlock.map((id) => gameLevels.find((l) => l.id === id)).find(Boolean);

type EditorProps = {
  level: CircuitLevel;
  progress: Progress;
  onProgress: (p: Progress) => void;
  onExit: () => void;
  /** Opens the level this one unlocks; absent after the last level. */
  onNext?: () => void;
  locale: Locale;
  backend: string;
  client: React.RefObject<WorkerClient | null>;
  revision: React.RefObject<number>;
};

/** Time each test case stays on the canvas during step-through verification. */
const STEP_MS = 420;
/** Editor commands that make a sound when they change the graph. */
const COMMAND_SOUNDS: Partial<Record<EditorCommand['type'], SfxName>> = {
  addNode: 'place',
  paste: 'place',
  insertExpression: 'place',
  connect: 'connect',
  rewire: 'connect',
  disconnect: 'disconnect',
  removeNodes: 'remove',
};

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
      .map((n) => formatValue(locale, n))
      .join(', ') + (values.length > max ? ', …' : '')
  );
}
function nodeValue(
  locale: Locale,
  result: { shape: readonly number[]; values: number[] },
): NodeValue {
  return {
    text:
      result.values.length === 1
        ? formatValue(locale, result.values[0])
        : listText(locale, result.values, 2),
    title: `${result.shape.length ? `[${result.shape.join('×')}]` : translate(locale, 'ui.scalar')} ${fullValues(result.values)}`,
    ok: result.values.every(Number.isFinite),
    nonfinite: nonfiniteKind(result.values),
    shape: result.shape,
    values: result.values,
    signal: signal(result.values),
  };
}

function Editor({
  level,
  progress,
  onProgress,
  onExit,
  onNext,
  locale,
  backend,
  client,
  revision,
}: EditorProps) {
  const entry = entryFor(level);
  const tracker = useRef<ReturnType<typeof levelTracker>>(null);
  useEffect(() => {
    const t = levelTracker(level.id, entry.chapter);
    tracker.current = t;
    return () => t.dispose();
  }, [level.id]);
  const registry = useMemo(
    () => buildPlayerRegistry(gameLevels, progress),
    [progress.passedSolutions, progress.unlockedComponents],
  );
  const available = availableComponents(level, progress);
  const [state, setState] = useState(() =>
    createEditorState(level, progress.savedGraphs[level.id], available),
  );
  const [diagnostics, setDiagnostics] = useState<WorkerError[]>([]);
  // The 0-1 walkthrough, until the level is passed or skipped. Storage may be unavailable.
  const [coachSkipped, setCoachSkipped] = useState(() => {
    try {
      return localStorage.getItem('differentiable.coach') === 'skipped';
    } catch {
      return false;
    }
  });
  // Errors the live probes hit on their own nodes. The whole-graph compile stops at its first
  // error (usually an unwired output), so a bad multiply upstream would otherwise stay hidden.
  const [probeErrors, setProbeErrors] = useState<WorkerError[]>([]);
  const [error, setError] = useState<WorkerError>();
  const [result, setResult] = useState<LevelResult>();
  const [running, setRunning] = useState(false);
  const [modal, setModal] = useState(false);
  const [gradView, setGradView] = useState<{
    graph: GraphJSON;
    backward: BackwardGraph;
    trace?: BackwardTrace;
    intro?: string;
    wrt: string[];
    output: string;
    playerBackwardStart?: string;
    playerBackwardLabelKey?: string;
    engineBackwardLabelKey?: string;
    parameterState?: 'initial' | 'current';
    attachedLoss?: boolean;
  }>();
  const [previousBest, setPreviousBest] = useState<LevelResult['metrics']>();
  const [training, setTraining] = useState<'idle' | 'running' | 'paused'>('idle');
  const [report, setReport] = useState<TrainingReport>();
  const [losses, setLosses] = useState<number[]>([]);
  // Single-step training, replayed on the canvas: forward, then backward, then the update.
  const [stepFlow, setStepFlow] = useState<{
    trace: TrainingStepTrace;
    phase: 'forward' | 'backward' | 'update';
  }>();
  const stepToken = useRef(0);
  const stepGraph = useRef<GraphJSON | undefined>(undefined);
  const tunable = level.kind === 'train' ? level.tunable?.learningRate : undefined;
  const learningRate = tunable
    ? (progress.trainingSettings?.[level.id]?.learningRate ?? tunable.default)
    : undefined;
  const optimizers = level.kind === 'train' ? level.tunable?.optimizer : undefined;
  const savedOptimizer = progress.trainingSettings?.[level.id]?.optimizer;
  const optimizer = optimizers
    ? savedOptimizer && optimizers.includes(savedOptimizer)
      ? savedOptimizer
      : optimizers[0]
    : undefined;
  // Language-model levels can write: a prompt, a temperature and what came out.
  const [gen, setGen] = useState<{
    prompt: string;
    temperature: number;
    text?: string;
    /** Leading characters of `text` that echo the prompt (always shown in the prompt colour). */
    kept?: number;
    /** How many characters of `text` are visible; grows for a typewriter reveal. */
    visible?: number;
    seed: number;
  }>({ prompt: 'the ', temperature: 1, seed: 1 });
  const [writing, setWriting] = useState(false);
  const typewriter = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  useEffect(
    () => () => {
      if (typewriter.current) clearInterval(typewriter.current);
    },
    [],
  );
  const write = async () => {
    const c = client.current;
    if (!c || level.kind !== 'train') return;
    if (typewriter.current) clearInterval(typewriter.current);
    setError(undefined);
    setWriting(true);
    try {
      const r = await c.generate(level.id, state.graph, gen.prompt, 200, gen.seed, gen.temperature);
      // A fresh seed each time: the same prompt can continue differently.
      // The reply starts with the prompt minus characters outside the vocabulary.
      const keptText = [...gen.prompt].filter((ch) => VOCAB.includes(ch)).join('');
      const kept = r.text.startsWith(keptText) ? keptText.length : 0;
      setGen((g) => ({ ...g, text: r.text, kept, visible: kept, seed: g.seed + 1 }));
      // Reveal the continuation one character at a time; a new click cancels via the ref above.
      if (kept < r.text.length) {
        typewriter.current = setInterval(() => {
          setGen((g) => {
            const next = (g.visible ?? 0) + 1;
            if (next >= (g.text?.length ?? 0)) {
              if (typewriter.current) clearInterval(typewriter.current);
              typewriter.current = undefined;
              return { ...g, visible: g.text?.length };
            }
            return { ...g, visible: next };
          });
        }, 40);
      }
    } catch (e) {
      fail(e);
    } finally {
      setWriting(false);
    }
  };
  // The training data, for the plot beside the loss chart.
  const [plotBase, setPlotBase] = useState<PlotData>();
  const [trainingPlot, setTrainingPlot] = useState<PlotData>();
  useEffect(() => {
    setPlotBase(undefined);
    if (level.kind === 'train' && level.sequenceProbe && level.interface.inputs.x?.length === 2) {
      const probe = sequenceProbe(level.sequenceProbe, level.interface.inputs.x[1]);
      setPlotBase({
        kind: 'regression',
        x: probe.curve.x,
        t: probe.curve.t,
        targetLine: probe.targetLine,
      });
      return;
    }
    // Text datasets are token ids, not coordinates: their card shows the text instead.
    const modality = level.kind === 'train' ? level.data?.dataset.modality : undefined;
    if (level.kind !== 'train' || !modality || modality === 'text') return;
    let cancelled = false;
    const batch = level.dataset(new Random(level.seed));
    const x = batch.inputs.x,
      t = batch.inputs.t ?? batch.targets;
    void (async () => {
      try {
        if (!x || !t) return;
        const xs = Array.from(await x.read()),
          ts = Array.from(await t.read());
        if (cancelled) return;
        if (modality === 'line') setPlotBase({ kind: 'regression', x: xs, t: ts });
        else {
          const classes = t.shape[1];
          setPlotBase({
            kind: 'classes',
            points: Array.from({ length: x.shape[0] }, (_, i) => [xs[2 * i], xs[2 * i + 1]]),
            label: Array.from({ length: t.shape[0] }, (_, i) =>
              argmax(ts.slice(i * classes, (i + 1) * classes)),
            ),
          });
        }
      } finally {
        Object.values(batch.inputs).forEach((v) => v.dispose());
        batch.targets?.dispose();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [level]);
  const [values, setValues] = useState<Record<string, NodeValue>>({});
  const [expression, setExpression] = useState('');
  const [seenFeatures, setSeenFeatures] = useState(() => {
    const stored = readSeenFeatures();
    const initialized = initializeSeenComponents(stored, [
      ...available,
      ...progress.unlockedComponents,
    ]);
    if (initialized.length !== stored.length)
      try {
        localStorage.setItem(SEEN_FEATURES_KEY, JSON.stringify(initialized));
      } catch {
        // Private mode: badges are kept for this tab only.
      }
    return initialized;
  });
  const markSeen = (...features: string[]) => {
    setSeenFeatures((current) => {
      const next = [...new Set([...current, ...features])];
      if (next.length === current.length) return current;
      try {
        localStorage.setItem(SEEN_FEATURES_KEY, JSON.stringify(next));
      } catch {
        // Private mode: the badge just comes back next visit.
      }
      return next;
    });
  };
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
  // Shape groups: the main shapes, then each shape variant the judge also tries.
  const shapeGroups = useMemo(
    () =>
      level.kind === 'train'
        ? []
        : [
            { inputs: level.interface.inputs, fixed: level.fixed, generated: level.generated },
            ...(level.shapeVariants ?? []),
          ].map((group) => {
            const generated = group.generated;
            const random = generated ? new Random(generated.seed) : undefined;
            return {
              inputs: group.inputs,
              cases: [
                ...group.fixed,
                ...(generated
                  ? Array.from({ length: generated.count }, (_, i) =>
                      generated.generate(random!, i),
                    )
                  : []),
              ],
            };
          }),
    [level],
  );
  const [shapeGroup, setShapeGroup] = useState(0);
  const inputShapes = shapeGroups[shapeGroup]?.inputs ?? level.interface.inputs;
  const testCases = shapeGroups[shapeGroup]?.cases ?? [];
  const [inputCase, setInputCase] = useState('0');
  const hasTensorInputs = Object.values(level.interface.inputs).some((shape) => shape.length > 0);
  const selectInputCase = (index: string) => {
    setInputCase(index);
    if (index !== '') setInputs(caseInputs(testCases[Number(index)].inputs));
  };
  const selectCase = (group: number, index: number) => {
    const c = shapeGroups[group]?.cases[index];
    if (!c) return;
    setShapeGroup(group);
    setInputCase(String(index));
    setInputs(caseInputs(c.inputs));
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
        batch.targets?.dispose();
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
    const sound = COMMAND_SOUNDS[cmd.type];
    if (sound) {
      // Preview the edit (the reducer is pure) so a refused wire buzzes instead of clicking.
      const next = editorReducer(state, cmd, registry);
      const removedNothing =
        cmd.type === 'removeNodes' && next.graph.nodes.length === state.graph.nodes.length;
      if (!next.error && !removedNothing) sfx.play(sound);
      else if (cmd.type === 'connect' || cmd.type === 'rewire') sfx.play('reject');
    }
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
    if (r.passed || r.tests.length) tracker.current?.attempt(r.passed);
    // Stopping a training run is not a failure, so only graded checks play the miss.
    if (r.passed) sfx.play('win');
    else if (level.kind !== 'train' && r.tests.length) sfx.play('fail');
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
        if (event.plot) setTrainingPlot(event.plot);
        setLosses((a) => [...a, event.report.loss]);
      } else finish(event.result);
    });
  }, [level, client]);
  const lastGraph = useRef(state.graph);
  const computationKey = JSON.stringify(computationGraph(state.graph));
  useEffect(() => {
    latest.current.onProgress({
      ...latest.current.progress,
      savedGraphs: { ...latest.current.progress.savedGraphs, [level.id]: state.graph },
    });
  }, [state.graph]);
  useEffect(() => {
    const c = client.current;
    if (lastGraph.current !== state.graph) {
      setResult(undefined);
      setModal(false);
      stepToken.current++;
      setStepFlow(undefined);
      setTrainingPlot(undefined);
    }
    lastGraph.current = state.graph;
    revision.current++;
    void c?.syncProgress(latest.current.progress, revision.current).catch(fail);
    setDiagnostics([]);
    setError(undefined);
    const timer = setTimeout(
      () =>
        c
          ?.compile(level.id, state.graph)
          .then((r) => setDiagnostics(r.diagnostics))
          .catch(fail),
      250,
    );
    return () => clearTimeout(timer);
  }, [computationKey, registry]);
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
      // Let the browser copy selected text (descriptions, hints); only copy nodes otherwise.
      if (mod && e.key.toLowerCase() === 'c' && window.getSelection()?.toString()) return;
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
          const size = (inputShapes[key] ?? []).reduce((x, y) => x * y, 1);
          if (a.length !== size || !a.every(Number.isFinite)) throw new Error(key);
          return [key, a];
        }),
      ) as Record<string, number[]>;
    } catch (e) {
      return { invalid: (e as Error).message };
    }
  }, [inputs, level, inputShapes]);

  // The answer for the same inputs, so the player can compare without running the tests.
  const [expected, setExpected] = useState<Record<string, ExpectedValue>>();
  useEffect(() => {
    setExpected(undefined);
    if (level.kind === 'train' || 'invalid' in parsedInputs) return;
    let cancelled = false;
    expectedOutputs(level, parsedInputs as Record<string, number[]>, inputShapes).then(
      (result) => !cancelled && setExpected(result),
      () => {},
    );
    return () => {
      cancelled = true;
    };
  }, [level, parsedInputs]);
  const primaryExpected = expected?.[level.interface.output];
  const outputAxes =
    primaryExpected && !('invalid' in parsedInputs)
      ? level.outputAxes?.(parsedInputs as Record<string, number[]>)
      : undefined;
  // Scalar lessons use a curve; two-input gradient lessons use a surface and vector field.
  const plotInput = level.kind === 'train' ? undefined : plottedInput(level);
  const surfaceInputs = level.kind === 'train' ? undefined : gradientSurfaceInputs(level);
  const transformKind = level.kind === 'train' ? undefined : tensorTransformKind(level.id);
  const transformInputName =
    transformKind === 'mse' || transformKind === 'cross_entropy' ? 'p' : 'x';
  const transformInput =
    transformKind && !('invalid' in parsedInputs) ? parsedInputs[transformInputName] : undefined;
  const transformTargets =
    (transformKind === 'mse' || transformKind === 'cross_entropy') && !('invalid' in parsedInputs)
      ? parsedInputs.t
      : undefined;
  const lossLandscape = level.kind !== 'train' && ['loss-intro', '3-1'].includes(level.id);
  const visualPlot =
    plotInput !== undefined ||
    surfaceInputs !== undefined ||
    transformKind !== undefined ||
    lossLandscape;

  // Live probe: every node shows its output for the probe inputs, refreshed as the graph changes.
  useEffect(() => {
    if (!probing || busy || 'invalid' in parsedInputs) {
      setValues({});
      setProbeErrors([]);
      return;
    }
    const probeRevision = revision.current;
    let cancelled = false;
    const timer = setTimeout(
      async () => {
        const c = client.current;
        if (!c) return;
        const next: Record<string, NodeValue> = {};
        const found: WorkerError[] = [];
        // Function plots evaluate their output first and publish it immediately; the remaining node
        // labels can catch up without holding the player tangent back while the graph is scrubbed.
        const nodes = plotInput
          ? [...state.graph.nodes].sort(
              (a, b) => Number(b.component === 'output') - Number(a.component === 'output'),
            )
          : state.graph.nodes;
        for (const node of nodes) {
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
              shapeGroup ? (inputShapes as Record<string, number[]>) : undefined,
            );
            if (cancelled || revision.current !== probeRevision) return;
            next[node.id] = nodeValue(locale, r);
            if (visualPlot && node.component === 'output')
              setValues((current) => ({ ...current, [node.id]: next[node.id] }));
          } catch (e) {
            if (cancelled || revision.current !== probeRevision) return;
            const d = e instanceof WorkerClientError ? e.detail : undefined;
            if (
              d?.node &&
              !['missing_port', 'missing_input'].includes(d.key) &&
              !found.some((f) => f.node === d.node && f.key === d.key)
            )
              found.push(d);
          }
        }
        setValues(next);
        setProbeErrors(found);
      },
      verify ? 10 : visualPlot ? 0 : 300,
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [computationKey, parsedInputs, diagnostics, busy, locale, visualPlot]);

  const PHASE_MS = 1100;
  const stepOnce = async () => {
    const c = client.current;
    if (!c) return;
    setError(undefined);
    setResult(undefined);
    const token = ++stepToken.current;
    setTrainingPlot(undefined);
    try {
      const trace = await c.trainStep(level.id, state.graph, learningRate, optimizer);
      if (token !== stepToken.current) return;
      // The worker restarts the session when the graph changed; so does the chart.
      const fresh = stepGraph.current !== state.graph || trace.step === 1;
      stepGraph.current = state.graph;
      setLosses((a) => (fresh ? [trace.loss] : [...a, trace.loss]));
      setReport({ step: trace.step, loss: trace.loss, accuracy: trace.accuracy });
      for (const phase of ['forward', 'backward', 'update'] as const) {
        if (token !== stepToken.current) return;
        setStepFlow({ trace, phase });
        if (phase !== 'update') await new Promise((done) => setTimeout(done, PHASE_MS));
      }
    } catch (e) {
      fail(e);
    }
  };
  const resetSteps = () => {
    setTrainingPlot(undefined);
    stepToken.current++;
    setStepFlow(undefined);
    setLosses([]);
    setReport(undefined);
    stepGraph.current = undefined;
    void client.current?.trainReset().catch(fail);
  };
  const saveTraining = (settings: { learningRate?: number; optimizer?: 'sgd' | 'adam' }) =>
    latest.current.onProgress({
      ...latest.current.progress,
      trainingSettings: {
        ...latest.current.progress.trainingSettings,
        [level.id]: {
          learningRate: learningRate ?? 0.01,
          ...(optimizer ? { optimizer } : {}),
          ...settings,
        },
      },
    });
  const setLearningRate = (lr: number) => saveTraining({ learningRate: lr });
  // A different optimizer starts a fresh session in the worker; start the chart over too.
  const setOptimizer = (kind: 'sgd' | 'adam') => {
    if (kind === optimizer) return;
    saveTraining({ optimizer: kind });
    resetSteps();
  };
  const traceText = (v: { values: number[]; size: number }) =>
    v.size === 1 ? formatValue(locale, v.values[0]) : listText(locale, v.values, 2);
  const flowValues = useMemo(() => {
    if (!stepFlow) return undefined;
    const next: Record<string, NodeValue> = {};
    const { forward } = stepFlow.trace;
    for (const node of state.graph.nodes) {
      const id =
        node.component === 'output'
          ? state.graph.edges.find((e) => e.to.node === node.id)?.from.node
          : node.id;
      const v = id ? (forward[node.id] ?? forward[id]) : undefined;
      if (!v) continue;
      next[node.id] = {
        text: traceText(v),
        title: `${v.shape.length ? `[${v.shape.join('×')}]` : t('ui.scalar')} ${fullValues(v.values)}${v.size > v.values.length ? ' …' : ''}`,
        ok: v.values.every(Number.isFinite),
        nonfinite: nonfiniteKind(v.values),
        shape: v.shape,
        values: v.values,
        signal: signal(v.values),
      };
    }
    return next;
  }, [stepFlow?.trace, state.graph, locale]);
  const plot = useMemo((): PlotData | undefined => {
    if (trainingPlot) return trainingPlot;
    if (!plotBase || !stepFlow) return plotBase;
    const { forward } = stepFlow.trace;
    if (plotBase.kind === 'regression') {
      // The prediction is whatever feeds the loss's p port.
      const edge = state.graph.edges.find(
        (e) =>
          e.to.port === 'p' &&
          state.graph.nodes.find((n) => n.id === e.to.node)?.component === 'mse',
      );
      const p = edge && forward[edge.from.node];
      return p && p.size === plotBase.x.length ? { ...plotBase, p: p.values } : plotBase;
    }
    const out = state.graph.nodes.find((n) => n.component === 'output');
    const source = out && state.graph.edges.find((e) => e.to.node === out.id)?.from.node;
    const y = (out && forward[out.id]) ?? (source ? forward[source] : undefined);
    const n = plotBase.points.length;
    if (!y || y.size !== y.values.length || y.shape.length !== 2 || y.shape[0] !== n)
      return plotBase;
    const c = y.shape[1];
    return {
      ...plotBase,
      predicted: Array.from({ length: n }, (_, i) => argmax(y.values.slice(i * c, (i + 1) * c))),
    };
  }, [plotBase, trainingPlot, stepFlow?.trace, state.graph]);
  const flow = stepFlow && {
    phase: stepFlow.phase,
    // Data and constants are not learned; their gradients would only be noise here.
    gradient: Object.fromEntries(
      Object.entries(stepFlow.trace.gradient)
        .filter(([id]) => {
          const c = state.graph.nodes.find((n) => n.id === id)?.component;
          return c !== 'input' && c !== 'const';
        })
        .map(([id, v]) => [id, traceText(v)]),
    ),
    update: Object.fromEntries(
      Object.entries(stepFlow.trace.parameters).map(([id, v]) => [
        id,
        v.before.length === 1
          ? `${formatValue(locale, v.before[0])} → ${formatValue(locale, v.after[0])}`
          : `${listText(locale, v.before, 2)} → ${listText(locale, v.after, 2)}`,
      ]),
    ),
  };

  const gradientWrt =
    level.kind === 'train'
      ? state.graph.nodes.filter((n) => n.component === 'param').map((n) => n.id)
      : level.kind === 'derivative'
        ? [level.derivativeInput ?? 'x']
        : Object.keys(level.interface.inputs);
  /** The engine's backward graph for `graph`, traced on the inputs in the live panel. */
  const openGradientView = async (
    graph: GraphJSON,
    intro?: string,
    reveal?: NonNullable<CircuitLevel['revealGraph']>,
  ) => {
    const c = client.current;
    if (!c) return;
    const inputs = reveal
      ? reveal.inputs
      : level.kind === 'train'
        ? undefined
        : 'invalid' in parsedInputs || shapeGroup
          ? level.fixed[0]?.inputs
          : (parsedInputs as Record<string, number[]>);
    const contract = reveal
      ? { inputShapes: reveal.inputShapes, output: reveal.output, wrt: [...reveal.wrt] }
      : undefined;
    try {
      const r = await c.backward(level.id, graph, inputs, contract);
      setGradView({
        graph: r.graph ?? graph,
        backward: r.backward,
        trace: r.trace,
        intro,
        wrt: reveal
          ? [...reveal.wrt]
          : level.kind === 'train'
            ? (r.wrt ?? graph.nodes.filter((n) => n.component === 'param').map((n) => n.id))
            : gradientWrt,
        output: reveal?.output ?? (level.kind === 'train' ? 'loss' : level.interface.output),
        playerBackwardStart: reveal?.playerBackwardStart,
        playerBackwardLabelKey: reveal?.playerBackwardLabelKey,
        engineBackwardLabelKey: reveal?.engineBackwardLabelKey,
        parameterState: r.parameterState,
        attachedLoss: r.attachedLoss,
      });
    } catch (e) {
      fail(e);
    }
  };

  const selected = state.graph.nodes.find((n) => n.id === state.selection[0]);
  const selectedPorts = useMemo(
    () => (selected ? selectedNodePorts(state.graph, selected, registry) : []),
    [selected, state.graph, registry],
  );
  const [selectedProbes, setSelectedProbes] = useState<Record<string, SelectedPortProbe>>({});
  useEffect(() => {
    const endpoints = selectedPorts.filter((port) => port.endpoint);
    if (!selected || !endpoints.length) {
      setSelectedProbes({});
      return;
    }
    if (busy || 'invalid' in parsedInputs) {
      setSelectedProbes(
        Object.fromEntries(endpoints.map((port) => [port.key, { status: 'unavailable' }])),
      );
      return;
    }
    const c = client.current;
    if (!c) return;
    let cancelled = false;
    setSelectedProbes(
      Object.fromEntries(endpoints.map((port) => [port.key, { status: 'loading' }])),
    );
    const byEndpoint = new Map<string, typeof endpoints>();
    for (const port of endpoints) {
      const key = JSON.stringify([port.endpoint!.node, port.endpoint!.port]);
      byEndpoint.set(key, [...(byEndpoint.get(key) ?? []), port]);
    }
    for (const ports of byEndpoint.values()) {
      const endpoint = ports[0].endpoint!;
      void c
        .probe(
          level.id,
          state.graph,
          endpoint,
          parsedInputs as Record<string, number[]>,
          shapeGroup ? (inputShapes as Record<string, number[]>) : undefined,
        )
        .then((result) => {
          if (cancelled) return;
          const value = nodeValue(locale, result);
          setSelectedProbes((current) => ({
            ...current,
            ...Object.fromEntries(
              ports.map((port) => [port.key, { status: 'value' as const, value }]),
            ),
          }));
        })
        .catch((caught) => {
          if (cancelled) return;
          const detail = caught instanceof WorkerClientError ? caught.detail : undefined;
          const probe: SelectedPortProbe = {
            status: 'error',
            message: detail
              ? graphDiagnostic(locale, state.graph, detail, registry, false)
              : t('ui.valueUnavailable'),
          };
          setSelectedProbes((current) => ({
            ...current,
            ...Object.fromEntries(ports.map((port) => [port.key, probe])),
          }));
        });
    }
    return () => {
      cancelled = true;
    };
  }, [selected, selectedPorts, computationKey, parsedInputs, shapeGroup, busy, locale]);
  const display = inside.at(-1)?.graph ?? state.graph;
  // "Not wired yet" is unfinished work, not a mistake: shown as a quiet note instead of in red.
  const SOFT = new Set(['missing_port', 'missing_input']);
  const errors = [
    ...(error?.node && !diagnostics.some((d) => d.node === error.node && d.key === error.key)
      ? [error]
      : []),
    ...diagnostics,
    ...probeErrors.filter((p) => !diagnostics.some((d) => d.node === p.node && d.key === p.key)),
  ].map((e) => ({
    ...e,
    soft: SOFT.has(e.key),
    node: diagnosticNode(state.graph, e),
    // Shown on the node itself, so no component-name prefix.
    message: graphDiagnostic(locale, state.graph, e, registry, false),
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
      const graph = expandComponentSubgraph(def, node.attrs ?? {})!;
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
        setTrainingPlot(undefined);
        setLosses([]);
        setReport(undefined);
        stepToken.current++;
        setStepFlow(undefined);
        setTraining('running');
        await client.current?.start(level.id, state.graph, learningRate, optimizer);
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
          const at = testCaseLocation(r.tests, i, shapeGroups);
          if (at) selectCase(at.group, at.index);
          // Stop on the first failing case and leave its values on the canvas to debug.
          if (!r.tests[i].passed) break;
          if (!skip.current) sfx.play('pass', i);
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

  const addPaletteComponent = (
    component: string,
    at: { x: number; y: number },
    findRoom: boolean,
  ) => {
    const attrs = defaults(
      component,
      registry.get(component),
      level.componentDefaults?.[component],
    );
    let position = { x: at.x - NODE_W / 2, y: at.y - 13 };
    if (findRoom) {
      // A drag drops exactly where the player let go; a click or keypress looks for room nearby.
      const placed = layoutGraph(state.graph, registry);
      const rects = state.graph.nodes.flatMap((n) => {
        const p = placed.get(n.id);
        return p ? [{ ...p, w: NODE_W, h: nodeHeight(n, registry) }] : [];
      });
      const h = nodeHeight({ id: '', component, attrs }, registry);
      position = freeSpot(position, NODE_W, h, rects);
    }
    command({ type: 'addNode', component, attrs, position });
    markSeen(componentSeenKey(component));
  };
  const activatePaletteComponent = (component: string) => {
    if (busy || inside.length) return;
    const rect = document.querySelector('.canvas')?.getBoundingClientRect();
    const at = rect
      ? canvas.current?.toGraph(rect.left + rect.width * 0.45, rect.top + rect.height * 0.45)
      : null;
    if (at) addPaletteComponent(component, at, true);
  };
  // Palette: press on an item and drag it onto the canvas; a plain click drops it near the middle.
  const startPaletteDrag = (component: string, e: React.PointerEvent) => {
    if (busy || inside.length || e.button !== 0) return;
    e.preventDefault();
    const pointerId = e.pointerId,
      sx = e.clientX,
      sy = e.clientY;
    let moved = false;
    setGhost({ component, x: sx, y: sy });
    const cleanup = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', cancel);
      window.removeEventListener('blur', cancel);
    };
    const cancel = () => {
      cleanup();
      setGhost(undefined);
    };
    const move = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      if (Math.hypot(ev.clientX - sx, ev.clientY - sy) > 3) moved = true;
      setGhost({ component, x: ev.clientX, y: ev.clientY });
    };
    const up = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      cancel();
      if (!moved) {
        activatePaletteComponent(component);
        return;
      }
      const at = canvas.current?.toGraph(ev.clientX, ev.clientY);
      if (at) addPaletteComponent(component, at, false);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', cancel);
    window.addEventListener('blur', cancel);
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
        // a - b / a / b become one sub / div node once the player has built them.
        operators: componentOperators(available),
        availableComponents: available,
      });
      const y = Math.max(...state.graph.nodes.map((n) => position(n).y), 0) + 160;
      const target =
        state.graph.nodes.find(
          (node) =>
            node.component === 'output' &&
            !state.graph.edges.some((edge) => edge.to.node === node.id),
        ) ?? state.graph.nodes.find((node) => node.component === 'output')!;
      command({
        type: 'insertExpression',
        expression: fragment,
        position: { x: 60, y },
        to: { node: target.id, port: 'x' },
      });
      markSeen(
        ...fragment.nodes
          .map((node) => node.component)
          .filter((component) => available.includes(component))
          .map(componentSeenKey),
      );
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
  const outputNodes = state.graph.nodes.filter((node) => node.component === 'output');
  const outputValues = Object.fromEntries(
    outputNodes.map((node) => [String(node.attrs?.name), values[node.id]]),
  );
  const outputActuals = Object.fromEntries(
    Object.entries(outputValues).flatMap(([name, value]) =>
      value?.shape && value.values
        ? [[name, { shape: [...value.shape], values: [...value.values] }]]
        : [],
    ),
  ) as Record<string, ExpectedValue>;
  const outputActual = outputActuals[level.interface.output];
  const changeScalarInputs = (changes: Readonly<Record<string, string>>) => {
    setInputCase('');
    setValues((current) => {
      const next = { ...current };
      outputNodes.forEach((node) => delete next[node.id]);
      return next;
    });
    setInputs((current) => ({ ...current, ...changes }));
  };
  const changeScalarInput = (name: string, value: string) => changeScalarInputs({ [name]: value });
  const canEdit = !busy && !inside.length;
  const coaching =
    level.id === '0-1' && !progress.passed.includes('0-1') && !coachSkipped && !inside.length;
  // The empty-canvas hint is for the very first look only: any edit (even wiring x straight to y)
  // or any history at all hides it for good.
  const untouched =
    !state.graph.edges.length &&
    !state.past.length &&
    state.graph.nodes.every((n) => ['input', 'output'].includes(n.component));
  const threshold = level.kind === 'train' ? level.threshold.loss : undefined;
  // The expression box is a tool you earn at the end of chapter 1; early levels are about wiring.
  const expressionUnlocked = isExpressionUnlocked(progress, gameLevels);
  // Not on levels up to and including the one that teaches backprop by hand: no spoilers.
  const autodiffLevel = gameLevels.findIndex((l) =>
    l.unlockFeatures?.includes('levels.features.autodiff'),
  );
  const gradientViewShown =
    !('derivativeInputs' in level && level.derivativeInputs?.length) &&
    isAutodiffUnlocked(progress, gameLevels) &&
    gameLevels.findIndex((l) => l.id === level.id) > autodiffLevel;
  const ioShapes = Object.fromEntries(
    state.graph.nodes.flatMap((n) =>
      n.component === 'input'
        ? [[n.id, inputShapes[String(n.attrs?.name)] ?? []]]
        : n.component === 'output'
          ? [
              [
                n.id,
                shapeGroup
                  ? (expected?.[String(n.attrs?.name)]?.shape ?? [])
                  : (levelOutputShapes(level)[String(n.attrs?.name)] ?? []),
              ],
            ]
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
          <div className="level-desc">
            <MathText text={t(level.descriptionKey)} />
          </div>
          <LevelFigure levelId={level.id} t={t} />
          <TemporalBenchmarkEvidence levelId={level.id} locale={locale} t={t} />
          {level.par !== undefined && <p className="hint">{t('ui.par', { n: level.par })}</p>}
          {lossLandscape && (
            <LossLandscapePlot
              key={level.id}
              t={t}
              levelId={level.id}
              inputs={'invalid' in parsedInputs ? {} : parsedInputs}
              actual={outputActual}
              onPick={(changes) =>
                changeScalarInputs(
                  Object.fromEntries(
                    Object.entries(changes).map(([name, value]) => [name, String(value)]),
                  ),
                )
              }
            />
          )}
          {plotInput && level.kind !== 'train' && (
            <FunctionPlot
              key={level.id}
              t={t}
              level={level}
              input={plotInput}
              x={'invalid' in parsedInputs ? undefined : parsedInputs[plotInput]?.[0]}
              inputs={'invalid' in parsedInputs ? {} : parsedInputs}
              actual={outputActual?.values[0]}
              onPick={(x) => changeScalarInput(plotInput, String(x))}
            />
          )}
          {surfaceInputs && level.kind !== 'train' && (
            <GradientSurfacePlot
              key={level.id}
              t={t}
              level={level}
              names={surfaceInputs}
              inputs={'invalid' in parsedInputs ? {} : parsedInputs}
              actual={outputActuals}
              onPick={(next) =>
                changeScalarInputs(
                  Object.fromEntries(
                    Object.entries(next).map(([name, value]) => [name, String(value)]),
                  ),
                )
              }
            />
          )}
          {transformKind && transformInput && (
            <TensorTransformView
              kind={transformKind}
              values={transformInput}
              targets={transformTargets}
              shape={inputShapes[transformInputName] ?? []}
              t={t}
              format={(value) => compact(locale, value)}
            />
          )}
          {level.hintsKeys?.slice(0, hintsShown).map((key, i) => (
            <div className="hint hint-open" key={key}>
              <span className="hint-n">{i + 1}</span>
              <div className="hint-text">
                <MathText text={t(key)} />
              </div>
            </div>
          ))}
          {hintsShown < (level.hintsKeys?.length ?? 0) && (
            <button className="hint-toggle" onClick={() => setHintsShown((n) => n + 1)}>
              {t('ui.showHint', { n: hintsShown + 1 })}
            </button>
          )}
          {(levelPapers[level.id] || levelLearningResources[level.id]) && (
            <div className="further-reading">
              {levelPapers[level.id] && (
                <div className="papers">
                  <h2 className="label">{t('ui.papers')}</h2>
                  <ul>
                    {levelPapers[level.id].map((paper) => (
                      <li key={paper.url}>
                        <a href={paper.url} target="_blank" rel="noreferrer">
                          {paper.title}
                        </a>
                        <span className="paper-meta">
                          {paper.authors} · {paper.year}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {levelLearningResources[level.id] && (
                <div className="learning-resources">
                  <h2 className="label">{t('ui.learnMore')}</h2>
                  <ul>
                    {levelLearningResources[level.id].map((resource) => (
                      <li key={resource.titleKey}>
                        <a href={resource.url[locale]} target="_blank" rel="noreferrer">
                          {t(resource.titleKey)}
                        </a>
                        <span className="paper-meta">{t(resource.descriptionKey)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
        <section className="section component-library">
          <h2 className="label">{t('ui.components')}</h2>
          <ComponentPalette
            entries={[
              ...primitives.map((name) => ({
                name,
                isNew: isNewComponent(seenFeatures, name),
              })),
              ...playerComponents.map((name) => ({
                name,
                player: true,
                isNew: isNewComponent(seenFeatures, name),
              })),
              ...(level.unlockComponents ?? []).map((name) => ({ name, locked: true })),
              ...presets.map((name) => ({
                name,
                isNew: isNewComponent(seenFeatures, name),
              })),
            ]}
            t={t}
            disabled={!canEdit}
            onPress={startPaletteDrag}
            onActivate={activatePaletteComponent}
          />
        </section>
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
              {!seenFeatures.includes('expression') && <span className="new-badge">NEW</span>}
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
                onFocus={() => markSeen('expression')}
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
          values={inside.length ? null : (flowValues ?? values)}
          flow={inside.length ? undefined : flow}
          ioShapes={inside.length ? undefined : ioShapes}
          onSelect={(ids, shift) => command({ type: 'select', ids, shift })}
          onMove={(positions) => command({ type: 'moveNodes', positions })}
          onDeleteEdge={(index) => command({ type: 'disconnect', edge: state.graph.edges[index] })}
          onDeleteNode={(id) => {
            command({ type: 'select', ids: [id], shift: false });
            command({ type: 'removeNodes', ids: [id] });
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
              sfx.play('reject');
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
          {gradientViewShown && (
            <>
              <span className="tool-sep" />
              <button
                className={`tool-btn tool-wide tool-grad ${seenFeatures.includes('autodiff') ? '' : 'is-new'}`}
                title={t(
                  level.kind === 'train' && level.chapter !== '3' && level.id !== 'xor'
                    ? 'worker.training_gradient_unavailable'
                    : 'gradView.openHint',
                )}
                disabled={
                  busy ||
                  !!inside.length ||
                  errors.some((e) => !e.soft) ||
                  (level.kind === 'train' && level.chapter !== '3' && level.id !== 'xor')
                }
                onClick={() => {
                  markSeen('autodiff');
                  openGradientView(state.graph);
                }}
              >
                ∂ {t('gradView.open')}
                {!seenFeatures.includes('autodiff') && <span className="new-badge">NEW</span>}
              </button>
            </>
          )}
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
        {coaching ? (
          <Coach
            t={t}
            graph={state.graph}
            onSkip={() => {
              setCoachSkipped(true);
              try {
                localStorage.setItem('differentiable.coach', 'skipped');
              } catch {
                // The walkthrough just comes back next visit.
              }
            }}
          />
        ) : (
          untouched && !inside.length && <div className="canvas-hint">{t('ui.emptyCanvas')}</div>
        )}
      </section>

      <aside className="panel panel-right">
        {(error || state.error) && (
          <p role="alert" className="status status-fail">
            {graphDiagnostic(locale, state.graph, error ?? { key: state.error! }, registry)}
          </p>
        )}
        {errors
          .filter((e) => !e.node && !e.soft)
          .map((e, i) => (
            <p className="status status-fail" key={i}>
              {e.message}
            </p>
          ))}
        {selected && !inside.length && (
          <SelectedNodeIO
            t={t}
            locale={locale}
            node={selected}
            ports={selectedPorts}
            probes={selectedProbes}
            errors={errors}
          />
        )}
        {selected && canEdit && (
          <Properties
            key={selected.id}
            node={selected}
            connectedInputs={state.graph.edges
              .filter((edge) => edge.to.node === selected.id)
              .map((edge) => edge.to.port)}
            definition={registry.get(selected.component)}
            t={t}
            onChange={(attrs) =>
              command({ type: 'setAttrs', id: selected.id, attrs, replace: true })
            }
          />
        )}
        {probing && (
          <section className="section">
            <h2 className="label">{t('ui.liveValues')}</h2>
            {shapeGroups.length > 1 && (
              <div className="case-switch">
                <span className="label-inline">{t('ui.shapeGroup')}</span>
                <div className="seg shape-switch" role="group" aria-label={t('ui.shapeGroup')}>
                  {shapeGroups.map((g, i) => (
                    <button
                      key={i}
                      className={`seg-btn${i === shapeGroup ? ' is-on' : ''}`}
                      aria-pressed={i === shapeGroup}
                      disabled={running || busy || !!verify}
                      title={Object.entries(g.inputs)
                        .map(([name, shape]) => `${name} ${shapeText(shape, t)}`)
                        .join(', ')}
                      onClick={() =>
                        // The all-zero case says little about shapes; start on a lively one.
                        selectCase(
                          i,
                          Math.max(
                            0,
                            g.cases.findIndex((c) =>
                              Object.values(c.inputs).some((v) => v.some((x) => x !== 0)),
                            ),
                          ),
                        )
                      }
                    >
                      {Object.values(g.inputs)
                        .map((shape) => (shape.length ? shape.join('×') : t('ui.scalar')))
                        .join(' · ')}
                    </button>
                  ))}
                </div>
              </div>
            )}
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
                    <option key={`${i}:${c.id}`} value={i}>
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
                className={`probe-field${inputShapes[name].length > 0 ? ' is-tensor' : ''}`}
                key={name}
              >
                <label htmlFor={`probe-${name}`} className="probe-name">
                  {name} =
                </label>
                {inputShapes[name].length > 0 ? (
                  <TensorGrid
                    label={name}
                    shape={inputShapes[name]}
                    values={value.split(',').map(Number)}
                    format={(n) => formatValue(locale, n)}
                  />
                ) : (
                  <input
                    id={`probe-${name}`}
                    className="field"
                    value={value}
                    aria-invalid={'invalid' in parsedInputs && parsedInputs.invalid === name}
                    spellCheck={false}
                    disabled={!!verify}
                    onChange={(e) => changeScalarInput(name, e.target.value)}
                  />
                )}
                {inputShapes[name].length > 0 && (
                  <span className="probe-shape">{shapeText(inputShapes[name], t)}</span>
                )}
                {inputShapes[name].length === 0 && (
                  <input
                    type="range"
                    className="probe-slider"
                    aria-label={name}
                    min={-5}
                    max={5}
                    step={0.05}
                    disabled={!!verify}
                    value={Number(value) || 0}
                    onChange={(e) => changeScalarInput(name, e.target.value)}
                  />
                )}
              </div>
            ))}
            {Object.keys(levelOutputShapes(level)).map((name) => {
              const outputValue = outputValues[name];
              const actual = outputActuals[name];
              const target = expected?.[name];
              const displayName = translateValueName(t, name);
              return (
                <div className="probe-output-group" key={name}>
                  <div className={`probe-out${outputValue?.shape?.length ? ' is-tensor' : ''}`}>
                    <span className="probe-name">{displayName} =</span>
                    {outputValue?.shape?.length ? (
                      <TensorGrid
                        label={displayName}
                        shape={outputValue.shape}
                        values={actual!.values}
                        rows={name === level.interface.output ? outputAxes?.rows : undefined}
                        cols={name === level.interface.output ? outputAxes?.cols : undefined}
                        off={target && actual ? offIndices(level, actual, target) : undefined}
                        format={(n) => formatValue(locale, n)}
                      />
                    ) : (
                      <span className="mono">{outputValue?.text ?? '—'}</span>
                    )}
                  </div>
                  {target && (
                    <div
                      className={`probe-out probe-expected${target.shape.length ? ' is-tensor' : ''}`}
                    >
                      <span className="probe-name">
                        {t('ui.expected')} {displayName} =
                      </span>
                      {target.shape.length ? (
                        <TensorGrid
                          label={t('ui.expected')}
                          shape={target.shape}
                          values={target.values}
                          rows={name === level.interface.output ? outputAxes?.rows : undefined}
                          cols={name === level.interface.output ? outputAxes?.cols : undefined}
                          format={(n) => formatValue(locale, n)}
                        />
                      ) : (
                        <span className="mono" title={fullValues(target.values)}>
                          {formatValue(locale, target.values[0])}
                        </span>
                      )}
                      {actual && (
                        <span
                          className={`probe-verdict ${matchesExpected(level, actual, target) ? 'ok' : 'off'}`}
                        >
                          {matchesExpected(level, actual, target) ? '✓' : '≠'}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
            {level.outputNoteKey && (
              <div className="probe-note">
                <MathText text={t(level.outputNoteKey)} />
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
              <TrainingLoop
                t={t}
                optimizer={optimizer ?? level.optimizer.kind}
                learningRate={compact(locale, learningRate ?? level.optimizer.learningRate)}
                steps={formatNumber(locale, level.steps)}
                resample={!!level.resample}
                running={training === 'running'}
              />
              {level.data ? (
                <DatasetCard t={t} data={level.data} inputs={level.interface.inputs} plot={plot} />
              ) : (
                plot && (
                  <DataPlot
                    data={plot}
                    label={t(
                      level.sequenceProbe ? `ui.probe.${level.sequenceProbe}.label` : 'ui.data',
                    )}
                    axisLabels={
                      level.sequenceProbe
                        ? {
                            x: t(`ui.probe.${level.sequenceProbe}.x`),
                            y: t(`ui.probe.${level.sequenceProbe}.y`),
                          }
                        : undefined
                    }
                  />
                )
              )}
              {level.sequenceProbe && (
                <>
                  <p className="hint">{t(`ui.probe.${level.sequenceProbe}.help`)}</p>
                  <SequenceTry
                    key={level.id}
                    t={t}
                    kind={level.sequenceProbe}
                    length={level.interface.inputs.x?.[1] ?? 1}
                    refreshKey={[
                      computationKey,
                      report ? Math.floor(report.step / 25) : -1,
                      training,
                    ]}
                    predict={(rows) => {
                      const c = client.current;
                      if (!c) return Promise.reject(new Error('no worker'));
                      return c.predict(level.id, state.graph, rows);
                    }}
                  />
                </>
              )}
              {optimizers && optimizers.length > 1 && (
                <div className="opt-switch" role="radiogroup" aria-label={t('ui.optimizer')}>
                  <span className="label">{t('ui.optimizer')}</span>
                  <div className="seg">
                    {optimizers.map((kind) => (
                      <button
                        key={kind}
                        role="radio"
                        aria-checked={kind === optimizer}
                        className={`seg-btn${kind === optimizer ? ' is-on' : ''}`}
                        disabled={busy}
                        onClick={() => setOptimizer(kind)}
                      >
                        {t(`ui.optimizer.${kind}`)}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {tunable && learningRate !== undefined && (
                <label className="lr-control">
                  <span className="lr-control-head">
                    <span className="label">{t('ui.learningRate')}</span>
                    <output>η = {compact(locale, learningRate)}</output>
                  </span>
                  <input
                    type="range"
                    min={Math.log10(tunable.min)}
                    max={Math.log10(tunable.max)}
                    step={0.01}
                    value={Math.log10(learningRate)}
                    disabled={busy}
                    onChange={(e) =>
                      setLearningRate(Number(Number(10 ** Number(e.target.value)).toPrecision(2)))
                    }
                  />
                  <span className="lr-scale">
                    <span>{compact(locale, tunable.min)}</span>
                    <span>{compact(locale, tunable.max)}</span>
                  </span>
                </label>
              )}
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
              {level.generates && (
                <div className="gen">
                  <div className="gen-row">
                    <label className="gen-prompt">
                      <span className="label">{t('ui.prompt')}</span>
                      <input
                        className="field"
                        value={gen.prompt}
                        maxLength={40}
                        spellCheck={false}
                        onChange={(e) => setGen((g) => ({ ...g, prompt: e.target.value }))}
                      />
                    </label>
                    <label className="gen-temp">
                      <span className="label">
                        {t('ui.temperature')} <output>{gen.temperature.toFixed(1)}</output>
                      </span>
                      <input
                        type="range"
                        min={0.2}
                        max={2}
                        step={0.1}
                        value={gen.temperature}
                        onChange={(e) =>
                          setGen((g) => ({ ...g, temperature: Number(e.target.value) }))
                        }
                      />
                    </label>
                  </div>
                  <button
                    className="btn"
                    disabled={busy || writing || !!inside.length}
                    onClick={() => void write()}
                  >
                    ✎ {t('ui.generate')}
                  </button>
                  {gen.text !== undefined && (
                    <pre className="gen-out" aria-live="polite">
                      {/* Prompt echo stays tinted; the rest types out character by character. */}
                      <span className="gen-out-prompt">
                        {gen.text.slice(0, Math.min(gen.kept ?? 0, gen.visible ?? 0))}
                      </span>
                      {gen.text.slice(gen.kept ?? 0, gen.visible ?? 0)}
                    </pre>
                  )}
                  <p className="hint">{t('ui.generateHelp')}</p>
                </div>
              )}
              {stepFlow && (
                <p className="step-phase">
                  <span className={`is-${stepFlow.phase === 'backward' ? 'backward' : 'forward'}`}>
                    {t(`ui.stepPhase.${stepFlow.phase}`, {
                      loss: compact(locale, stepFlow.trace.loss),
                    })}
                  </span>
                </p>
              )}
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
            {level.kind === 'train' && !busy && (
              <>
                <button
                  className="btn"
                  title={t('ui.stepHelp')}
                  disabled={running || !!inside.length}
                  onClick={() => void stepOnce()}
                >
                  ⏯ {t('ui.stepOnce')}
                </button>
                {losses.length > 0 && (
                  <button className="btn" onClick={resetSteps}>
                    ↺ {t('ui.resetParams')}
                  </button>
                )}
              </>
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
              ) : shown.tests.length === 0 ? null : (
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
                          // Fixed and generated cases can share an id; the row index keeps keys unique.
                          key={`${i}:${r.id}`}
                          className={
                            verify && i === verify.index
                              ? 'is-current'
                              : !verify && testCaseLocation(shown.tests, i, shapeGroups)
                                ? 'is-pickable'
                                : undefined
                          }
                          title={
                            !verify && testCaseLocation(shown.tests, i, shapeGroups)
                              ? t('ui.showCase')
                              : undefined
                          }
                          onClick={() => {
                            const at = !verify && testCaseLocation(shown.tests, i, shapeGroups);
                            if (at) selectCase(at.group, at.index);
                          }}
                        >
                          <td>{i + 1}</td>
                          <td title={fullValues(r.expected)}>{listText(locale, r.expected, 2)}</td>
                          <td
                            title={fullValues(r.actual)}
                            className={r.errorKey ? 'cell-error' : undefined}
                          >
                            {r.errorKey
                              ? graphDiagnostic(locale, state.graph, {
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
          introduced={newlyIntroducedComponents(gameLevels, level)}
          par={level.par}
          features={level.unlockFeatures ?? []}
          metrics={result.metrics}
          previous={previousBest}
          ioNodes={
            state.graph.nodes.filter((n) => ['input', 'output'].includes(n.component)).length
          }
          onClose={() => setModal(false)}
          onExit={onExit}
          onNext={onNext}
          nextLabel={level.id === 'time-rnn' ? t('levels.time-player-comparison.start') : undefined}
          onReveal={
            level.revealGraph
              ? () => {
                  setModal(false);
                  openGradientView(
                    level.revealGraph!.graph,
                    t(level.revealGraph!.introKey ?? 'gradView.revealIntro'),
                    level.revealGraph,
                  );
                }
              : undefined
          }
        />
      )}
      {gradView && (
        <GradientView
          t={t}
          graph={gradView.graph}
          backward={gradView.backward}
          trace={gradView.trace}
          wrt={gradView.wrt}
          output={gradView.output}
          playerBackwardStart={gradView.playerBackwardStart}
          playerBackwardLabelKey={gradView.playerBackwardLabelKey}
          engineBackwardLabelKey={gradView.engineBackwardLabelKey}
          parameterState={gradView.parameterState}
          attachedLoss={gradView.attachedLoss}
          composite={(component) => !registry.get(component)?.primitive}
          format={(n) => formatValue(locale, n)}
          intro={gradView.intro}
          onClose={() => setGradView(undefined)}
        />
      )}
    </main>
  );
}

function WinModal({
  t,
  locale,
  code,
  title,
  unlocked,
  introduced,
  par,
  features,
  metrics,
  previous,
  ioNodes,
  onClose,
  onExit,
  onNext,
  nextLabel,
  onReveal,
}: {
  t: T;
  locale: Locale;
  code: string;
  title: string;
  unlocked: readonly string[];
  introduced: readonly string[];
  par?: number;
  features: readonly string[];
  metrics: LevelResult['metrics'];
  previous?: LevelResult['metrics'];
  /** Input and output nodes come with the level, so the node count leaves them out. */
  ioNodes: number;
  onClose: () => void;
  onExit: () => void;
  onNext?: () => void;
  nextLabel?: string;
  onReveal?: () => void;
}) {
  const rows = ['topNodes', 'expandedPrimitives', 'parameterCount'] as const;
  const shown = (m: LevelResult['metrics'], k: (typeof rows)[number]) =>
    k === 'topNodes' ? Math.max(0, m[k] - ioNodes) : m[k];
  return (
    <div className="modal-back" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <section className="modal is-win" role="dialog" aria-modal="true" aria-label={title}>
        <div className="level-id">
          LEVEL {code} · {t('ui.complete')}
        </div>
        <h2>{t('ui.levelComplete', { title })}</h2>
        {features.map((key) => (
          <div className="unlock" key={key}>
            <span
              className={`unlock-sym ${(FEATURE_SYMBOLS[key] ?? '★').length > 3 ? 'is-long' : ''}`}
            >
              {FEATURE_SYMBOLS[key] ?? '★'}
            </span>
            <div>
              <b>{t(key)}</b>
              <p className="hint">{t(`${key}.body`)}</p>
            </div>
          </div>
        ))}
        {unlocked.map((c) => (
          <div className="unlock" data-category={categoryOf(c)} key={c}>
            <span className={`unlock-sym ${(SYMBOLS[c] ?? '?').length > 3 ? 'is-long' : ''}`}>
              {SYMBOLS[c] ?? '?'}
            </span>
            <div>
              <b>{t('ui.addedToLibrary', { name: t(`components.${c}`) })}</b>
              <p className="hint">{t('ui.addedHint')}</p>
            </div>
          </div>
        ))}
        {introduced.map((c) => (
          <div className="unlock" data-category={categoryOf(c)} key={c}>
            <span className={`unlock-sym ${(SYMBOLS[c] ?? '?').length > 3 ? 'is-long' : ''}`}>
              {SYMBOLS[c] ?? '?'}
            </span>
            <div>
              <b>{t('ui.unlockedForLibrary', { name: t(`components.${c}`) })}</b>
              <p className="hint">{t('ui.unlockedHint')}</p>
            </div>
          </div>
        ))}
        {!unlocked.length && !introduced.length && !features.length && (
          <p className="muted">{t('ui.noUnlock')}</p>
        )}
        {par !== undefined && (
          <p className="hint">
            {t('ui.par', { n: par })} ·{' '}
            {t(metrics.expandedPrimitives <= par ? 'ui.parMet' : 'ui.parNotMet')}
          </p>
        )}
        {onReveal && (
          <button className="btn btn-grad" onClick={onReveal}>
            ∂ {t('gradView.revealButton')} →
          </button>
        )}
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
                    {formatNumber(locale, shown(metrics, k))}
                    {better ? ' ↓' : ''}
                  </td>
                  <td className="muted">
                    {previous ? formatNumber(locale, shown(previous, k)) : '—'}
                  </td>
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
          <button
            className={`btn ${onNext ? '' : 'btn-primary'}`}
            onClick={onExit}
            autoFocus={!onNext}
          >
            {t('ui.backToLevels')}
          </button>
          {onNext && (
            <button className="btn btn-primary" onClick={onNext} autoFocus>
              {nextLabel ?? t('ui.nextLevel')} →
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function BackendFallbackDialog({
  t,
  onContinue,
  onDontShowAgain,
}: {
  t: T;
  onContinue: () => void;
  onDontShowAgain: () => void;
}) {
  return (
    <div className="modal-back" onClick={(e) => e.target === e.currentTarget && onContinue()}>
      <section
        className="modal backend-fallback-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={t('ui.backendFallbackDialog.title')}
      >
        <h2>{t('ui.backendFallbackDialog.title')}</h2>
        <p>{t('ui.backendFallbackDialog.body')}</p>
        <div className="btn-row modal-actions">
          <button className="btn btn-primary" onClick={onContinue} autoFocus>
            {t('ui.backendFallbackDialog.continue')}
          </button>
          <button className="btn" onClick={onDontShowAgain}>
            {t('ui.backendFallbackDialog.dontShowAgain')}
          </button>
        </div>
      </section>
    </div>
  );
}

function argmax(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
