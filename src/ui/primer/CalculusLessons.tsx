// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react';
import {
  LEARNING_RATE_RANGE,
  descend,
  reachedMinimum,
  learningRateStages,
  type LearningRateStage,
} from '../../levels/learning-rate';
import {
  SLOPE_PASS_SCORE,
  chainRate,
  curves,
  gearSpeedCorrect,
  gearSpeeds,
  gearTrains,
  nudgeActual,
  nudgeCorrect,
  nudgePassed,
  nudgeRounds,
  slopeRounds,
  slopeScore,
} from '../../levels/primer-calculus';
import { MathText } from '../MathText';
import {
  Feedback,
  InlineAnswer,
  NextButton,
  NumberAnswer,
  PlotGrid,
  PrimerPanel,
  PrimerShell,
  curvePath,
  dataX,
  dataY,
  formatNumber,
  niceStep,
  sx,
  sy,
  type PrimerLessonProps,
  type Verdict,
  type Viewport,
} from './PrimerParts';

const PLOT = 360;
const MAX_RULER_DEGREES = 80;
const toDegrees = (radians: number) => (radians * 180) / Math.PI;
const toRadians = (degrees: number) => (degrees * Math.PI) / 180;

/** A square window with equal x/y scale, so on-screen angles equal true slopes. */
function squareView(cx: number, cy: number, halfWidth: number): Viewport {
  return {
    x0: cx - halfWidth,
    x1: cx + halfWidth,
    y0: cy - halfWidth,
    y1: cy + halfWidth,
    width: PLOT,
    height: PLOT,
  };
}

/** A line through (x, y) with the given slope, long enough to cross the whole view. */
function lineThrough(v: Viewport, x: number, y: number, slope: number) {
  const reach = (v.x1 - v.x0) * 2;
  const dx = reach / Math.sqrt(1 + slope * slope);
  return {
    x1: sx(v, x - dx),
    y1: sy(v, y - slope * dx),
    x2: sx(v, x + dx),
    y2: sy(v, y + slope * dx),
  };
}

function PlotFrame({ id, label, children }: { id: string; label: string; children: ReactNode }) {
  return (
    <svg className="primer-plot" viewBox={`0 0 ${PLOT} ${PLOT}`} role="img" aria-label={label}>
      <defs>
        <clipPath id={id}>
          <rect width={PLOT} height={PLOT} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${id})`}>{children}</g>
    </svg>
  );
}

// ---------------------------------------------------------------- calc-slope

export function SlopeLesson(props: PrimerLessonProps) {
  const { t } = props;
  const [round, setRound] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [degrees, setDegrees] = useState(0);
  const [score, setScore] = useState<number>();
  const svg = useRef<SVGSVGElement>(null);
  const done = round >= slopeRounds.length;
  const task = slopeRounds[Math.min(round, slopeRounds.length - 1)];
  const curve = curves[task.curve];
  const y = curve.f(task.x);
  const baseHalf = (curve.domain[1] - curve.domain[0]) / 2;
  const view = squareView(task.x, y, baseHalf / zoom);
  const guess = Math.tan(toRadians(degrees));
  const passed = score !== undefined && score >= SLOPE_PASS_SCORE;

  const aimAt = (event: PointerEvent<SVGSVGElement>) => {
    if (score !== undefined || !svg.current) return;
    const box = svg.current.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * PLOT;
    const py = ((event.clientY - box.top) / box.height) * PLOT;
    let dx = dataX(view, px) - task.x;
    let dy = dataY(view, py) - y;
    if (Math.hypot(dx, dy) < (view.x1 - view.x0) / 40) return;
    if (dx < 0) [dx, dy] = [-dx, -dy];
    const angle = toDegrees(Math.atan2(dy, dx));
    setDegrees(Math.max(-MAX_RULER_DEGREES, Math.min(MAX_RULER_DEGREES, Math.round(angle))));
  };
  const next = () => {
    setRound(round + 1);
    setZoom(1);
    setDegrees(0);
    setScore(undefined);
  };

  return (
    <PrimerShell
      {...props}
      steps={slopeRounds.map((_, i) => t('primer.ui.round', { n: i + 1 }))}
      current={round}
      done={done}
    >
      <PrimerPanel
        title={t('primer.ui.calc-slope.task')}
        meta={t('primer.ui.calc-slope.point', { x: formatNumber(task.x, 2) })}
      >
        <div className="primer-split">
          <svg
            ref={svg}
            className="primer-plot is-aimable"
            viewBox={`0 0 ${PLOT} ${PLOT}`}
            role="img"
            aria-label={t('primer.ui.calc-slope.plot')}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              aimAt(event);
            }}
            onPointerMove={(event) => {
              if (event.buttons) aimAt(event);
            }}
          >
            <defs>
              <clipPath id="primer-slope-clip">
                <rect width={PLOT} height={PLOT} />
              </clipPath>
            </defs>
            <g clipPath="url(#primer-slope-clip)">
              <PlotGrid v={view} step={niceStep(view.x1 - view.x0)} />
              <path className="primer-curve" d={curvePath(view, curve.f)} />
              {score !== undefined && (
                <line
                  className="primer-tangent"
                  {...lineThrough(view, task.x, y, curve.df(task.x))}
                />
              )}
              <line className="primer-ruler" {...lineThrough(view, task.x, y, guess)} />
              <circle className="primer-point" cx={sx(view, task.x)} cy={sy(view, y)} r={5} />
            </g>
          </svg>
          <div className="primer-controls">
            <p className="primer-instruction">{t('primer.ui.calc-slope.instruction')}</p>
            <label className="primer-slider">
              <span className="label">
                {t('primer.ui.calc-slope.zoom')} <b>×{formatNumber(zoom, 1)}</b>
              </span>
              <input
                type="range"
                min={0}
                max={Math.log(60)}
                step={0.01}
                value={Math.log(zoom)}
                onChange={(event) => setZoom(Math.exp(Number(event.target.value)))}
              />
            </label>
            <label className="primer-slider">
              <span className="label">
                {t('primer.ui.calc-slope.ruler')} <b>{formatNumber(guess, 2)}</b>
              </span>
              <input
                type="range"
                min={-MAX_RULER_DEGREES}
                max={MAX_RULER_DEGREES}
                step={1}
                value={degrees}
                disabled={score !== undefined}
                onChange={(event) => setDegrees(Number(event.target.value))}
              />
            </label>
            {score === undefined ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setScore(slopeScore(guess, task))}
              >
                {t('primer.ui.check')}
              </button>
            ) : (
              <>
                <div className="primer-meter" aria-label={t('primer.ui.calc-slope.fit')}>
                  <span style={{ width: `${Math.round(score * 100)}%` }} />
                </div>
                <Feedback verdict={passed ? 'good' : 'bad'}>
                  {t(passed ? 'primer.ui.calc-slope.good' : 'primer.ui.calc-slope.bad', {
                    fit: Math.round(score * 100),
                    slope: formatNumber(curve.df(task.x), 2),
                  })}
                </Feedback>
                <NextButton onClick={passed ? next : () => setScore(undefined)}>
                  {t(passed ? 'primer.ui.nextRound' : 'primer.ui.retry')} →
                </NextButton>
              </>
            )}
          </div>
        </div>
      </PrimerPanel>
    </PrimerShell>
  );
}

// ---------------------------------------------------------------- calc-nudge

export function NudgeLesson(props: PrimerLessonProps) {
  const { t } = props;
  const [round, setRound] = useState(0);
  const [results, setResults] = useState<boolean[]>([]);
  const [guess, setGuess] = useState<number>();
  const finished = results.length === nudgeRounds.length;
  const done = round >= nudgeRounds.length;
  const task = nudgeRounds[Math.min(round, nudgeRounds.length - 1)];
  const curve = curves[task.curve];
  const y = curve.f(task.x);
  const slope = curve.df(task.x);
  const actual = nudgeActual(task);
  const answered = guess !== undefined;
  const correct = answered && nudgeCorrect(guess, task);

  // Before answering: the whole curve. After: a magnifier around [x, x + dx].
  const span = (curve.domain[1] - curve.domain[0]) / 2;
  const view = answered
    ? squareView(
        task.x + task.dx / 2,
        y + actual / 2,
        Math.max(Math.abs(task.dx), Math.abs(actual)) * 1.6,
      )
    : squareView((curve.domain[0] + curve.domain[1]) / 2, y, span);

  const submit = (value: number) => {
    setGuess(value);
    setResults([...results, nudgeCorrect(value, task)]);
  };
  const next = () => {
    setRound(round + 1);
    setGuess(undefined);
  };
  const restart = () => {
    setRound(0);
    setResults([]);
    setGuess(undefined);
  };

  return (
    <PrimerShell
      {...props}
      steps={nudgeRounds.map((_, i) => t('primer.ui.round', { n: i + 1 }))}
      current={round}
      done={done}
    >
      <PrimerPanel
        title={t('primer.ui.calc-nudge.task')}
        meta={t('primer.ui.calc-nudge.score', {
          correct: results.filter(Boolean).length,
          total: nudgeRounds.length,
        })}
      >
        <div className="primer-split">
          <PlotFrame id="primer-nudge-clip" label={t('primer.ui.calc-nudge.plot')}>
            <PlotGrid v={view} step={niceStep(view.x1 - view.x0)} />
            <path className="primer-curve" d={curvePath(view, curve.f)} />
            {answered && (
              <>
                <line className="primer-tangent" {...lineThrough(view, task.x, y, slope)} />
                <line
                  className="primer-delta"
                  x1={sx(view, task.x + task.dx)}
                  x2={sx(view, task.x + task.dx)}
                  y1={sy(view, y)}
                  y2={sy(view, y + actual)}
                />
                <line
                  className="primer-run"
                  x1={sx(view, task.x)}
                  x2={sx(view, task.x + task.dx)}
                  y1={sy(view, y)}
                  y2={sy(view, y)}
                />
                <circle
                  className="primer-point is-moved"
                  cx={sx(view, task.x + task.dx)}
                  cy={sy(view, y + actual)}
                  r={5}
                />
              </>
            )}
            <circle className="primer-point" cx={sx(view, task.x)} cy={sy(view, y)} r={5} />
          </PlotFrame>
          <div className="primer-controls">
            <dl className="primer-facts">
              <div>
                <dt>{t('primer.ui.calc-nudge.at')}</dt>
                <dd>x = {formatNumber(task.x)}</dd>
              </div>
              <div>
                <dt>{t('primer.ui.calc-nudge.slopeHere')}</dt>
                <dd>{formatNumber(slope)}</dd>
              </div>
              <div>
                <dt>{t('primer.ui.calc-nudge.turn')}</dt>
                <dd>
                  dx = {task.dx > 0 ? '+' : ''}
                  {formatNumber(task.dx)}
                </dd>
              </div>
            </dl>
            <p className="primer-instruction">{t('primer.ui.calc-nudge.instruction')}</p>
            {!answered ? (
              <NumberAnswer
                key={round}
                label={t('primer.ui.calc-nudge.guess')}
                submitLabel={t('primer.ui.calc-nudge.turnKnob')}
                onSubmit={submit}
              />
            ) : (
              <>
                <dl className="primer-facts">
                  <div>
                    <dt>{t('primer.ui.calc-nudge.yours')}</dt>
                    <dd>{formatNumber(guess, 4)}</dd>
                  </div>
                  <div>
                    <dt>{t('primer.ui.calc-nudge.actual')}</dt>
                    <dd>{formatNumber(actual, 4)}</dd>
                  </div>
                  <div>
                    <dt>Δf / dx</dt>
                    <dd>{formatNumber(actual / task.dx, 3)}</dd>
                  </div>
                </dl>
                <Feedback verdict={correct ? 'good' : 'bad'}>
                  {t(correct ? 'primer.ui.calc-nudge.good' : 'primer.ui.calc-nudge.bad', {
                    slope: formatNumber(slope),
                    dx: formatNumber(task.dx),
                  })}
                </Feedback>
                {finished && !nudgePassed(results) ? (
                  <NextButton onClick={restart}>{t('primer.ui.calc-nudge.again')} →</NextButton>
                ) : (
                  <NextButton onClick={next}>{t('primer.ui.nextRound')} →</NextButton>
                )}
              </>
            )}
          </div>
        </div>
      </PrimerPanel>
    </PrimerShell>
  );
}

// ---------------------------------------------------------------- calc-gears

const GEAR_TEETH = 10;
function gearOutline(radius: number): string {
  const points: string[] = [];
  const inner = radius * 0.8;
  for (let i = 0; i < GEAR_TEETH * 4; i += 1) {
    const angle = (i / (GEAR_TEETH * 4)) * Math.PI * 2;
    const r = i % 4 < 2 ? radius : inner;
    points.push(`${(Math.cos(angle) * r).toFixed(2)},${(Math.sin(angle) * r).toFixed(2)}`);
  }
  return points.join(' ');
}
const GEAR_OUTLINE = gearOutline(30);

/** A gear spinning at `speed` turns per second (negative = the other way); unknown = still. */
function Gear({
  speed,
  label,
  state,
  answer,
  t,
}: {
  speed?: number;
  label: string;
  state: 'known' | 'current' | 'pending';
  /** Rendered in place of the speed while this gear is the one being filled in. */
  answer?: ReactNode;
  t: PrimerLessonProps['t'];
}) {
  const spinning = speed !== undefined && speed !== 0;
  // One turn every 4 s at speed 1; fast gears are capped so they still read as turning.
  const duration = spinning ? Math.max(0.35, 4 / Math.abs(speed)) : 0;
  return (
    <figure className={`primer-gear is-${state}`}>
      <svg viewBox="-34 -34 68 68" aria-hidden="true">
        <g
          className={`primer-gear-body${spinning ? ' is-spinning' : ''}`}
          style={
            spinning
              ? {
                  animationDuration: `${duration}s`,
                  animationDirection: speed < 0 ? 'reverse' : 'normal',
                }
              : undefined
          }
        >
          <polygon points={GEAR_OUTLINE} />
          <line x1={0} y1={0} x2={0} y2={-20} />
          <circle r={4} />
        </g>
      </svg>
      <figcaption>
        <span className="label">{label}</span>
        {answer ?? (
          <b>
            {speed === undefined
              ? '?'
              : t('primer.ui.calc-gears.speed', { n: formatNumber(speed, 3) })}
          </b>
        )}
      </figcaption>
    </figure>
  );
}

export function GearsLesson(props: PrimerLessonProps) {
  const { t } = props;
  const [trainIndex, setTrainIndex] = useState(0);
  // How many gears after the input the player has filled in correctly.
  const [filled, setFilled] = useState(0);
  const [verdict, setVerdict] = useState<Verdict>();
  const [attempt, setAttempt] = useState(0);
  const done = trainIndex >= gearTrains.length;
  const train = gearTrains[Math.min(trainIndex, gearTrains.length - 1)];
  const speeds = gearSpeeds(train.stages);
  const complete = filled >= train.stages.length;
  // The gear being filled in now (index into speeds) and the gearbox that drives it.
  const current = filled + 1;
  const stage = train.stages[current - 1];
  const isFunction = train.stages.some((s) => s.labelKey);
  const gearLabel = (i: number) =>
    i === 0 ? t('primer.ui.calc-gears.input') : t('primer.ui.calc-gears.gearN', { n: i + 1 });

  return (
    <PrimerShell
      {...props}
      steps={gearTrains.map((_, i) => t('primer.ui.round', { n: i + 1 }))}
      current={trainIndex}
      done={done}
    >
      <PrimerPanel
        title={t(isFunction ? 'primer.ui.calc-gears.taskFn' : 'primer.ui.calc-gears.task')}
      >
        <p className="primer-instruction">
          {t(isFunction ? 'primer.ui.calc-gears.introFn' : 'primer.ui.calc-gears.intro')}
        </p>
        <div className="primer-gears">
          {speeds.map((speed, i) => {
            const known = i <= filled;
            return (
              <div className="primer-gear-step" key={i}>
                {i > 0 && (
                  <div className={`primer-gearbox${i === current ? ' is-current' : ''}`}>
                    {train.stages[i - 1].labelKey && (
                      <span className="primer-gearbox-fn">{t(train.stages[i - 1].labelKey!)}</span>
                    )}
                    <b>
                      {train.stages[i - 1].labelKey && !known
                        ? t('primer.ui.calc-gears.rateUnknown')
                        : `×${formatNumber(train.stages[i - 1].rate, 3)}`}
                    </b>
                  </div>
                )}
                <Gear
                  speed={known ? speed : undefined}
                  state={known ? 'known' : i === current ? 'current' : 'pending'}
                  label={gearLabel(i)}
                  answer={
                    i === current && !complete ? (
                      <InlineAnswer
                        key={`${trainIndex}-${current}-${attempt}`}
                        label={t('primer.ui.calc-gears.ask', { n: current + 1 })}
                        submitLabel={t('primer.ui.check')}
                        invalid={verdict === 'bad'}
                        onSubmit={(value) => {
                          setAttempt(attempt + 1);
                          if (!gearSpeedCorrect(value, speeds[current])) return setVerdict('bad');
                          setVerdict(undefined);
                          setFilled(filled + 1);
                        }}
                      />
                    ) : undefined
                  }
                  t={t}
                />
              </div>
            );
          })}
        </div>

        {!complete ? (
          <>
            <Feedback verdict={verdict}>
              {verdict === 'bad' &&
                t(stage.labelKey ? 'primer.ui.calc-gears.hintFn' : 'primer.ui.calc-gears.hint', {
                  previous: formatNumber(speeds[current - 1], 3),
                  rate: formatNumber(stage.rate, 3),
                })}
            </Feedback>
          </>
        ) : (
          <>
            <Feedback verdict="good">
              {t(isFunction ? 'primer.ui.calc-gears.doneFn' : 'primer.ui.calc-gears.done', {
                product: train.stages.map((s) => formatNumber(s.rate, 3)).join(' × '),
                rate: formatNumber(chainRate(train.stages), 3),
              })}
            </Feedback>
            <NextButton
              onClick={() => {
                setTrainIndex(trainIndex + 1);
                setFilled(0);
                setVerdict(undefined);
              }}
            >
              {t(trainIndex === gearTrains.length - 1 ? 'primer.ui.finish' : 'primer.ui.nextRound')}{' '}
              →
            </NextButton>
          </>
        )}
      </PrimerPanel>
    </PrimerShell>
  );
}

// ---------------------------------------------------------------- learning-rate (chapter 3)

const HILL_W = 560;
const HILL_H = 260;
const LOSS_H = 70;
/** Milliseconds between animated steps. */
const STEP_MS = 420;

type RunVerdict = 'arrived' | 'crawl' | 'diverge' | 'wobble' | 'short';

/** Why a run did or did not reach the bottom, read off its path. */
function judgeRun(path: readonly number[], stage: LearningRateStage): RunVerdict {
  if (reachedMinimum(path, stage)) return 'arrived';
  const crossings = path.slice(1).filter((x, i) => Math.sign(x) !== Math.sign(path[i])).length;
  const last = Math.abs(path.at(-1)!);
  if (crossings === 0) return last > stage.start / 4 ? 'crawl' : 'short';
  return last >= Math.abs(path[path.length - 2]) ? 'diverge' : 'wobble';
}

function useStepper(length: number, runId: number) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!runId) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return setShown(length - 1);
    setShown(0);
    let index = 0;
    const timer = window.setInterval(() => {
      index += 1;
      setShown(index);
      if (index >= length - 1) window.clearInterval(timer);
    }, STEP_MS);
    return () => window.clearInterval(timer);
  }, [runId, length]);
  return shown;
}

export function LearningRateLesson(props: PrimerLessonProps) {
  const { t } = props;
  const [stageIndex, setStageIndex] = useState(0);
  const done = stageIndex >= learningRateStages.length;
  const stage = learningRateStages[Math.min(stageIndex, learningRateStages.length - 1)];
  const [chosenRate, setChosenRate] = useState(0.05);
  const rate = stage.rate ?? chosenRate;
  const [run, setRun] = useState<{ id: number; path: number[] }>();
  const shown = useStepper(run?.path.length ?? 0, run?.id ?? 0);
  const finished = run !== undefined && shown >= run.path.length - 1;
  const verdict = finished ? judgeRun(run.path, stage) : undefined;
  const isGoal = stage.tolerance !== undefined;
  const passed = isGoal ? verdict === 'arrived' : finished;
  const curve = curves[stage.curve];

  const top = curve.f(curve.domain[1]) * 1.05;
  const view: Viewport = {
    x0: curve.domain[0],
    x1: curve.domain[1],
    y0: -top * 0.04,
    y1: top,
    width: HILL_W,
    height: HILL_H,
  };
  const visible = run ? run.path.slice(0, shown + 1) : [stage.start];
  const clampX = (x: number) => Math.max(view.x0, Math.min(view.x1, x));
  const point = (x: number) => ({
    cx: sx(view, clampX(x)),
    cy: sy(view, Math.min(top, curve.f(x))),
  });
  const walker = visible.at(-1)!;
  const escaped = Math.abs(walker) > view.x1;
  const before = visible.length > 1 ? visible[visible.length - 2] : undefined;
  const lossStart = curve.f(stage.start);
  // Scale bars to the run's worst loss (capped), so a diverging run visibly grows.
  const lossScale = Math.min(
    4 * lossStart,
    Math.max(lossStart, ...visible.map((x) => curve.f(x)).filter(Number.isFinite)),
  );

  const go = () => setRun({ id: (run?.id ?? 0) + 1, path: descend(stage, rate) });
  const advance = () => {
    setStageIndex(stageIndex + 1);
    setRun(undefined);
  };
  const stageKey = ['tooSmall', 'tooLarge', 'tune', 'steep'][Math.min(stageIndex, 3)];

  return (
    <PrimerShell
      {...props}
      badgeKey="primer.ui.experimentBadge"
      steps={['tooSmall', 'tooLarge', 'tune', 'steep'].map((key) =>
        t(`primer.ui.learning-rate.${key}Step`),
      )}
      current={stageIndex}
      done={done}
    >
      <PrimerPanel
        title={t(`primer.ui.learning-rate.${stageKey}Task`)}
        meta={t('primer.ui.learning-rate.steps', {
          used: Math.max(0, visible.length - 1),
          max: stage.steps,
        })}
      >
        <div className="primer-instruction">
          <MathText text={t(`primer.ui.learning-rate.${stageKey}Intro`)} />
        </div>
        <svg
          className="primer-plot is-wide"
          viewBox={`0 0 ${HILL_W} ${HILL_H}`}
          role="img"
          aria-label={t('primer.ui.learning-rate.plot')}
        >
          <PlotGrid v={view} step={1} yStep={niceStep(top)} />
          <path className="primer-curve" d={curvePath(view, curve.f)} />
          <line
            className="primer-valley-goal"
            x1={sx(view, 0)}
            x2={sx(view, 0)}
            y1={sy(view, 0)}
            y2={sy(view, top * 0.08)}
          />
          {visible.length > 1 && (
            <polyline
              className="primer-valley-path"
              points={visible.map((x) => `${point(x).cx},${point(x).cy}`).join(' ')}
            />
          )}
          {visible.slice(0, -1).map((x, i) => (
            <circle key={i} className="primer-trail" {...point(x)} r={3.5} />
          ))}
          <circle
            className={`primer-walker ${escaped ? 'is-escaped' : ''}`}
            {...point(walker)}
            r={8}
          />
          {escaped && (
            <text
              className="primer-escaped"
              x={walker > 0 ? HILL_W - 8 : 8}
              y={18}
              textAnchor={walker > 0 ? 'end' : 'start'}
            >
              {t('primer.ui.learning-rate.escaped')}
            </text>
          )}
        </svg>

        <div className="primer-loss" aria-label={t('primer.ui.learning-rate.loss')}>
          <span className="label">{t('primer.ui.learning-rate.loss')}</span>
          <div className="primer-loss-bars">
            {Array.from({ length: stage.steps + 1 }, (_, i) => {
              const x = visible[i];
              const loss = x === undefined ? undefined : curve.f(x);
              const ratio = loss === undefined ? 0 : Math.min(1, loss / lossScale);
              return (
                <span
                  key={i}
                  className={`primer-loss-bar ${loss !== undefined && loss > lossStart ? 'is-up' : ''}`}
                  style={{ height: `${Math.max(loss === undefined ? 0 : 3, ratio * LOSS_H)}px` }}
                  title={loss === undefined ? undefined : formatNumber(loss, 3)}
                />
              );
            })}
          </div>
        </div>

        <div className="primer-rate-row">
          <label className="primer-slider primer-rate">
            <span className="label">
              {t('primer.ui.learning-rate.rate')}{' '}
              <b className="primer-rate-value">η = {formatNumber(rate, 2)}</b>
              {stage.rate !== undefined && (
                <em className="primer-rate-lock">{t('primer.ui.learning-rate.locked')}</em>
              )}
            </span>
            <input
              type="range"
              min={LEARNING_RATE_RANGE.min}
              max={LEARNING_RATE_RANGE.max}
              step={LEARNING_RATE_RANGE.step}
              value={rate}
              disabled={stage.rate !== undefined || (run !== undefined && !finished)}
              onChange={(event) => {
                setChosenRate(Number(event.target.value));
                setRun(undefined);
              }}
            />
          </label>
          <button
            type="button"
            className="btn btn-primary"
            disabled={run !== undefined && !finished}
            onClick={go}
          >
            {t(run ? 'primer.ui.learning-rate.again' : 'primer.ui.learning-rate.go')}
          </button>
        </div>

        <p className="primer-dot-line primer-rule">
          {before === undefined
            ? t('primer.ui.learning-rate.rule')
            : `w ← ${formatNumber(before, 3)} − ${formatNumber(rate, 2)} × ${formatNumber(curve.df(before), 3)} = ${formatNumber(walker, 3)}`}
        </p>

        {finished && (
          <Feedback verdict={isGoal ? (passed ? 'good' : 'bad') : undefined}>
            {t(`primer.ui.learning-rate.${isGoal ? verdict : `${stageKey}Seen`}`, {
              steps: run.path.findIndex((x) => Math.abs(x) <= (stage.tolerance ?? 0)),
              rate: formatNumber(rate, 2),
            })}
          </Feedback>
        )}
        {passed && <NextButton onClick={advance}>{t('primer.ui.nextRound')} →</NextButton>}
      </PrimerPanel>
    </PrimerShell>
  );
}
