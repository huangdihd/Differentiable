// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { Fragment, useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react';
import {
  BATCH_HAND_CELLS,
  batchAnswer,
  batchPeople,
  batchQuestions,
  dot,
  featureKeys,
  games,
  productShape,
  WEIGHT_RANGE,
  rankGames,
  rankingGoalMet,
  rankingGoals,
  COMPASS_RANGE,
  compassFeatureKeys,
  compassGoalMet,
  compassGoals,
  compassPeople,
  compassScore,
  compassStart,
  standing,
  startingWeights,
  type RankingGoal,
  scoreRounds,
  scoreTable,
  shapePuzzles,
  shapesMultiply,
  type GameCard,
} from '../../levels/primer-linear';
import {
  Feedback,
  InlineAnswer,
  NextButton,
  PrimerPanel,
  PrimerShell,
  formatNumber,
  type PrimerLessonProps,
  type Translator,
} from './PrimerParts';

const gameById = (id: string): GameCard => games.find((game) => game.id === id)!;

// ---------------------------------------------------------------- la-score

/**
 * One scoring round written as the dot product itself, filled in left to right:
 * rating × weight = ? for every feature, then the total.
 */
function ScoreEquation({
  game,
  weights,
  t,
  onSolved,
}: {
  game: GameCard;
  weights: readonly number[];
  t: Translator;
  onSolved: () => void;
}) {
  const products = game.features.map((value, i) => value * weights[i]);
  const total = dot(game.features, weights);
  // 0..n-1: filling product i; n: filling the total; n+1: solved.
  const [filled, setFilled] = useState(0);
  const [wrong, setWrong] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const solved = filled > products.length;
  const answer = (expected: number) => (
    <InlineAnswer
      key={`${filled}-${attempt}`}
      label={
        filled < products.length
          ? t('primer.ui.la-score.askProduct', { feature: t(featureKeys[filled]) })
          : t('primer.ui.la-score.askTotal')
      }
      submitLabel={t('primer.ui.check')}
      invalid={wrong}
      onSubmit={(value) => {
        setAttempt(attempt + 1);
        if (Math.abs(value - expected) >= 0.05) return setWrong(true);
        setWrong(false);
        setFilled(filled + 1);
        if (filled === products.length) onSolved();
      }}
    />
  );
  const shown = (value: number) => <b className="primer-term-value">{formatNumber(value)}</b>;
  return (
    <>
      <div className="primer-equation" role="group" aria-label={t(game.nameKey)}>
        {products.map((product, i) => (
          <Fragment key={featureKeys[i]}>
            {i > 0 && <span className="primer-equation-op">+</span>}
            <div className={`primer-term${i === filled ? ' is-current' : ''}`}>
              <span className="label">{t(featureKeys[i])}</span>
              <span className="primer-term-math">
                <b className="primer-term-rating">{formatNumber(game.features[i])}</b>
                <span>×</span>
                <b className="primer-term-weight">{formatNumber(weights[i])}</b>
              </span>
              <span className="primer-term-result">
                = {i < filled ? shown(product) : i === filled ? answer(product) : <b>?</b>}
              </span>
            </div>
          </Fragment>
        ))}
        <span className="primer-equation-op">=</span>
        <div className={`primer-term is-total${filled === products.length ? ' is-current' : ''}`}>
          <span className="label">{t('primer.ui.la-score.total')}</span>
          <span className="primer-term-result">
            {solved ? shown(total) : filled === products.length ? answer(total) : <b>?</b>}
          </span>
        </div>
      </div>
      <p className="primer-legend">
        <span className="primer-term-rating">{t('primer.ui.la-score.legendRating')}</span> ×{' '}
        <span className="primer-term-weight">{t('primer.ui.la-score.legendWeight')}</span>
      </p>
      <Feedback verdict={solved ? 'good' : wrong ? 'bad' : undefined}>
        {solved
          ? t('primer.ui.la-score.good')
          : wrong &&
            (filled < products.length
              ? t('primer.ui.la-score.hintProduct', {
                  a: formatNumber(game.features[filled]),
                  b: formatNumber(weights[filled]),
                })
              : t('primer.ui.la-score.hintTotal', {
                  sum: products.map((product) => formatNumber(product)).join(' + '),
                }))}
      </Feedback>
    </>
  );
}

const RANK_ROW = 72;
// Placeholders that survive translation, so game names inside a goal sentence can be emphasised.
const NAME_TOP = '\u0000top\u0000';
const NAME_BOTTOM = '\u0000bottom\u0000';
function withNames(text: string, top: string, bottom: string) {
  return text.split(/(\u0000(?:top|bottom)\u0000)/).map((part, i) =>
    part === NAME_TOP ? (
      <b key={i}>{top}</b>
    ) : part === NAME_BOTTOM ? (
      <b key={i} className="is-bottom">
        {bottom}
      </b>
    ) : (
      part
    ),
  );
}

type GoalState = 'there' | 'tied' | undefined;
/** The big goal statement with live progress lines; turns green once the goal holds. */
function GoalBanner({
  met,
  text,
  status,
}: {
  met: boolean;
  text: ReactNode;
  status: readonly { key: string; text: string; state: GoalState }[];
}) {
  return (
    <div className={`primer-goal${met ? ' is-met' : ''}`} aria-live="polite">
      <span className="primer-goal-mark" aria-hidden="true">
        {met ? '✓' : '★'}
      </span>
      <div className="primer-goal-body">
        <p className="primer-goal-text">{text}</p>
        <ul className="primer-goal-status">
          {status.map((line) => (
            <li key={line.key} className={line.state ? `is-${line.state}` : undefined}>
              {line.text}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** One − / value / + dial per feature. */
function FeatureDials({
  values,
  range,
  disabled,
  onChange,
  groupLabel,
  lessLabel,
  moreLabel,
  t,
}: {
  values: readonly number[];
  range: { min: number; max: number; step: number };
  disabled: boolean;
  onChange: (values: number[]) => void;
  groupLabel: string;
  lessLabel: (feature: string) => string;
  moreLabel: (feature: string) => string;
  t: Translator;
}) {
  const nudge = (i: number, delta: number) => {
    const next = [...values];
    next[i] = Math.max(range.min, Math.min(range.max, next[i] + delta));
    onChange(next);
  };
  return (
    <div className="primer-weight-dials" role="group" aria-label={groupLabel}>
      {featureKeys.map((key, i) => (
        <div key={key} className="primer-weight-dial">
          <span className="label">{t(key)}</span>
          <div className="primer-weight-dial-row">
            <button
              type="button"
              className="btn"
              aria-label={lessLabel(t(key))}
              disabled={disabled || values[i] <= range.min}
              onClick={() => nudge(i, -range.step)}
            >
              −
            </button>
            <b className="primer-term-weight">{formatNumber(values[i])}</b>
            <button
              type="button"
              className="btn"
              aria-label={moreLabel(t(key))}
              disabled={disabled || values[i] >= range.max}
              onClick={() => nudge(i, range.step)}
            >
              +
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Competition rank from a score table: 1 + how many score strictly better; tied if any equal. */
function placeIn(scores: ReadonlyMap<string, number>, id: string, fromBottom = false) {
  const mine = scores.get(id)!;
  const others = [...scores].filter(([other]) => other !== id).map(([, value]) => value);
  return {
    rank: 1 + others.filter((value) => (fromBottom ? value < mine : value > mine)).length,
    tied: others.includes(mine),
  };
}

/**
 * The ranking game: drag the weights and watch every game's dot product, and the leaderboard,
 * change live until the goal holds.
 */
function RankingGoalPanel({
  goal,
  goalIndex,
  weights,
  onWeights,
  moved,
  onNext,
  last,
  t,
}: {
  goal: RankingGoal;
  goalIndex: number;
  weights: readonly number[];
  onWeights: (weights: number[]) => void;
  moved: boolean;
  onNext: () => void;
  last: boolean;
  t: Translator;
}) {
  const order = rankGames(weights);
  const scores = new Map(games.map((game) => [game.id, dot(game.features, weights)]));
  const maxScore = Math.max(1, ...[...scores.values()].map(Math.abs));
  // Only a goal the player reached by moving something counts (weights carry over).
  const met = moved && rankingGoalMet(weights, goal);
  const name = (id: string) => t(gameById(id).nameKey);

  return (
    <PrimerPanel
      title={t('primer.ui.la-score.rankPanel')}
      meta={t('primer.ui.la-score.goalCount', { n: goalIndex + 1, total: rankingGoals.length })}
    >
      <GoalBanner
        met={met}
        text={withNames(
          t(goal.bottom ? 'primer.ui.la-score.goalTopBottom' : 'primer.ui.la-score.goalTop', {
            top: NAME_TOP,
            bottom: NAME_BOTTOM,
          }),
          name(goal.top),
          goal.bottom ? name(goal.bottom) : '',
        )}
        status={[
          { id: goal.top, want: 1, fromBottom: false },
          ...(goal.bottom ? [{ id: goal.bottom, want: games.length, fromBottom: true }] : []),
        ].map(({ id, want, fromBottom }) => {
          const { rank: now, tied } = standing(weights, id);
          // A shared place never counts: the goal needs it alone at the top (or bottom).
          const there = !tied && standing(weights, id, fromBottom).rank === 1;
          return {
            key: id,
            state: there ? 'there' : tied ? 'tied' : undefined,
            text: t(tied ? 'primer.ui.la-score.statusTied' : 'primer.ui.la-score.status', {
              game: name(id),
              now,
              want,
            }),
          };
        })}
      />
      <p className="primer-instruction">{t('primer.ui.la-score.rankIntro')}</p>
      <FeatureDials
        values={weights}
        range={WEIGHT_RANGE}
        disabled={met}
        onChange={onWeights}
        groupLabel={t('primer.ui.la-score.weights')}
        lessLabel={(feature) => t('primer.ui.la-score.less', { feature })}
        moreLabel={(feature) => t('primer.ui.la-score.more', { feature })}
        t={t}
      />

      <ol
        className="primer-board"
        style={{ height: games.length * RANK_ROW }}
        aria-label={t('primer.ui.la-score.board')}
      >
        {games.map((game) => {
          const rank = order.indexOf(game.id);
          const score = scores.get(game.id)!;
          const isTop = game.id === goal.top;
          const isBottom = game.id === goal.bottom;
          return (
            <li
              key={game.id}
              className={`primer-board-row${isTop || isBottom ? ' is-goal' : ''}${met && (isTop || isBottom) ? ' is-met' : ''}`}
              style={{ transform: `translateY(${rank * RANK_ROW}px)` }}
              aria-posinset={rank + 1}
            >
              <span className="primer-board-rank">
                {standing(weights, game.id).rank}
                {standing(weights, game.id).tied && <small>{t('primer.ui.la-score.tied')}</small>}
              </span>
              <span className="primer-board-name">
                {name(game.id)}
                {isTop && <em>{t('primer.ui.la-score.wantTop')}</em>}
                {isBottom && <em>{t('primer.ui.la-score.wantBottom')}</em>}
              </span>
              <span className="primer-board-math">
                {game.features.map((value, i) => (
                  <Fragment key={i}>
                    {i > 0 && ' + '}
                    <span className="primer-term-rating">{formatNumber(value)}</span>×
                    <span className="primer-term-weight">{formatNumber(weights[i])}</span>
                  </Fragment>
                ))}
              </span>
              <span className="primer-board-bar">
                <span
                  className={score < 0 ? 'is-neg' : ''}
                  style={{ width: `${(Math.abs(score) / maxScore) * 100}%` }}
                />
              </span>
              <b className="primer-board-score">{formatNumber(score)}</b>
            </li>
          );
        })}
      </ol>

      <Feedback verdict={met ? 'good' : undefined}>
        {met &&
          t(`primer.ui.la-score.goalMet${goalIndex + 1}`, {
            top: name(goal.top),
            bottom: goal.bottom ? name(goal.bottom) : '',
          })}
      </Feedback>
      {met && (
        <NextButton onClick={onNext}>
          {t(last ? 'primer.ui.finish' : 'primer.ui.la-score.nextGoal')} →
        </NextButton>
      )}
    </PrimerPanel>
  );
}

export function ScoreLesson(props: PrimerLessonProps) {
  const { t } = props;
  // Steps 0..n-1 score one game each; then one step per ranking goal.
  const [step, setStep] = useState(0);
  const [solved, setSolved] = useState(false);
  const [weights, setWeights] = useState<number[]>(() => [...startingWeights]);
  const [moved, setMoved] = useState(false);
  const goalIndex = step - scoreRounds.length;
  const done = goalIndex >= rankingGoals.length;
  const steps = [
    ...scoreRounds.map((_, i) => t('primer.ui.round', { n: i + 1 })),
    ...rankingGoals.map((_, i) => t('primer.ui.la-score.goalStep', { n: i + 1 })),
  ];
  const advance = () => {
    setStep(step + 1);
    setSolved(false);
    setMoved(false);
  };

  return (
    <PrimerShell {...props} steps={steps} current={step} done={done}>
      {goalIndex < 0 ? (
        <PrimerPanel
          title={t('primer.ui.la-score.task', {
            game: t(gameById(scoreRounds[step].gameId).nameKey),
          })}
        >
          <p className="primer-instruction">
            {t(step === 0 ? 'primer.ui.la-score.intro' : 'primer.ui.la-score.introNegative')}
          </p>
          <ScoreEquation
            key={step}
            game={gameById(scoreRounds[step].gameId)}
            weights={scoreRounds[step].weights}
            t={t}
            onSolved={() => setSolved(true)}
          />
          {solved && <NextButton onClick={advance}>{t('primer.ui.nextRound')} →</NextButton>}
        </PrimerPanel>
      ) : (
        !done && (
          <RankingGoalPanel
            goal={rankingGoals[goalIndex]}
            goalIndex={goalIndex}
            weights={weights}
            onWeights={(next) => {
              setWeights(next);
              setMoved(true);
            }}
            moved={moved}
            onNext={advance}
            last={goalIndex === rankingGoals.length - 1}
            t={t}
          />
        )
      )}
    </PrimerShell>
  );
}

// ---------------------------------------------------------------- la-similar

const COMPASS = 380;
const COMPASS_R = 2.6;
const cx = (x: number) => COMPASS / 2 + (x / COMPASS_R) * (COMPASS / 2);
const cy = (y: number) => COMPASS / 2 - (y / COMPASS_R) * (COMPASS / 2);
const snapCompass = (value: number) =>
  Math.max(
    COMPASS_RANGE.min,
    Math.min(COMPASS_RANGE.max, Math.round(value / COMPASS_RANGE.step) * COMPASS_RANGE.step),
  );
const relation = (score: number) => (score > 0 ? 'agree' : score < 0 ? 'clash' : 'none');
const compassName = (id: string) => compassPeople.find((person) => person.id === id)!;

/**
 * The compass: everyone's taste is an arrow; the player drags theirs. Each arrow is coloured by
 * the sign of its dot product with yours, and the goal people cast a "shadow" onto your arrow.
 */
export function SimilarLesson(props: PrimerLessonProps) {
  const { t } = props;
  const [goalIndex, setGoalIndex] = useState(0);
  const [you, setYou] = useState<number[]>(() => [...compassStart]);
  const [moved, setMoved] = useState(false);
  const plane = useRef<SVGSVGElement>(null);
  const done = goalIndex >= compassGoals.length;
  const goal = compassGoals[Math.min(goalIndex, compassGoals.length - 1)];
  const scores = new Map(compassPeople.map((person) => [person.id, compassScore(you, person.id)]));
  const met = moved && compassGoalMet(you, goal);
  const name = (id: string) => t(compassName(id).nameKey);
  // The two people a goal sentence names, in the order its template uses them.
  const goalTop = goal.best ?? goal.zero!;
  const goalBottom = goal.best ? (goal.zero ?? goal.worst) : goal.worst;
  const roleOf = (id: string) =>
    id === goal.best
      ? 'wantBest'
      : id === goal.worst
        ? 'wantWorst'
        : id === goal.zero
          ? 'wantZero'
          : undefined;
  const textKey =
    goal.best && goal.zero
      ? 'primer.ui.la-similar.goalBestZero'
      : goal.zero && goal.worst
        ? 'primer.ui.la-similar.goalZeroWorst'
        : 'primer.ui.la-similar.goalBest';
  const setTip = (next: number[]) => {
    if (met) return;
    if (next[0] === you[0] && next[1] === you[1]) return;
    setYou(next);
    setMoved(true);
  };
  const dragTo = (event: PointerEvent<SVGSVGElement>) => {
    const box = plane.current!.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * COMPASS;
    const py = ((event.clientY - box.top) / box.height) * COMPASS;
    const x = snapCompass(((px - COMPASS / 2) / (COMPASS / 2)) * COMPASS_R);
    const y = snapCompass((-(py - COMPASS / 2) / (COMPASS / 2)) * COMPASS_R);
    if (x === 0 && y === 0) return;
    setTip([x, y]);
  };

  const status: { key: string; text: string; state: GoalState }[] = [];
  if (goal.best) {
    const { rank, tied } = placeIn(scores, goal.best);
    status.push({
      key: 'best',
      state: rank === 1 && !tied ? 'there' : tied ? 'tied' : undefined,
      text: t(tied ? 'primer.ui.la-similar.statusTied' : 'primer.ui.la-similar.statusBest', {
        name: name(goal.best),
        now: rank,
      }),
    });
  }
  if (goal.zero) {
    const value = scores.get(goal.zero)!;
    status.push({
      key: 'zero',
      state: value === 0 ? 'there' : undefined,
      text: t('primer.ui.la-similar.statusZero', {
        name: name(goal.zero),
        value: formatNumber(value),
      }),
    });
  }
  if (goal.worst) {
    const { rank, tied } = placeIn(scores, goal.worst, true);
    status.push({
      key: 'worst',
      state: rank === 1 && !tied ? 'there' : tied ? 'tied' : undefined,
      text: t(tied ? 'primer.ui.la-similar.statusTied' : 'primer.ui.la-similar.statusWorst', {
        name: name(goal.worst),
        now: placeIn(scores, goal.worst).rank,
        want: compassPeople.length,
      }),
    });
  }

  // Unit vector along your arrow, for the shadows.
  const length = Math.hypot(you[0], you[1]);
  const ux = you[0] / length;
  const uy = you[1] / length;
  const axisLabel = (feature: string, like: boolean) =>
    t(like ? 'primer.ui.la-similar.like' : 'primer.ui.la-similar.dislike', { feature: t(feature) });

  return (
    <PrimerShell
      {...props}
      steps={compassGoals.map((_, i) => t('primer.ui.la-score.goalStep', { n: i + 1 }))}
      current={goalIndex}
      done={done}
    >
      <PrimerPanel
        title={t('primer.ui.la-similar.panel')}
        meta={t('primer.ui.la-score.goalCount', { n: goalIndex + 1, total: compassGoals.length })}
      >
        <GoalBanner
          met={met}
          text={withNames(
            t(textKey, { top: NAME_TOP, bottom: NAME_BOTTOM }),
            name(goalTop),
            goalBottom ? name(goalBottom) : '',
          )}
          status={status}
        />
        <p className="primer-instruction">{t('primer.ui.la-similar.intro')}</p>
        <div className="primer-split primer-compass-split">
          <svg
            ref={plane}
            className="primer-plot primer-compass"
            viewBox={`0 0 ${COMPASS} ${COMPASS}`}
            role="img"
            tabIndex={0}
            aria-label={t('primer.ui.la-similar.plane', {
              x: formatNumber(you[0]),
              y: formatNumber(you[1]),
            })}
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              dragTo(event);
            }}
            onPointerMove={(event) => {
              if (event.buttons) dragTo(event);
            }}
            onKeyDown={(event) => {
              const step = COMPASS_RANGE.step;
              const delta: Record<string, [number, number]> = {
                ArrowLeft: [-step, 0],
                ArrowRight: [step, 0],
                ArrowUp: [0, step],
                ArrowDown: [0, -step],
              };
              const move = delta[event.key];
              if (!move) return;
              event.preventDefault();
              let next = [snapCompass(you[0] + move[0]), snapCompass(you[1] + move[1])];
              // A zero-length arrow has no direction: step straight over the origin.
              if (next[0] === 0 && next[1] === 0)
                next = [snapCompass(move[0] * 2), snapCompass(move[1] * 2)];
              setTip(next);
            }}
          >
            <defs>
              {['agree', 'clash', 'none', 'you'].map((kind) => (
                <marker
                  key={kind}
                  id={`compass-head-${kind}`}
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M0,0 L10,5 L0,10 z" className={`primer-compass-head is-${kind}`} />
                </marker>
              ))}
            </defs>
            {[-2, -1, 1, 2].map((v) => (
              <g key={v} className="primer-gridline">
                <line x1={cx(v)} x2={cx(v)} y1={0} y2={COMPASS} />
                <line x1={0} x2={COMPASS} y1={cy(v)} y2={cy(v)} />
              </g>
            ))}
            <line className="primer-axis" x1={0} x2={COMPASS} y1={cy(0)} y2={cy(0)} />
            <line className="primer-axis" x1={cx(0)} x2={cx(0)} y1={0} y2={COMPASS} />
            <text className="primer-compass-axis" x={COMPASS - 6} y={cy(0) - 8} textAnchor="end">
              {axisLabel(compassFeatureKeys[0], true)} →
            </text>
            <text className="primer-compass-axis" x={6} y={cy(0) - 8}>
              ← {axisLabel(compassFeatureKeys[0], false)}
            </text>
            <text className="primer-compass-axis" x={cx(0) - 8} y={16} textAnchor="end">
              ↑ {axisLabel(compassFeatureKeys[1], true)}
            </text>
            <text className="primer-compass-axis" x={cx(0) - 8} y={COMPASS - 8} textAnchor="end">
              ↓ {axisLabel(compassFeatureKeys[1], false)}
            </text>

            {compassPeople.map((person) => {
              const role = roleOf(person.id);
              if (!role) return null;
              // Projection of their arrow onto yours: the dot product divided by your length.
              const along = person.taste[0] * ux + person.taste[1] * uy;
              const fx = along * ux;
              const fy = along * uy;
              const kind = relation(scores.get(person.id)!);
              return (
                <g key={`shadow-${person.id}`} className={`primer-compass-shadow is-${kind}`}>
                  <line
                    className="primer-compass-drop"
                    x1={cx(person.taste[0])}
                    y1={cy(person.taste[1])}
                    x2={cx(fx)}
                    y2={cy(fy)}
                  />
                  <line
                    className="primer-compass-cast"
                    x1={cx(0)}
                    y1={cy(0)}
                    x2={cx(fx)}
                    y2={cy(fy)}
                  />
                </g>
              );
            })}

            {compassPeople.map((person) => {
              const kind = relation(scores.get(person.id)!);
              const role = roleOf(person.id);
              const [x, y] = person.taste;
              const away = Math.hypot(x, y);
              return (
                <g
                  key={person.id}
                  className={`primer-compass-person is-${kind}${role ? ' is-goal' : ''}`}
                >
                  <line
                    x1={cx(0)}
                    y1={cy(0)}
                    x2={cx(x)}
                    y2={cy(y)}
                    markerEnd={`url(#compass-head-${kind})`}
                  />
                  <text
                    x={Math.max(44, Math.min(COMPASS - 44, cx(x + (x / away) * 0.32)))}
                    y={cy(y + (y / away) * 0.32) + 4}
                    textAnchor="middle"
                  >
                    {role === 'wantBest'
                      ? '★ '
                      : role === 'wantWorst'
                        ? '▼ '
                        : role === 'wantZero'
                          ? '○ '
                          : ''}
                    {name(person.id)}
                  </text>
                </g>
              );
            })}

            <line
              className="primer-compass-you"
              x1={cx(0)}
              y1={cy(0)}
              x2={cx(you[0])}
              y2={cy(you[1])}
              markerEnd="url(#compass-head-you)"
            />
            <circle className="primer-compass-handle" cx={cx(you[0])} cy={cy(you[1])} r={9} />
            <text
              className="primer-compass-you-label"
              // Below the handle, where the other people's names (placed past their tips) are not.
              x={Math.max(44, Math.min(COMPASS - 44, cx(you[0])))}
              y={cy(you[1]) + (you[1] >= 0 ? 26 : -16)}
              textAnchor="middle"
            >
              {t('primer.ui.la-similar.you')}
            </text>
          </svg>

          <div className="primer-controls">
            <ul className="primer-compass-list">
              {compassPeople.map((person) => {
                const score = scores.get(person.id)!;
                const kind = relation(score);
                const role = roleOf(person.id);
                return (
                  <li
                    key={person.id}
                    className={`is-${kind}${role ? ' is-goal' : ''}${met && role ? ' is-met' : ''}`}
                  >
                    <span className="primer-compass-who">
                      {name(person.id)}
                      {role && <em>{t(`primer.ui.la-similar.${role}`)}</em>}
                    </span>
                    <span className="primer-compass-math">
                      {you.map((mine, i) => (
                        <Fragment key={i}>
                          {i > 0 && ' + '}
                          <span className="primer-term-weight">{formatNumber(mine)}</span>×
                          <span className="primer-term-rating">
                            {formatNumber(person.taste[i])}
                          </span>
                        </Fragment>
                      ))}
                    </span>
                    <b>{formatNumber(score)}</b>
                    <small>
                      {t(
                        kind === 'agree'
                          ? 'primer.ui.la-similar.similar'
                          : kind === 'clash'
                            ? 'primer.ui.la-similar.opposite'
                            : 'primer.ui.la-similar.unrelated',
                      )}
                    </small>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
        <Feedback verdict={met ? 'good' : undefined}>
          {met &&
            t(`primer.ui.la-similar.goalMet${goalIndex + 1}`, {
              top: name(goalTop),
              bottom: goalBottom ? name(goalBottom) : '',
            })}
        </Feedback>
        {met && (
          <NextButton
            onClick={() => {
              setGoalIndex(goalIndex + 1);
              setMoved(false);
            }}
          >
            {t(
              goalIndex === compassGoals.length - 1
                ? 'primer.ui.finish'
                : 'primer.ui.la-score.nextGoal',
            )}{' '}
            →
          </NextButton>
        )}
      </PrimerPanel>
    </PrimerShell>
  );
}

// ---------------------------------------------------------------- la-batch

type Cell = [number, number];
const sameCell = (a: Cell | undefined, b: Cell) => !!a && a[0] === b[0] && a[1] === b[1];
/** Row-major order: the hand-filled first row, then the rest swept in by the machine. */
const cellOrder: readonly Cell[] = batchPeople.flatMap((_, r) =>
  games.map((__, c): Cell => [r, c]),
);
/** Milliseconds between cells while "compute the rest" sweeps the table. */
const SWEEP_MS = 110;

/**
 * One table: each row is a person's weights, each column a game's ratings, each cell their dot
 * product. `focus` is highlighted with its row and column; in pick mode every cell is a button.
 */
function MatrixBoard({
  shown,
  focus,
  answer,
  onPick,
  picked,
  t,
}: {
  /** How many cells (row-major) show their value. */
  shown: number;
  focus?: Cell;
  /** Rendered inside the focused cell instead of its value. */
  answer?: ReactNode;
  onPick?: (cell: Cell) => void;
  picked?: { cell: Cell; right: boolean };
  t: Translator;
}) {
  const table = scoreTable();
  const [hover, setHover] = useState<Cell>();
  const lit = focus ?? hover;
  return (
    <div className="primer-matmul-scroll">
      <table className="primer-matmul">
        <thead>
          <tr>
            <th className="primer-matmul-corner" colSpan={1 + featureKeys.length}>
              <span className="primer-matmul-corner-key">
                <span>{t('primer.ui.la-batch.weightsHeader')} ↓</span>
                <span>{t('primer.ui.la-batch.ratingsHeader')} →</span>
              </span>
              {/* Lines up with each game's stacked ratings, so every rating has its name. */}
              <span className="primer-matmul-col primer-matmul-col-names">
                {featureKeys.map((key) => (
                  <span key={key}>{t(key)}</span>
                ))}
              </span>
            </th>
            {games.map((game, c) => (
              <th key={game.id} scope="col" className={lit && lit[1] === c ? 'is-lit' : undefined}>
                {t(game.nameKey)}
                <span className="primer-matmul-col">
                  {game.features.map((value, i) => (
                    <span key={i} title={t(featureKeys[i])}>
                      {formatNumber(value)}
                    </span>
                  ))}
                </span>
              </th>
            ))}
          </tr>
          <tr className="primer-matmul-features">
            <th />
            {featureKeys.map((key) => (
              <th key={key} scope="col">
                {t(key)}
              </th>
            ))}
            <th colSpan={games.length} />
          </tr>
        </thead>
        <tbody>
          {batchPeople.map((person, r) => (
            <tr key={person.id} className={lit && lit[0] === r ? 'is-lit' : undefined}>
              <th scope="row">{t(person.nameKey)}</th>
              {person.taste.map((value, i) => (
                <td key={i} className="is-weight">
                  {formatNumber(value)}
                </td>
              ))}
              {games.map((game, c) => {
                const index = r * games.length + c;
                const cell: Cell = [r, c];
                const isFocus = sameCell(focus, cell);
                const isPicked = picked && sameCell(picked.cell, cell);
                const value = index < shown ? formatNumber(table[r][c]) : '?';
                const classes = [
                  'primer-matmul-cell',
                  index < shown ? 'is-filled' : '',
                  isFocus ? 'is-focus' : '',
                  isPicked ? (picked!.right ? 'is-right' : 'is-wrong') : '',
                ].join(' ');
                return (
                  <td
                    key={game.id}
                    className={classes}
                    onPointerEnter={onPick ? () => setHover(cell) : undefined}
                    onPointerLeave={onPick ? () => setHover(undefined) : undefined}
                  >
                    {isFocus && answer ? (
                      answer
                    ) : onPick ? (
                      <button
                        type="button"
                        aria-label={`${t(person.nameKey)} · ${t(game.nameKey)}`}
                        onClick={() => onPick(cell)}
                        onFocus={() => setHover(cell)}
                        onBlur={() => setHover(undefined)}
                      >
                        {value}
                      </button>
                    ) : (
                      <span>{value}</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A matrix drawn as a block of rows × cols squares. */
function ShapeBlock({ shape, tone }: { shape: readonly [number, number]; tone?: string }) {
  return (
    <figure className={`primer-block ${tone ?? ''}`}>
      <span
        className="primer-block-grid"
        style={{ gridTemplateColumns: `repeat(${shape[1]}, 12px)` }}
        aria-hidden="true"
      >
        {Array.from({ length: shape[0] * shape[1] }, (_, i) => (
          <span key={i} />
        ))}
      </span>
      <figcaption>
        [<b className="is-rows">{shape[0]}</b>, <b className="is-cols">{shape[1]}</b>]
      </figcaption>
    </figure>
  );
}

function ShapePuzzleView({
  index,
  onSolved,
  t,
}: {
  index: number;
  onSolved: () => void;
  t: Translator;
}) {
  const puzzle = shapePuzzles[index];
  const [choice, setChoice] = useState<number>();
  const [attempt, setAttempt] = useState(0);
  const chosen = choice === undefined ? undefined : puzzle.options[choice];
  const fits = chosen !== undefined && shapesMultiply(puzzle.left, chosen);
  return (
    <PrimerPanel
      title={t('primer.ui.la-batch.shapeTask')}
      meta={t('primer.ui.la-batch.shapeCount', { n: index + 1, total: shapePuzzles.length })}
    >
      <p className="primer-instruction">{t('primer.ui.la-batch.shapeIntro')}</p>
      <div className="primer-shape-eq">
        <ShapeBlock shape={puzzle.left} tone="is-left" />
        <span className="primer-equation-op">×</span>
        {chosen && fits ? (
          <ShapeBlock shape={chosen} tone="is-placed" />
        ) : (
          <span className="primer-block-socket">
            {t('primer.ui.la-batch.socket', { n: puzzle.left[1] })}
          </span>
        )}
        <span className="primer-equation-op">=</span>
        {chosen && fits ? (
          <ShapeBlock shape={productShape(puzzle.left, chosen)} tone="is-result" />
        ) : (
          <span className="primer-block-socket is-result">?</span>
        )}
      </div>
      <div className="primer-block-tray" role="group" aria-label={t('primer.ui.la-batch.tray')}>
        {puzzle.options.map((option, i) => (
          <button
            type="button"
            key={`${i}-${choice === i ? attempt : 0}`}
            className={`primer-block-pick${choice === i && !fits ? ' is-wrong' : ''}`}
            disabled={fits}
            onClick={() => {
              setChoice(i);
              setAttempt(attempt + 1);
            }}
          >
            <ShapeBlock shape={option} />
          </button>
        ))}
      </div>
      <Feedback verdict={chosen ? (fits ? 'good' : 'bad') : undefined}>
        {chosen &&
          (fits
            ? t('primer.ui.la-batch.shapeGood', {
                inner: puzzle.left[1],
                rows: puzzle.left[0],
                cols: chosen[1],
              })
            : t('primer.ui.la-batch.shapeBad', { left: puzzle.left[1], right: chosen[0] }))}
      </Feedback>
      {fits && (
        <NextButton onClick={onSolved}>
          {t(index === shapePuzzles.length - 1 ? 'primer.ui.finish' : 'primer.ui.nextRound')} →
        </NextButton>
      )}
    </PrimerPanel>
  );
}

export function BatchLesson(props: PrimerLessonProps) {
  const { t } = props;
  const table = scoreTable();
  const total = cellOrder.length;
  // 0: fill the table; 1..q: read it; then one step per shape puzzle.
  const [stage, setStage] = useState(0);
  const [shown, setShown] = useState(0);
  const [sweeping, setSweeping] = useState(false);
  const [wrong, setWrong] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [picked, setPicked] = useState<{ cell: Cell; right: boolean }>();
  const questionIndex = stage - 1;
  const shapeIndex = stage - 1 - batchQuestions.length;
  const done = shapeIndex >= shapePuzzles.length;

  // The machine fills the remaining cells one after another.
  useEffect(() => {
    if (!sweeping) return;
    if (shown >= total) return setSweeping(false);
    const timer = window.setTimeout(() => setShown(shown + 1), SWEEP_MS);
    return () => window.clearTimeout(timer);
  }, [sweeping, shown, total]);

  const steps = [
    t('primer.ui.la-batch.fillStep'),
    ...batchQuestions.map((_, i) => t('primer.ui.la-batch.readStep', { n: i + 1 })),
    ...shapePuzzles.map((_, i) => t('primer.ui.la-batch.shapeStep', { n: i + 1 })),
  ];
  const advance = () => {
    setStage(stage + 1);
    setPicked(undefined);
  };

  const handFilling = stage === 0 && shown < BATCH_HAND_CELLS;
  const focus = handFilling ? cellOrder[shown] : undefined;
  const [fr, fc] = focus ?? [0, 0];

  let panel: ReactNode = null;
  if (stage === 0) {
    panel = (
      <PrimerPanel
        title={t('primer.ui.la-batch.fillTask')}
        meta={`[${batchPeople.length}, ${featureKeys.length}] × [${featureKeys.length}, ${games.length}] = [${batchPeople.length}, ${games.length}]`}
      >
        <p className="primer-instruction">{t('primer.ui.la-batch.fillIntro')}</p>
        <MatrixBoard
          shown={shown}
          focus={focus}
          answer={
            focus && (
              <InlineAnswer
                key={`${shown}-${attempt}`}
                label={t('primer.ui.la-batch.cell', {
                  person: t(batchPeople[fr].nameKey),
                  game: t(games[fc].nameKey),
                })}
                submitLabel={t('primer.ui.check')}
                invalid={wrong}
                onSubmit={(value) => {
                  setAttempt(attempt + 1);
                  if (Math.abs(value - table[fr][fc]) >= 0.05) return setWrong(true);
                  setWrong(false);
                  setShown(shown + 1);
                }}
              />
            )
          }
          t={t}
        />
        {focus && (
          <p className="primer-dot-line">
            {t(batchPeople[fr].nameKey)} · {t(games[fc].nameKey)} ={' '}
            {batchPeople[fr].taste.map((weight, i) => (
              <Fragment key={i}>
                {i > 0 && ' + '}
                <span className="primer-term-weight">{formatNumber(weight)}</span>×
                <span className="primer-term-rating">{formatNumber(games[fc].features[i])}</span>
              </Fragment>
            ))}
          </p>
        )}
        <Feedback verdict={wrong ? 'bad' : shown >= total ? 'good' : undefined}>
          {wrong
            ? t('primer.ui.la-batch.cellBad')
            : shown >= total
              ? t('primer.ui.la-batch.tableGood', { n: total })
              : ''}
        </Feedback>
        {!handFilling && shown < total && !sweeping && (
          <NextButton onClick={() => setSweeping(true)}>
            {t('primer.ui.la-batch.allAtOnce', { n: total - shown })}
          </NextButton>
        )}
        {shown >= total && <NextButton onClick={advance}>{t('primer.ui.nextRound')} →</NextButton>}
      </PrimerPanel>
    );
  } else if (questionIndex < batchQuestions.length) {
    const question = batchQuestions[questionIndex];
    const subject =
      question.kind === 'rowBest'
        ? t(batchPeople.find((person) => person.id === question.personId)!.nameKey)
        : t(games.find((game) => game.id === question.gameId)!.nameKey);
    const right = picked?.right ?? false;
    panel = (
      <PrimerPanel
        title={t('primer.ui.la-batch.readTask')}
        meta={t('primer.ui.la-batch.readCount', {
          n: questionIndex + 1,
          total: batchQuestions.length,
        })}
      >
        <GoalBanner
          met={right}
          text={withNames(
            t(`primer.ui.la-batch.ask.${question.kind}`, { top: NAME_TOP, bottom: NAME_BOTTOM }),
            subject,
            '',
          )}
          status={[]}
        />
        <p className="primer-instruction">{t('primer.ui.la-batch.readIntro')}</p>
        <MatrixBoard
          key={questionIndex}
          shown={total}
          onPick={(cell) => {
            if (right) return;
            const [ar, ac] = batchAnswer(question);
            setPicked({ cell, right: cell[0] === ar && cell[1] === ac });
          }}
          picked={picked}
          t={t}
        />
        <Feedback verdict={picked ? (right ? 'good' : 'bad') : undefined}>
          {picked &&
            t(`primer.ui.la-batch.${right ? 'good' : 'bad'}.${question.kind}`, { name: subject })}
        </Feedback>
        {right && <NextButton onClick={advance}>{t('primer.ui.nextRound')} →</NextButton>}
      </PrimerPanel>
    );
  } else if (!done) {
    panel = <ShapePuzzleView key={shapeIndex} index={shapeIndex} onSolved={advance} t={t} />;
  }

  return (
    <PrimerShell {...props} steps={steps} current={stage} done={done}>
      {panel}
    </PrimerShell>
  );
}
