// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ObservationLevel } from '../engine/levels';

type Shape = readonly [number, number];

export const featureKeys: readonly string[] = [
  'primer.feature.graphics',
  'primer.feature.difficulty',
  'primer.feature.story',
];
export interface GameCard {
  id: string;
  nameKey: string;
  features: readonly number[];
}
export const games: readonly GameCard[] = [
  { id: 'rush', nameKey: 'primer.game.rush', features: [5, 1, 0] },
  { id: 'archive', nameKey: 'primer.game.archive', features: [1, 5, 3] },
  { id: 'maze', nameKey: 'primer.game.maze', features: [0, 2, 5] },
  { id: 'orbit', nameKey: 'primer.game.orbit', features: [4, 3, 2] },
];

export interface ScoreRound {
  gameId: string;
  weights: readonly number[];
}
export const scoreRounds: readonly ScoreRound[] = [
  { gameId: 'rush', weights: [2, 1, 0] },
  { gameId: 'archive', weights: [1, -2, 1] },
];
export function dot(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length)
    throw new RangeError('Dot product requires equal vector lengths');
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}
export function scoreCorrect(answer: number, round: ScoreRound): boolean {
  const game = games.find((candidate) => candidate.id === round.gameId);
  if (!game) throw new RangeError(`Unknown game: ${round.gameId}`);
  return Math.abs(answer - dot(game.features, round.weights)) < 0.05;
}
export function rankGames(weights: readonly number[]): string[] {
  return games
    .map((game) => ({ id: game.id, score: dot(game.features, weights) }))
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id, 'en'))
    .map((game) => game.id);
}
/**
 * la-score's last step: the player drags the three weights and watches the ranking re-sort
 * live, until a goal holds. Wins are strict (no ties), so the result is unambiguous.
 */
export const WEIGHT_RANGE = { min: -2, max: 2, step: 0.5 } as const;
/** Only graphics matters at first: the story-heavy maze starts at the bottom. */
export const startingWeights: readonly number[] = [1, 0, 0];
export interface RankingGoal {
  top: string;
  bottom?: string;
}
export const rankingGoals: readonly RankingGoal[] = [
  { top: 'maze' },
  { top: 'rush', bottom: 'archive' },
  { top: 'orbit' },
];
/**
 * Competition rank: 1 + how many games score strictly higher; `tied` when another game has the
 * same score. `fromBottom` counts from the lowest score instead.
 */
export function standing(
  weights: readonly number[],
  id: string,
  fromBottom = false,
): { rank: number; tied: boolean } {
  const score = (gameId: string) =>
    dot(games.find((game) => game.id === gameId)!.features, weights);
  const mine = score(id);
  const others = games.filter((game) => game.id !== id).map((game) => score(game.id));
  const ahead = others.filter((value) => (fromBottom ? value < mine : value > mine)).length;
  return { rank: ahead + 1, tied: others.includes(mine) };
}
export function rankingGoalMet(weights: readonly number[], goal: RankingGoal): boolean {
  const score = (id: string) => dot(games.find((game) => game.id === id)!.features, weights);
  const others = (id: string) =>
    games.filter((game) => game.id !== id).map((game) => score(game.id));
  if (others(goal.top).some((value) => value >= score(goal.top))) return false;
  if (goal.bottom && others(goal.bottom).some((value) => value <= score(goal.bottom!)))
    return false;
  return true;
}

export interface Person {
  id: string;
  nameKey: string;
  taste: readonly number[];
}
export const you: readonly number[] = [2, -1, 1];
export const people: readonly Person[] = [
  { id: 'mira', nameKey: 'primer.person.mira', taste: [1, -1, 2] },
  { id: 'kai', nameKey: 'primer.person.kai', taste: [1, 2, 0] },
  { id: 'noor', nameKey: 'primer.person.noor', taste: [-1, 2, 0] },
  { id: 'sora', nameKey: 'primer.person.sora', taste: [1, 1, 2] },
];
export const similarity = dot;
export function mostSimilar(): string {
  return people.reduce((best, person) =>
    similarity(you, person.taste) > similarity(you, best.taste) ? person : best,
  ).id;
}
export function classify(id: string): 'similar' | 'unrelated' | 'opposite' {
  const person = people.find((candidate) => candidate.id === id);
  if (!person) throw new RangeError(`Unknown person: ${id}`);
  const score = similarity(you, person.taste);
  if (score > 1) return 'similar';
  if (score < -1) return 'opposite';
  return 'unrelated';
}
/**
 * la-similar: a compass. Taste is two numbers (graphics, story), drawn as arrows from the
 * origin. The player drags their own arrow; similarity is the dot product, so pointing the same
 * way is positive, at right angles is zero and opposite is negative. Best/worst are strict and
 * `zero` means exactly unrelated; a zero-length arrow never meets a goal.
 */
export const compassFeatureKeys: readonly string[] = [
  'primer.feature.graphics',
  'primer.feature.story',
];
export const COMPASS_RANGE = { min: -2, max: 2, step: 0.5 } as const;
/** Starts nearest Kai, so no goal holds yet. */
export const compassStart: readonly number[] = [2, -2];
export const compassPeople: readonly Person[] = [
  { id: 'mira', nameKey: 'primer.person.mira', taste: [2, 1] },
  { id: 'sora', nameKey: 'primer.person.sora', taste: [0, 2] },
  { id: 'kai', nameKey: 'primer.person.kai', taste: [1, -2] },
  { id: 'noor', nameKey: 'primer.person.noor', taste: [-2, -1] },
];
export interface CompassGoal {
  best?: string;
  worst?: string;
  zero?: string;
}
export const compassGoals: readonly CompassGoal[] = [
  { best: 'sora' },
  { zero: 'kai', worst: 'noor' },
  { best: 'mira', zero: 'sora' },
];
export function compassScore(you: readonly number[], id: string): number {
  return similarity(you, compassPeople.find((person) => person.id === id)!.taste);
}
export function compassGoalMet(you: readonly number[], goal: CompassGoal): boolean {
  if (you.every((value) => value === 0)) return false;
  const others = (id: string) =>
    compassPeople
      .filter((person) => person.id !== id)
      .map((person) => compassScore(you, person.id));
  if (goal.best && others(goal.best).some((value) => value >= compassScore(you, goal.best!)))
    return false;
  if (goal.worst && others(goal.worst).some((value) => value <= compassScore(you, goal.worst!)))
    return false;
  if (goal.zero && compassScore(you, goal.zero) !== 0) return false;
  return true;
}

/** la-batch: the first row (one person against every game) is worked out by hand. */
export const BATCH_HAND_CELLS = 4;
export const batchPeople: readonly Person[] = [people[0], people[1], people[2]];
export function scoreTable(): number[][] {
  return batchPeople.map((person) => games.map((game) => dot(person.taste, game.features)));
}
/**
 * Reading the finished table: a row is one person's view of every game, a column is every
 * person's view of one game. The player answers by clicking the cell.
 */
export type BatchQuestion =
  { kind: 'rowBest'; personId: string } | { kind: 'columnBest'; gameId: string };
export const batchQuestions: readonly BatchQuestion[] = [
  { kind: 'rowBest', personId: 'mira' },
  { kind: 'columnBest', gameId: 'archive' },
];
/** The single cell that answers a question, as [row, column]. */
export function batchAnswer(question: BatchQuestion): [number, number] {
  const table = scoreTable();
  if (question.kind === 'rowBest') {
    const row = batchPeople.findIndex((person) => person.id === question.personId);
    const column = table[row].indexOf(Math.max(...table[row]));
    return [row, column];
  }
  const column = games.findIndex((game) => game.id === question.gameId);
  const values = table.map((cells) => cells[column]);
  return [values.indexOf(Math.max(...values)), column];
}

export interface ShapePuzzle {
  left: Shape;
  options: readonly Shape[];
  answer: number;
}
export const shapePuzzles: readonly ShapePuzzle[] = [
  {
    left: [4, 3],
    answer: 1,
    options: [
      [4, 2],
      [3, 2],
      [2, 3],
    ],
  },
  {
    left: [2, 5],
    answer: 2,
    options: [
      [2, 5],
      [3, 2],
      [5, 4],
    ],
  },
  {
    left: [3, 1],
    options: [
      [1, 4],
      [3, 1],
      [4, 3],
    ],
    answer: 0,
  },
];
export function shapesMultiply(left: Shape, right: Shape): boolean {
  return left[1] === right[0];
}
export function productShape(left: Shape, right: Shape): [number, number] {
  if (!shapesMultiply(left, right))
    throw new RangeError('Matrix product requires matching inner dimensions');
  return [left[0], right[1]];
}

export const linearPrimerLevels: ObservationLevel[] = [
  ['la-score', '2', '2-1'],
  ['la-batch', '2', '2-5'],
  ['la-similar', '6', '5-1'],
].map(([id, chapter, next]) => ({
  id,
  chapter,
  kind: 'observation',
  titleKey: `levels.${id}.title`,
  descriptionKey: `levels.${id}.description`,
  unlock: [next],
}));
