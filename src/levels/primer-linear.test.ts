// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  BATCH_HAND_CELLS,
  batchAnswer,
  batchPeople,
  batchQuestions,
  classify,
  dot,
  featureKeys,
  games,
  linearPrimerLevels,
  mostSimilar,
  people,
  productShape,
  rankGames,
  WEIGHT_RANGE,
  rankingGoalMet,
  COMPASS_RANGE,
  compassGoalMet,
  compassGoals,
  compassPeople,
  compassScore,
  compassStart,
  standing,
  rankingGoals,
  startingWeights,
  scoreCorrect,
  scoreRounds,
  scoreTable,
  shapePuzzles,
  shapesMultiply,
  similarity,
  you,
} from './primer-linear';

describe('linear primer', () => {
  it('uses one graphics/difficulty/story feature per axis on four game cards', () => {
    expect(featureKeys).toEqual([
      'primer.feature.graphics',
      'primer.feature.difficulty',
      'primer.feature.story',
    ]);
    expect(games).toHaveLength(4);
    expect(new Set(games.map((game) => game.id)).size).toBe(4);
    for (const game of games) {
      expect(game.nameKey).toBe(`primer.game.${game.id}`);
      expect(game.features).toHaveLength(3);
      expect(
        game.features.every((value) => Number.isInteger(value) && value >= 0 && value <= 5),
      ).toBe(true);
    }
  });
  it('sums aligned products and rejects mismatched lengths in either order', () => {
    for (const operation of [dot, similarity]) {
      expect(operation([], [])).toBe(0);
      expect(operation([3, 2, 4], [-1, 0, 2])).toBe(5);
      expect(() => operation([1], [])).toThrow(RangeError);
      expect(() => operation([], [1])).toThrow(RangeError);
    }
  });
  it('scores both rounds, including one that penalizes difficulty', () => {
    expect(scoreRounds).toHaveLength(2);
    expect(scoreRounds.filter((round) => round.weights[1] < 0)).toHaveLength(1);
    for (const round of scoreRounds) {
      const game = games.find((candidate) => candidate.id === round.gameId)!;
      expect(round.weights).toHaveLength(3);
      expect(scoreCorrect(dot(game.features, round.weights), round)).toBe(true);
    }
    const round = { gameId: games[0].id, weights: [0, 0, 0] };
    for (const guess of [-0.0499, 0, 0.0499]) expect(scoreCorrect(guess, round)).toBe(true);
    for (const guess of [-0.051, -0.05, 0.05, 0.051, NaN])
      expect(scoreCorrect(guess, round)).toBe(false);
    expect(() => scoreCorrect(0, { ...round, gameId: 'missing' })).toThrow(RangeError);
  });
  it('ranks by score, highest first, ties by id', () => {
    expect(rankGames([0, 0, 1])).toEqual(['maze', 'archive', 'orbit', 'rush']);
    expect(rankGames([0, 0, 0])).toEqual(games.map((game) => game.id).sort());
    expect(() => rankGames([])).toThrow(RangeError);
  });
  it('ranks ties together and never calls a tie a win', () => {
    // Orbit and Comet both score 10.5 here: a shared first place.
    const weights = [2, 0.5, 0.5];
    expect(standing(weights, 'orbit')).toEqual({ rank: 1, tied: true });
    expect(standing(weights, 'rush')).toEqual({ rank: 1, tied: true });
    expect(standing(weights, 'archive')).toEqual({ rank: 3, tied: false });
    expect(standing(weights, 'maze', true)).toEqual({ rank: 1, tied: false });
    expect(rankingGoalMet(weights, { top: 'orbit' })).toBe(false);
    expect(rankingGoalMet([2, 1, 0.5], { top: 'orbit' })).toBe(true);
  });
  it('makes every compass goal reachable on the grid, with no goal true at the start', () => {
    const values: number[] = [];
    for (let v = COMPASS_RANGE.min; v <= COMPASS_RANGE.max + 1e-9; v += COMPASS_RANGE.step)
      values.push(v);
    for (const goal of compassGoals) {
      const solutions = values
        .flatMap((a) => values.map((b) => [a, b]))
        .filter((you) => compassGoalMet(you, goal));
      // Goal 2 has a single right direction (two snapped lengths); the others leave more room.
      expect(solutions.length).toBeGreaterThanOrEqual(2);
      expect(compassGoalMet(compassStart, goal)).toBe(false);
    }
    expect(compassPeople.map((person) => compassScore(compassStart, person.id))).toEqual([
      2, -4, 6, -2,
    ]);
    // Goal 2 has two perpendicular directions to Kai; only one puts Noor last.
    expect(compassGoalMet([2, 1], compassGoals[1])).toBe(true);
    expect(compassGoalMet([-2, -1], compassGoals[1])).toBe(false);
    // Goal 3 needs pure graphics, not pointing straight at Mira.
    expect(compassGoalMet([2, 0], compassGoals[2])).toBe(true);
    expect(compassGoalMet([2, 1], compassGoals[2])).toBe(false);
    expect(compassGoalMet([0, 0], { zero: 'kai' })).toBe(false);
  });
  it('makes every ranking goal reachable on the slider grid, and none true at the start', () => {
    const values: number[] = [];
    for (let v = WEIGHT_RANGE.min; v <= WEIGHT_RANGE.max + 1e-9; v += WEIGHT_RANGE.step)
      values.push(v);
    for (const goal of rankingGoals) {
      let reachable = 0;
      for (const a of values)
        for (const b of values)
          for (const c of values) if (rankingGoalMet([a, b, c], goal)) reachable += 1;
      // Several settings work, so the player can find one by feel.
      expect(reachable).toBeGreaterThanOrEqual(5);
    }
    expect(rankingGoalMet(startingWeights, rankingGoals[0])).toBe(false);
    // Strict wins: a tie at the top does not count.
    expect(rankingGoalMet([1, 1, 0], { top: 'orbit' })).toBe(true);
    expect(rankingGoalMet([1, 0, 0], { top: 'orbit' })).toBe(false);
    expect(rankingGoalMet([1, -1, 0], { top: 'rush', bottom: 'archive' })).toBe(true);
    expect(rankingGoalMet([1, 0, 0], { top: 'rush', bottom: 'archive' })).toBe(false);
  });
  it('compares four similarly sized taste vectors with a unique best match', () => {
    expect(people).toHaveLength(4);
    expect(new Set(people.map((person) => person.id)).size).toBe(4);
    for (const person of people) expect(person.nameKey).toBe(`primer.person.${person.id}`);
    for (const taste of [you, ...people.map((person) => person.taste)]) {
      expect(taste).toHaveLength(3);
      expect(taste.every((value) => Number.isInteger(value) && Math.abs(value) <= 2)).toBe(true);
      expect(Math.abs(Math.hypot(...taste) - Math.hypot(...you))).toBeLessThan(0.5);
    }
    const scores = people.map((person) => similarity(you, person.taste));
    expect([...scores].sort((a, b) => a - b)).toEqual([-4, 0, 3, 5]);
    expect(
      people.filter((_, index) => scores[index] === Math.max(...scores)).map((person) => person.id),
    ).toEqual([mostSimilar()]);
    expect(['mira', 'kai', 'noor', 'sora'].map(classify)).toEqual([
      'similar',
      'unrelated',
      'opposite',
      'similar',
    ]);
    expect(() => classify('missing')).toThrow(RangeError);
  });
  it('treats both classification thresholds as unrelated', () => {
    const person = people[0];
    const original = person.taste;
    try {
      for (const [score, expected] of [
        [-2, 'opposite'],
        [-1, 'unrelated'],
        [0, 'unrelated'],
        [1, 'unrelated'],
        [2, 'similar'],
      ] as const) {
        person.taste = [0, 0, score];
        expect(dot(you, person.taste)).toBe(score);
        expect(classify(person.id)).toBe(expected);
      }
    } finally {
      person.taste = original;
    }
  });
  it('returns a three-person by four-game score matrix', () => {
    expect(BATCH_HAND_CELLS).toBe(games.length);
    expect(batchPeople).toHaveLength(3);
    expect(new Set(batchPeople.map((person) => person.id)).size).toBe(3);
    const table = scoreTable();
    expect(table).toHaveLength(3);
    for (const [row, person] of batchPeople.entries()) {
      expect(table[row]).toHaveLength(4);
      expect(table[row]).toEqual(games.map((game) => dot(person.taste, game.features)));
      for (const [column, game] of games.entries())
        expect(scoreCorrect(table[row][column], { gameId: game.id, weights: person.taste })).toBe(
          true,
        );
    }
  });
  it('answers each table question with one unambiguous cell', () => {
    const table = scoreTable();
    expect(batchQuestions.map(batchAnswer)).toEqual([
      [0, 2],
      [1, 1],
    ]);
    for (const question of batchQuestions) {
      const [row, column] = batchAnswer(question);
      const line = question.kind === 'rowBest' ? table[row] : table.map((cells) => cells[column]);
      // The best value is unique, so clicking any other cell is clearly wrong.
      expect(line.filter((value) => value === table[row][column])).toHaveLength(1);
    }
  });
  it('accepts exactly the indexed answer in each of three shape puzzles', () => {
    expect(shapePuzzles).toHaveLength(3);
    for (const { left, options, answer } of shapePuzzles) {
      expect(options.flatMap((right, index) => (right[0] === left[1] ? [index] : []))).toEqual([
        answer,
      ]);
      for (const [index, right] of options.entries()) {
        expect(shapesMultiply(left, right)).toBe(index === answer);
        if (index === answer) expect(productShape(left, right)).toEqual([left[0], right[1]]);
        else expect(() => productShape(left, right)).toThrow(RangeError);
      }
    }
  });
  it('places each hint-free observation before its corresponding circuit', () => {
    const placements = [
      ['la-score', '2', '2-1'],
      ['la-batch', '2', '2-5'],
      ['la-similar', '6', '5-1'],
    ];
    expect(linearPrimerLevels).toEqual(
      placements.map(([id, chapter, successor]) => ({
        id,
        chapter,
        kind: 'observation',
        titleKey: `levels.${id}.title`,
        descriptionKey: `levels.${id}.description`,
        unlock: [successor],
      })),
    );
  });
});
