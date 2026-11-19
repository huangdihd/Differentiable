// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ComponentType } from 'react';
import { GearsLesson, LearningRateLesson, NudgeLesson, SlopeLesson } from './CalculusLessons';
import { BatchLesson, ScoreLesson, SimilarLesson } from './LinearLessons';
import type { PrimerLessonProps } from './PrimerParts';
import './primer.css';

/** Interactive observation lessons (math primer plus later experiments), by stable level ID. */
const lessons: Readonly<Record<string, ComponentType<PrimerLessonProps>>> = {
  'calc-slope': SlopeLesson,
  'calc-nudge': NudgeLesson,
  'calc-gears': GearsLesson,
  'learning-rate': LearningRateLesson,
  'la-score': ScoreLesson,
  'la-similar': SimilarLesson,
  'la-batch': BatchLesson,
};

export const isPrimerLesson = (levelId: string) => Object.hasOwn(lessons, levelId);

export function PrimerLesson(props: PrimerLessonProps) {
  const Lesson = lessons[props.levelId];
  // Keyed so switching lessons never carries one lesson's state into the next.
  return <Lesson key={props.levelId} {...props} />;
}
