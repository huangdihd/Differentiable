// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { Random } from '../engine/random';

export type Voice = 'pad' | 'pluck' | 'bass' | 'echo' | 'pulse' | 'air';
export interface Note {
  /** Seconds relative to the beginning of the bar. */
  time: number;
  midi: number;
  duration: number;
  velocity: number;
  voice: Voice;
}

export const BPM = 70;
export const BAR_SECONDS = (60 / BPM) * 4;
const pentatonic = [0, 3, 5, 7, 10];
const minor = [0, 2, 3, 5, 7, 8, 10];
export const CHAPTER_MUSIC = [
  { root: 57, scale: pentatonic }, // A minor pentatonic
  { root: 50, scale: pentatonic }, // D minor pentatonic
  { root: 52, scale: minor }, // E natural minor
  { root: 55, scale: pentatonic }, // G minor pentatonic
  { root: 48, scale: minor }, // C natural minor
  { root: 59, scale: minor }, // B natural minor (time series)
  { root: 53, scale: minor }, // F natural minor (attention)
  { root: 57, scale: minor }, // A natural minor (GPT)
] as const;

export function clampChapter(chapter: number): number {
  return Number.isNaN(chapter) ? 0 : Math.max(0, Math.min(7, Math.floor(chapter)));
}

/** Pure, independently addressable bars; no mutable PRNG shared with the scheduler. */
export function generateBar(chapter: number, bar: number, seed: number): Note[] {
  chapter = clampChapter(chapter);
  bar = Math.max(0, Math.floor(bar));
  const { root, scale } = CHAPTER_MUSIC[chapter];
  const random = new Random((seed ^ Math.imul(bar + 1, 0x45d9f3b) ^ (chapter << 24)) | 0);
  const degree = [0, 3, 4, 2][Math.floor(bar / 2) % 4];
  const pitch = (n: number) => root + scale[n % scale.length] + 12 * Math.floor(n / scale.length);
  const notes: Note[] = [];
  const add = (voice: Voice, time: number, midi: number, duration: number, velocity: number) =>
    notes.push({ voice, time, midi, duration, velocity });
  for (const offset of [0, 2, 4]) add('pad', 0, pitch(degree + offset), BAR_SECONDS * 1.3, 0.1);
  // One guaranteed quiet note keeps the voice present even in very sparse bars.
  add('pluck', BAR_SECONDS / 4, pitch(degree) + 12, 1.2, 0.12);
  for (let step = 0; step < 8; step++) {
    if (step === 2 || random.uniform() > 0.1 + chapter * 0.065) continue;
    add('pluck', (step * BAR_SECONDS) / 8, pitch(Math.floor(random.uniform() * 5)) + 12, 1.2, 0.1);
  }
  if (chapter >= 1) add('bass', 0, pitch(degree) - 12, BAR_SECONDS * 1.1, 0.13);
  if (chapter >= 2) add('echo', BAR_SECONDS * 0.625, pitch(degree + 2) + 12, 1.8, 0.075);
  if (chapter >= 3) add('pulse', BAR_SECONDS / 2, root, 0.18, 0.045);
  if (chapter >= 4) add('air', 0, pitch(degree + 4), BAR_SECONDS * 1.4, 0.045);
  if (chapter >= 5) add('echo', BAR_SECONDS * 0.875, pitch(degree + 4), 1.8, 0.065);
  if (chapter >= 6) add('pulse', BAR_SECONDS * 0.75, root + 12, 0.16, 0.035);
  return notes.sort((a, b) => a.time - b.time);
}
