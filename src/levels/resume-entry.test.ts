import { expect, it } from 'vitest';
import { catalog, gameLevels, resumeEntry } from './catalog';

it('resumes at the first unlocked, unpassed level', () => {
  expect(resumeEntry(catalog, { unlocked: ['0-1'], passed: [] })?.level.id).toBe('0-1');
  const passed = gameLevels.slice(0, 3).map((l) => l.id);
  const next = gameLevels[3].id;
  expect(resumeEntry(catalog, { unlocked: [...passed, next], passed })?.level.id).toBe(next);
});

it('does not wrap to 0-1 after the last level (7-4) is passed', () => {
  const all = gameLevels.map((l) => l.id);
  expect(resumeEntry(catalog, { unlocked: all, passed: all })).toBeUndefined();
});
