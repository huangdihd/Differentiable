import type { GraphJSON } from './engine/graph';
import { evaluateLevel, initialProgress, type Level, type Progress } from './engine/levels';
import { availableComponents, buildPlayerRegistry } from './levels/registry';
import { browserLocale, isLocale, type Locale } from './i18n';

export const STORAGE_VERSION = 3;
export const STORAGE_KEY = 'differentiable.progress';
export const LANGUAGE_KEY = 'differentiable.language';
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}
/** A getter keeps even denied access to window.localStorage inside the try/catch. */
export type StorageSource = () => StorageLike | undefined;
const browserStorage: StorageSource = () => globalThis.localStorage;
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null &&
  typeof value === 'object' &&
  !Array.isArray(value) &&
  (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);
const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(text) && new Set(value).size === value.length;
/** Only finite JSON data; reject cycles and prototype-polluting keys at every depth. */
function jsonData(value: unknown, active = new Set<object>()): boolean {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || active.has(value)) return false;
  active.add(value);
  const valid = Array.isArray(value)
    ? value.every((item) => jsonData(item, active))
    : record(value) &&
      Object.entries(value).every(
        ([key, item]) =>
          !['__proto__', 'prototype', 'constructor'].includes(key) && jsonData(item, active),
      );
  active.delete(value);
  return valid;
}
/** Structural validation permits unfinished editor graphs; compilation owns semantic checks. */
export function isGraphJSON(value: unknown): value is GraphJSON {
  try {
    if (
      !record(value) ||
      value.version !== 1 ||
      !Array.isArray(value.nodes) ||
      !Array.isArray(value.edges) ||
      !jsonData(value)
    )
      return false;
    const ids = new Set<string>();
    for (const node of value.nodes) {
      if (
        !record(node) ||
        !text(node.id) ||
        !text(node.component) ||
        ids.has(node.id) ||
        (node.attrs !== undefined && !record(node.attrs)) ||
        (node.ui !== undefined && !record(node.ui))
      )
        return false;
      ids.add(node.id);
    }
    const endpoint = (item: unknown) =>
      record(item) && text(item.node) && ids.has(item.node) && text(item.port);
    return value.edges.every((edge) => record(edge) && endpoint(edge.from) && endpoint(edge.to));
  } catch {
    return false;
  }
}
export function isProgress(value: unknown): value is Progress {
  try {
    if (
      !record(value) ||
      !jsonData(value) ||
      !strings(value.unlocked) ||
      !strings(value.unlockedComponents) ||
      !strings(value.passed) ||
      !record(value.savedGraphs) ||
      !record(value.passedSolutions) ||
      !record(value.bestScores)
    )
      return false;
    if (
      value.trainingSettings !== undefined &&
      (!record(value.trainingSettings) ||
        !Object.entries(value.trainingSettings).every(
          ([id, settings]) =>
            text(id) &&
            record(settings) &&
            typeof settings.learningRate === 'number' &&
            Number.isFinite(settings.learningRate) &&
            settings.learningRate > 0,
        ))
    )
      return false;
    if (!value.passed.every((id) => (value.unlocked as string[]).includes(id))) return false;
    if (!Object.values(value.savedGraphs).every(isGraphJSON)) return false;
    if (
      !Object.entries(value.passedSolutions).every(
        ([id, graph]) => (value.passed as string[]).includes(id) && isGraphJSON(graph),
      )
    )
      return false;
    return Object.values(value.bestScores).every(
      (score) =>
        record(score) &&
        ['topNodes', 'expandedPrimitives', 'parameterCount'].every(
          (key) =>
            typeof score[key] === 'number' && Number.isSafeInteger(score[key]) && score[key] >= 0,
        ),
    );
  } catch {
    return false;
  }
}
/** Preserve the expression tool earned before chapter 1 gained extra lessons. */
export function isExpressionUnlocked(progress: Progress, levels: readonly Level[]): boolean {
  return (
    progress.unlocked.includes('2-1') ||
    levels.some(
      (level) =>
        progress.passed.includes(level.id) &&
        level.unlockFeatures?.includes('levels.features.expression'),
    )
  );
}

export function createInitialProgress(): Progress {
  return initialProgress(['0-1']);
}
export function loadProgress(source: StorageSource = browserStorage): Progress {
  try {
    const raw = source()?.getItem(STORAGE_KEY);
    if (raw) {
      const envelope: unknown = JSON.parse(raw);
      if (record(envelope) && envelope.version === STORAGE_VERSION && isProgress(envelope.progress))
        return envelope.progress;
    }
  } catch {
    /* Corrupt data, unavailable storage, and getter failures are recoverable. */
  }
  return createInitialProgress();
}
/** Remap level IDs only: graph component names and unlocked components are stable. */
function migrateChapter2(progress: Progress): Progress {
  const ids: Record<string, string> = {
    '2-2': '2-3',
    '2-3': '2-4',
    '2-4': '2-5',
    '2-5': '2-7',
    '2-6': '2-8',
  };
  const mapId = (id: string) => (Object.hasOwn(ids, id) ? ids[id] : id);
  const mapKeys = <T>(values: Record<string, T>): Record<string, T> =>
    Object.fromEntries(Object.entries(values).map(([id, value]) => [mapId(id), value]));
  return {
    ...progress,
    unlocked: [...new Set(progress.unlocked.map(mapId))],
    passed: [...new Set(progress.passed.map(mapId))],
    savedGraphs: mapKeys(progress.savedGraphs),
    passedSolutions: mapKeys(progress.passedSolutions),
    bestScores: mapKeys(progress.bestScores),
    ...(progress.trainingSettings ? { trainingSettings: mapKeys(progress.trainingSettings) } : {}),
  };
}

/** Load current saves synchronously elsewhere; App awaits this entry point to validate legacy solutions.
 * Catalog order is progression order. Legacy milestones credit their prerequisite lessons,
 * using verified reference circuits when old graphs are absent or no longer compatible.
 */
export async function loadProgressWithMigration(
  levels: readonly Level[],
  source: StorageSource = browserStorage,
): Promise<Progress> {
  let legacy: Progress;
  try {
    const raw = source()?.getItem(STORAGE_KEY);
    if (!raw) return createInitialProgress();
    const envelope: unknown = JSON.parse(raw);
    if (!record(envelope)) return createInitialProgress();
    if (envelope.version === STORAGE_VERSION || envelope.version === 2) {
      if (!isProgress(envelope.progress)) return createInitialProgress();
      const progress =
        envelope.version === 2 ? migrateChapter2(envelope.progress) : envelope.progress;
      // New catalog successors must also unlock for players who already passed a lesson.
      const passedLevels = levels.filter((level) => progress.passed.includes(level.id));
      const restored = {
        ...progress,
        unlocked: [...new Set([...progress.unlocked, ...passedLevels.flatMap((l) => l.unlock)])],
        unlockedComponents: [
          ...new Set([
            ...progress.unlockedComponents,
            ...passedLevels.flatMap((l) => l.unlockComponents ?? []),
          ]),
        ],
      };
      if (envelope.version === 2) saveProgress(restored, source);
      return restored;
    }
    if (envelope.version !== 1 || !record(envelope.progress)) return createInitialProgress();
    const candidate = { ...envelope.progress, passedSolutions: {} };
    if (!isProgress(candidate)) return createInitialProgress();
    legacy = migrateChapter2(candidate);
  } catch {
    return createInitialProgress();
  }
  const mapId = (id: string) => {
    if (levels.some((level) => level.id === id)) return id;
    if (id === 'sigmoid' || id === 'derivative')
      return (
        levels.find((level) => level.titleKey === `levels.${id}.title`)?.id ??
        (id === 'sigmoid' ? '0-5' : '1-2')
      );
    return id;
  };
  const known = new Set(levels.map((level) => level.id));
  const progress = createInitialProgress();
  progress.unlocked = [
    ...new Set([...progress.unlocked, ...legacy.unlocked.map(mapId).filter((id) => known.has(id))]),
  ];
  for (const [id, graph] of Object.entries(legacy.savedGraphs)) {
    if (known.has(mapId(id))) progress.savedGraphs[mapId(id)] = structuredClone(graph);
  }
  if (legacy.trainingSettings) {
    progress.trainingSettings = Object.fromEntries(
      Object.entries(legacy.trainingSettings)
        .filter(([id]) => known.has(mapId(id)))
        .map(([id, settings]) => [mapId(id), { ...settings }]),
    );
  }
  const passed = new Set(legacy.passed.map(mapId));
  const last = levels.reduce((index, level, i) => (passed.has(level.id) ? i : index), -1);
  for (const level of levels.slice(0, last + 1)) {
    const saved = progress.savedGraphs[level.id];
    for (const graph of [saved, level.referenceGraph]) {
      if (!graph) continue;
      const result = await evaluateLevel(
        { ...level, available: availableComponents(level, progress) },
        graph,
        { registry: buildPlayerRegistry(levels, progress) },
      );
      if (!result.passed) continue;
      progress.passed.push(level.id);
      progress.passedSolutions[level.id] = structuredClone(graph);
      progress.bestScores[level.id] = result.metrics;
      progress.unlocked = [...new Set([...progress.unlocked, level.id, ...level.unlock])];
      progress.unlockedComponents = [
        ...new Set([...progress.unlockedComponents, ...(level.unlockComponents ?? [])]),
      ];
      break;
    }
  }
  saveProgress(progress, source);
  return progress;
}
export function saveProgress(progress: Progress, source: StorageSource = browserStorage): boolean {
  try {
    if (!isProgress(progress)) return false;
    const storage = source();
    if (!storage) return false;
    storage.setItem(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress }));
    return true;
  } catch {
    return false;
  }
}
export function loadLanguage(
  source: StorageSource = browserStorage,
  fallback: Locale = browserLocale(),
): Locale {
  try {
    const raw = source()?.getItem(LANGUAGE_KEY);
    if (raw) {
      const envelope: unknown = JSON.parse(raw);
      if (
        record(envelope) &&
        (envelope.version === 1 ||
          envelope.version === 2 ||
          envelope.version === STORAGE_VERSION) &&
        isLocale(envelope.language)
      )
        return envelope.language;
    }
  } catch {
    /* Use browser preference if persistence is corrupt, unsupported, or blocked. */
  }
  return fallback;
}
export function saveLanguage(locale: Locale, source: StorageSource = browserStorage): boolean {
  try {
    if (!isLocale(locale)) return false;
    const storage = source();
    if (!storage) return false;
    storage.setItem(LANGUAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, language: locale }));
    return true;
  } catch {
    return false;
  }
}
/** Trailing throttle (not debounce): continuous edits save at most once per interval.
 * Call flush on pagehide/unmount; cancel discards pending work. Exceptions report false.
 */
export function createThrottledSave<T>(save: (value: T) => boolean | void, delayMs = 300) {
  if (!Number.isFinite(delayMs) || delayMs < 0) throw new RangeError('Invalid save interval');
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: { value: T } | undefined;
  const flush = (): boolean => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    const next = pending;
    pending = undefined;
    if (!next) return true;
    try {
      return save(next.value) !== false;
    } catch {
      return false;
    }
  };
  return {
    schedule(value: T) {
      pending = { value };
      if (timer === undefined) timer = setTimeout(flush, delayMs);
    },
    flush,
    cancel() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      pending = undefined;
    },
  };
}

/** UI convenience facade; individual progress and language APIs remain independently usable. */
export interface SaveData {
  version: typeof STORAGE_VERSION;
  progress: Progress;
  settings: { language: Locale };
}
export function loadSave(source: StorageSource = browserStorage): SaveData {
  return {
    version: STORAGE_VERSION,
    progress: loadProgress(source),
    settings: { language: loadLanguage(source) },
  };
}
export function saveSave(save: SaveData, source: StorageSource = browserStorage): boolean {
  try {
    if (
      save.version !== STORAGE_VERSION ||
      !isProgress(save.progress) ||
      !isLocale(save.settings.language)
    )
      return false;
    const progressSaved = saveProgress(save.progress, source);
    const languageSaved = saveLanguage(save.settings.language, source);
    return progressSaved && languageSaved;
  } catch {
    return false;
  }
}
