// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { GraphJSON } from './engine/graph';
import {
  acknowledgeObservation,
  evaluateLevel,
  initialProgress,
  type CircuitLevel,
  type Level,
  type Progress,
} from './engine/levels';
import { availableComponents, buildPlayerRegistry } from './levels/registry';
import { browserLocale, isLocale, type Locale } from './i18n';

export const STORAGE_VERSION = 10;
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
            settings.learningRate > 0 &&
            (settings.optimizer === undefined ||
              settings.optimizer === 'sgd' ||
              settings.optimizer === 'adam'),
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
/** The gradient view is the engine's autodiff; it opens once the player has hand-built one. */
export function isAutodiffUnlocked(progress: Progress, levels: readonly Level[]): boolean {
  return levels.some(
    (level) =>
      progress.passed.includes(level.id) &&
      level.unlockFeatures?.includes('levels.features.autodiff'),
  );
}

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
/** An absent store/key is new progress; a failed read must never authorize an empty save. */
function readProgressEnvelope(source: StorageSource): {
  storage: StorageLike | undefined;
  raw: string | null;
  envelope?: Record<string, unknown>;
} {
  const storage = source();
  const raw = storage?.getItem(STORAGE_KEY) ?? null;
  if (raw === null) return { storage, raw };
  const envelope: unknown = JSON.parse(raw);
  if (!record(envelope)) throw new Error('Invalid progress envelope');
  return { storage, raw, envelope };
}
export function loadProgress(source: StorageSource = browserStorage): Progress {
  try {
    const { envelope } = readProgressEnvelope(source);
    if (!envelope) return createInitialProgress();
    if (envelope.version !== STORAGE_VERSION || !isProgress(envelope.progress))
      throw new Error('Unsupported or invalid progress');
    return envelope.progress;
  } catch (cause) {
    throw new Error('storage.progress_load_failed', { cause });
  }
}
/** Remap level IDs only: graph component names and unlocked components are stable. */
function remapProgressIds(progress: Progress, ids: Readonly<Record<string, string>>): Progress {
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

/** Levels taken out of the catalog, and the level a player who reached them goes on to. */
const retiredLevels: Readonly<Record<string, string>> = {
  // The final-state bottleneck lesson now opens the attention level's description.
  'time-recall-bottleneck': 'time-attention',
};

/** Forget retired levels without losing the way forward from them. */
export function removeRetiredLevels(progress: Progress): Progress {
  const retired = (id: string) => Object.hasOwn(retiredLevels, id);
  const touched = [...progress.unlocked, ...progress.passed].filter(retired);
  if (
    !touched.length &&
    ![progress.savedGraphs, progress.passedSolutions, progress.bestScores].some((v) =>
      Object.keys(v).some(retired),
    )
  )
    return progress;
  const keep = <T>(values: Record<string, T>): Record<string, T> =>
    Object.fromEntries(Object.entries(values).filter(([id]) => !retired(id)));
  return {
    ...progress,
    unlocked: [
      ...new Set([
        ...progress.unlocked.filter((id) => !retired(id)),
        ...touched.map((id) => retiredLevels[id]),
      ]),
    ],
    passed: progress.passed.filter((id) => !retired(id)),
    savedGraphs: keep(progress.savedGraphs),
    passedSolutions: keep(progress.passedSolutions),
    bestScores: keep(progress.bestScores),
    ...(progress.trainingSettings ? { trainingSettings: keep(progress.trainingSettings) } : {}),
  };
}

function migrateChapter2(progress: Progress): Progress {
  const ids: Record<string, string> = {
    '2-2': '2-3',
    '2-3': '2-4',
    '2-4': '2-5',
    '2-5': '2-7',
    '2-6': '2-8',
  };
  return remapProgressIds(progress, ids);
}

/** v4 put two gradient lessons before the former 1-10 chapter finale. */
function migrateChapter1GradientOrder(progress: Progress): Progress {
  const oldUnlocked = new Set(progress.unlocked);
  const oldPassed = new Set(progress.passed);
  const migrated = remapProgressIds(progress, {
    '1-10': '1-12',
    '1-11': '1-10',
    '1-12': '1-11',
  });
  const unlocked = migrated.unlocked.filter((id) => !['1-10', '1-11', '1-12'].includes(id));
  const reachedOldFinale = ['1-10', '1-11', '1-12'].some((id) => oldUnlocked.has(id));
  if (reachedOldFinale) unlocked.push('1-10');
  if (oldPassed.has('1-11') || oldUnlocked.has('1-12')) unlocked.push('1-11');
  if (oldPassed.has('1-10')) unlocked.push('1-12');
  return { ...migrated, unlocked: [...new Set(unlocked)] };
}

/** v5 removes the redundant second gradient lesson and moves backpropagation to 1-11. */
function migrateChapter1GradientMerge(progress: Progress): Progress {
  const oldUnlocked = new Set(progress.unlocked);
  const mapKeys = <T>(values: Record<string, T>): Record<string, T> => {
    const migrated = Object.fromEntries(
      Object.entries(values).filter(([id]) => id !== '1-11' && id !== '1-12'),
    );
    if (Object.hasOwn(values, '1-12')) migrated['1-11'] = values['1-12'];
    return migrated;
  };
  const savedGraphs = mapKeys(progress.savedGraphs);
  const passedSolutions = mapKeys(progress.passedSolutions);
  // v4's default 140px output spacing overlapped wrapped missing-input diagnostics.
  for (const graphs of [savedGraphs, passedSolutions]) {
    const graph = graphs['1-10'];
    const outputs = graph?.nodes.filter(
      (node) => node.component === 'output' && ['∂a', '∂b'].includes(String(node.attrs?.name)),
    );
    if (
      outputs?.length === 2 &&
      outputs[0].ui?.x === 660 &&
      outputs[0].ui?.y === 120 &&
      outputs[1].ui?.x === 660 &&
      outputs[1].ui?.y === 260
    ) {
      graphs['1-10'] = structuredClone(graph);
      graphs['1-10'].nodes.find((node) => node.id === outputs[1].id)!.ui = { x: 660, y: 320 };
    }
  }
  return {
    ...progress,
    unlocked: [
      ...new Set([
        ...progress.unlocked.filter((id) => id !== '1-11' && id !== '1-12'),
        ...(['1-11', '1-12'].some((id) => oldUnlocked.has(id)) ? ['1-11'] : []),
      ]),
    ],
    passed: progress.passed
      .filter((id) => id !== '1-11' && id !== '1-12')
      .concat(progress.passed.includes('1-12') ? ['1-11'] : []),
    savedGraphs,
    passedSolutions,
    bestScores: mapKeys(progress.bestScores),
    ...(progress.trainingSettings ? { trainingSettings: mapKeys(progress.trainingSettings) } : {}),
  };
}

/** v7 replaces every earlier 1-11 circuit with the local backward step for z=w·x+b. */
function migrateBackpropFullPass(progress: Progress, levels: readonly Level[]): Progress {
  const changed = levels.some(
    (level) =>
      level.kind !== 'observation' &&
      level.id === '1-11' &&
      Object.hasOwn(level.interface.inputs, '∂f/∂z'),
  );
  if (!changed) return progress;
  const graduated = progress.passed.includes('1-11');
  const withoutBackprop = <T>(values: Record<string, T>): Record<string, T> =>
    Object.fromEntries(Object.entries(values).filter(([id]) => id !== '1-11'));
  return {
    ...progress,
    unlocked: [
      ...new Set([
        ...progress.unlocked,
        ...(graduated && levels.some((level) => level.id === 'axes') ? ['axes'] : []),
      ]),
    ],
    passed: progress.passed.filter((id) => id !== '1-11'),
    // The old graph recomputed one Sigmoid derivative and has a different fixed interface.
    // A fresh editor graph is safer than leaving protected obsolete input/output nodes behind.
    savedGraphs: withoutBackprop(progress.savedGraphs),
    passedSolutions: withoutBackprop(progress.passedSolutions),
    bestScores: withoutBackprop(progress.bestScores),
    ...(progress.trainingSettings
      ? { trainingSettings: withoutBackprop(progress.trainingSettings) }
      : {}),
  };
}

/** Old chapter graduates can study the inserted gradient lesson without losing later access. */
function gradientLessonUnlocks(progress: Progress, levels: readonly Level[]): string[] {
  return levels.some((level) => level.id === '1-10') && progress.passed.includes('1-11')
    ? ['1-10']
    : [];
}

/** The tensor chapter predates the axes lesson; restore access, never completion. */
function axesUnlocks(progress: Progress, levels: readonly Level[]): string[] {
  return levels.some((level) => level.id === 'axes') &&
    (progress.unlocked.includes('2-1') || progress.passed.includes('1-11'))
    ? ['axes']
    : [];
}

/** The two-character model milestone predates reshape; restore access, never completion. */
function reshapeUnlocks(progress: Progress, levels: readonly Level[]): string[] {
  return levels.some((level) => level.id === 'reshape') &&
    (progress.unlocked.includes('4-5') || progress.passed.includes('4-4'))
    ? ['reshape']
    : [];
}

/** Loss now has a playable introduction; old players keep their route and can visit it. */
function lossIntroUnlocks(progress: Progress, levels: readonly Level[]): string[] {
  const intro = levels.findIndex((level) => level.id === 'loss-intro');
  if (intro < 0) return [];
  const reachedLaterLesson = levels
    .slice(intro + 1)
    .some((level) => progress.unlocked.includes(level.id) || progress.passed.includes(level.id));
  return reachedLaterLesson ? ['loss-intro'] : [];
}

/** Construction lessons moved before the vector-gradient milestone; restore access, never completion. */
function constructionUnlocks(progress: Progress, levels: readonly Level[]): string[] {
  const reachedVectorGradient =
    progress.unlocked.includes('3-2') || progress.passed.includes('3-2');
  const reachedLaterLesson = progress.unlocked.includes('3-3') || progress.passed.includes('3-3');
  return [
    ...(reachedVectorGradient || reachedLaterLesson ? ['neuron'] : []),
    ...(progress.passed.includes('3-2') || reachedLaterLesson ? ['layer'] : []),
  ].filter((id) => levels.some((level) => level.id === id));
}

/** Keep changed local-backprop drafts editable by repairing their protected interface inputs. */
function migrateUpstreamGradientInputs(progress: Progress, levels: readonly Level[]): Progress {
  const changed = new Map(
    levels
      .filter(
        (level): level is CircuitLevel =>
          level.kind !== 'observation' &&
          !!level.revalidateOnLoad &&
          Object.hasOwn(level.interface.inputs, '∂L/∂p') &&
          !Object.hasOwn(level.interface.inputs, 't'),
      )
      .map((level) => [level.id, level] as const),
  );
  const migrateGraphs = (graphs: Record<string, GraphJSON>): Record<string, GraphJSON> =>
    Object.fromEntries(
      Object.entries(graphs).map(([id, graph]) => {
        const level = changed.get(id);
        if (!level) return [id, graph];
        let migrated = graph;
        if (
          !graph.nodes.some((node) => node.component === 'input' && node.attrs?.name === '∂L/∂p')
        ) {
          const target = graph.nodes.find(
            (node) => node.component === 'input' && node.attrs?.name === 't',
          );
          if (target) {
            migrated = structuredClone(graph);
            const input = migrated.nodes.find((node) => node.id === target.id)!;
            input.attrs = { ...input.attrs, name: '∂L/∂p' };
          }
        }
        const expected = new Set(Object.keys(level.interface.inputs));
        const obsolete = new Set(
          migrated.nodes
            .filter(
              (node) =>
                node.component === 'input' &&
                typeof node.attrs?.name === 'string' &&
                !expected.has(node.attrs.name),
            )
            .map((node) => node.id),
        );
        if (!obsolete.size) return [id, migrated];
        if (migrated === graph) migrated = structuredClone(graph);
        migrated.nodes = migrated.nodes.filter((node) => !obsolete.has(node.id));
        migrated.edges = migrated.edges.filter(
          (edge) => !obsolete.has(edge.from.node) && !obsolete.has(edge.to.node),
        );
        return [id, migrated];
      }),
    );
  return {
    ...progress,
    savedGraphs: migrateGraphs(progress.savedGraphs),
    passedSolutions: migrateGraphs(progress.passedSolutions),
  };
}

/** Regression now places a scalar-batch Linear; repair drafts saved with its old 2-wide defaults. */
function migrateRegressionLinearDefaults(progress: Progress, levels: readonly Level[]): Progress {
  const level = levels.find((level) => level.id === '3-4');
  if (!level || level.kind === 'observation') return progress;
  const defaults = level.componentDefaults?.linear;
  if (defaults?.in_features !== 1 || defaults.out_features !== 1) return progress;
  const migrateGraphs = (graphs: Record<string, GraphJSON>): Record<string, GraphJSON> => {
    const graph = graphs['3-4'];
    if (!graph) return graphs;
    const input = graph.nodes.find(
      (node) => node.component === 'input' && node.attrs?.name === 'x',
    );
    const mse = new Set(
      graph.nodes.filter((node) => node.component === 'mse').map((node) => node.id),
    );
    const obsolete = graph.nodes.filter(
      (node) =>
        node.component === 'linear' &&
        node.attrs?.automatic_parameters === true &&
        [1, 2].includes(node.attrs.in_features as number) &&
        [1, 2].includes(node.attrs.out_features as number) &&
        (node.attrs.in_features !== 1 || node.attrs.out_features !== 1) &&
        graph.edges.some(
          (edge) =>
            edge.from.node === input?.id && edge.to.node === node.id && edge.to.port === 'x',
        ) &&
        graph.edges.some(
          (edge) => edge.from.node === node.id && mse.has(edge.to.node) && edge.to.port === 'p',
        ),
    );
    if (!obsolete.length) return graphs;
    const migrated = structuredClone(graph);
    const ids = new Set(obsolete.map((node) => node.id));
    for (const node of migrated.nodes)
      if (ids.has(node.id)) node.attrs = { ...node.attrs, in_features: 1, out_features: 1 };
    return { ...graphs, '3-4': migrated };
  };
  return {
    ...progress,
    savedGraphs: migrateGraphs(progress.savedGraphs),
  };
}

/** Momentum now exposes both recurrent state values; keep old drafts editable, then revalidate passes. */
function migrateMomentumOutputs(progress: Progress, levels: readonly Level[]): Progress {
  const level = levels.find((level) => level.id === '3-7');
  if (!level || level.kind === 'observation') return progress;
  const outputs = level.interface.outputs;
  if (!outputs?.updatedWeight || !outputs.updatedMomentum) return progress;
  const migrateGraphs = (graphs: Record<string, GraphJSON>): Record<string, GraphJSON> => {
    const graph = graphs['3-7'];
    if (!graph) return graphs;
    const weightOutput = graph.nodes.find(
      (node) =>
        node.component === 'output' &&
        ['y', 'updatedWeight'].includes(String(node.attrs?.name ?? 'y')),
    );
    const hasMomentum = graph.nodes.some(
      (node) => node.component === 'output' && node.attrs?.name === 'updatedMomentum',
    );
    if (!weightOutput || hasMomentum) return graphs;
    const migrated = structuredClone(graph);
    const output = migrated.nodes.find((node) => node.id === weightOutput.id)!;
    output.attrs = { ...output.attrs, name: 'updatedWeight' };
    const ids = new Set(migrated.nodes.map((node) => node.id));
    let id = 'output_momentum';
    for (let suffix = 2; ids.has(id); suffix++) id = `output_momentum_${suffix}`;
    migrated.nodes.push({
      id,
      component: 'output',
      attrs: { name: 'updatedMomentum' },
      ui: {
        x: typeof output.ui?.x === 'number' ? output.ui.x : 660,
        y: typeof output.ui?.y === 'number' ? output.ui.y + 200 : 320,
      },
    });
    return { ...graphs, '3-7': migrated };
  };
  return {
    ...progress,
    savedGraphs: migrateGraphs(progress.savedGraphs),
    passedSolutions: migrateGraphs(progress.passedSolutions),
  };
}

/** v8 introduced semantic outputs; v9 also repairs v8 graphs that retained generic `y`. */
function migrateSemanticOutputNames(
  progress: Progress,
  levels: readonly Level[],
  mapLevelId: (id: string) => string = (id) => id,
): Progress {
  const outputByLevel = new Map(
    levels
      .filter(
        (level): level is CircuitLevel =>
          level.kind !== 'observation' &&
          !level.interface.outputs &&
          level.interface.output !== 'y',
      )
      .map((level) => [level.id, level.interface.output]),
  );
  const migrateGraphs = (graphs: Record<string, GraphJSON>): Record<string, GraphJSON> =>
    Object.fromEntries(
      Object.entries(graphs).map(([id, graph]) => {
        const output = outputByLevel.get(mapLevelId(id));
        if (
          !output ||
          !graph.nodes.some(
            (node) => node.component === 'output' && String(node.attrs?.name ?? 'y') === 'y',
          )
        )
          return [id, graph];
        const migrated = structuredClone(graph);
        for (const node of migrated.nodes) {
          if (node.component !== 'output' || String(node.attrs?.name ?? 'y') !== 'y') continue;
          node.attrs = { ...node.attrs, name: output };
        }
        return [id, migrated];
      }),
    );
  return {
    ...progress,
    savedGraphs: migrateGraphs(progress.savedGraphs),
    passedSolutions: migrateGraphs(progress.passedSolutions),
  };
}

/** Observation completion has no circuit artifacts; keep read credit and all other drafts. */
function removeObservationCircuits(progress: Progress, levels: readonly Level[]): Progress {
  const observations = new Set(
    levels.filter((level) => level.kind === 'observation').map((level) => level.id),
  );
  const withoutObservations = <T>(values: Record<string, T>): Record<string, T> =>
    Object.fromEntries(Object.entries(values).filter(([id]) => !observations.has(id)));
  return {
    ...progress,
    savedGraphs: withoutObservations(progress.savedGraphs),
    passedSolutions: withoutObservations(progress.passedSolutions),
    bestScores: withoutObservations(progress.bestScores),
  };
}

/** Recheck changed contracts in catalog order so later lessons use repaired component fallbacks. */
async function revalidateSolutions(
  progress: Progress,
  levels: readonly Level[],
): Promise<string[]> {
  const revoked: string[] = [];
  for (const level of levels) {
    const solution = progress.passedSolutions[level.id];
    if (
      level.kind === 'observation' ||
      !level.revalidateOnLoad ||
      !progress.passed.includes(level.id)
    )
      continue;
    const result = solution
      ? await evaluateLevel(
          { ...level, available: availableComponents(level, progress) },
          solution,
          { registry: buildPlayerRegistry(levels, progress) },
        )
      : undefined;
    if (result?.passed) continue;
    if (solution) progress.savedGraphs[level.id] ??= structuredClone(solution);
    progress.passed = progress.passed.filter((id) => id !== level.id);
    delete progress.passedSolutions[level.id];
    delete progress.bestScores[level.id];
    revoked.push(level.id);
  }
  return revoked;
}

/** App awaits this entry point to validate legacy and changed-contract solutions.
 * Catalog order is progression order. Legacy milestones credit their prerequisite lessons,
 * using verified reference circuits when old graphs are absent or no longer compatible.
 * Changed-contract saved solutions are never replaced with automatic completion credit.
 * Optional onRevoked receives revoked level IDs once, after persistence (only when nonempty).
 * Failed reads, invalid saves, and migration exceptions reject without replacing the stored data.
 * Inactive loads or changed stored bytes reject before writing, notifying, or returning stale progress.
 * Missing storage or a missing key starts a new game; denied reads are not proof of missing data.
 * The return value remains Promise<Progress>; synchronous loadProgress does not revalidate.
 */
export async function loadProgressWithMigration(
  levels: readonly Level[],
  source: StorageSource = browserStorage,
  onRevoked?: (levelIds: string[]) => void,
  isCurrent: () => boolean = () => true,
): Promise<Progress> {
  try {
    const { storage, raw, envelope } = readProgressEnvelope(source);
    const finishLoad = (progress: Progress, persist: boolean, revoked: string[] = []): Progress => {
      // Even unchanged loads must not hand stale progress to the caller's autosave.
      const currentRaw = storage?.getItem(STORAGE_KEY) ?? null;
      if (!isCurrent() || currentRaw !== raw) throw new Error('Progress load is no longer current');
      if (persist) saveProgress(progress, () => storage);
      if (revoked.length) onRevoked?.(revoked);
      return progress;
    };
    if (!envelope) return finishLoad(createInitialProgress(), false);
    if (
      envelope.version === STORAGE_VERSION ||
      envelope.version === 9 ||
      envelope.version === 8 ||
      envelope.version === 7 ||
      envelope.version === 6 ||
      envelope.version === 5 ||
      envelope.version === 4 ||
      envelope.version === 3 ||
      envelope.version === 2
    ) {
      if (!isProgress(envelope.progress)) throw new Error('Invalid progress');
      let progress = envelope.progress;
      if (envelope.version === 2) progress = migrateChapter2(progress);
      if (envelope.version === 2 || envelope.version === 3)
        progress = migrateChapter1GradientOrder(progress);
      if ([2, 3, 4].includes(envelope.version as number))
        progress = migrateChapter1GradientMerge(progress);
      if ([2, 3, 4, 5, 6].includes(envelope.version as number))
        progress = migrateBackpropFullPass(progress, levels);
      // Idempotently repair saves written by early v8 builds as well as older envelopes.
      progress = migrateSemanticOutputNames(progress, levels);
      progress = migrateMomentumOutputs(progress, levels);
      progress = migrateUpstreamGradientInputs(progress, levels);
      progress = migrateRegressionLinearDefaults(progress, levels);
      // Repair current envelopes too: 5-1 became an observation without a version bump.
      progress = removeObservationCircuits(progress, levels);
      progress = removeRetiredLevels(progress);
      // New catalog successors must also unlock for players who already passed a lesson.
      const passedLevels = levels.filter((level) => progress.passed.includes(level.id));
      const restored = {
        ...progress,
        unlocked: [
          ...new Set([
            ...progress.unlocked,
            ...passedLevels.flatMap((l) => l.unlock),
            ...gradientLessonUnlocks(progress, levels),
            ...lossIntroUnlocks(progress, levels),
            ...constructionUnlocks(progress, levels),
            ...axesUnlocks(progress, levels),
            ...reshapeUnlocks(progress, levels),
          ]),
        ],
        unlockedComponents: [
          ...new Set([
            ...progress.unlockedComponents,
            ...passedLevels.flatMap((l) => l.unlockComponents ?? []),
          ]),
        ],
      };
      const repaired = JSON.stringify(restored) !== JSON.stringify(envelope.progress);
      const revoked = await revalidateSolutions(restored, levels);
      return finishLoad(
        restored,
        envelope.version !== STORAGE_VERSION || repaired || revoked.length > 0,
        revoked,
      );
    }
    if (envelope.version !== 1 || !record(envelope.progress))
      throw new Error('Unsupported or invalid legacy progress');
    const candidate = { ...envelope.progress, passedSolutions: {} };
    if (!isProgress(candidate)) throw new Error('Invalid legacy progress');
    const legacy = migrateBackpropFullPass(
      migrateChapter1GradientMerge(migrateChapter1GradientOrder(migrateChapter2(candidate))),
      levels,
    );
    const { progress, revoked } = await migrateLegacyProgress(legacy, levels);
    return finishLoad(progress, true, revoked);
  } catch (cause) {
    throw new Error('storage.progress_load_failed', { cause });
  }
}

async function migrateLegacyProgress(
  legacy: Progress,
  levels: readonly Level[],
): Promise<{ progress: Progress; revoked: string[] }> {
  const mapId = (id: string) => {
    if (levels.some((level) => level.id === id)) return id;
    if (id === 'sigmoid' || id === 'derivative')
      return (
        levels.find((level) => level.titleKey === `levels.${id}.title`)?.id ??
        (id === 'sigmoid' ? '0-5' : '1-2')
      );
    return id;
  };
  legacy = migrateRegressionLinearDefaults(
    migrateUpstreamGradientInputs(
      migrateMomentumOutputs(migrateSemanticOutputNames(legacy, levels, mapId), levels),
      levels,
    ),
    levels,
  );
  const known = new Set(levels.map((level) => level.id));
  let progress = createInitialProgress();
  progress.unlocked = [
    ...new Set([
      ...progress.unlocked,
      ...legacy.unlocked.map(mapId).filter((id) => known.has(id)),
      ...gradientLessonUnlocks(legacy, levels),
      ...lossIntroUnlocks(legacy, levels),
      ...constructionUnlocks(legacy, levels),
      ...axesUnlocks(legacy, levels),
      ...reshapeUnlocks(legacy, levels),
    ]),
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
  progress.unlockedComponents = [...legacy.unlockedComponents];
  const revoked: string[] = [];
  const passed = new Set(legacy.passed.map(mapId));
  const last = levels.reduce((index, level, i) => (passed.has(level.id) ? i : index), -1);
  for (const level of levels.slice(0, last + 1)) {
    if (level.kind === 'observation') {
      if (passed.has(level.id)) progress = acknowledgeObservation(progress, level);
      continue;
    }
    // New lessons must not become completed merely because a legacy milestone follows them.
    if (
      ['square', '1-10', '1-11', 'axes', 'reshape', 'loss-intro', 'neuron', 'layer'].includes(
        level.id,
      ) &&
      !passed.has(level.id)
    )
      continue;
    const saved = progress.savedGraphs[level.id];
    // A previously completed changed-contract circuit must be repaired by the player.
    const revalidate = level.revalidateOnLoad && passed.has(level.id) && saved;
    for (const graph of revalidate ? [saved] : [saved, level.referenceGraph]) {
      if (!graph) continue;
      const result = await evaluateLevel(
        { ...level, available: availableComponents(level, progress) },
        graph,
        { registry: buildPlayerRegistry(levels, progress) },
      );
      if (!result.passed) {
        if (revalidate) revoked.push(level.id);
        continue;
      }
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
  return { progress: removeObservationCircuits(progress, levels), revoked };
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
          envelope.version === 3 ||
          envelope.version === 4 ||
          envelope.version === 5 ||
          envelope.version === 6 ||
          envelope.version === 7 ||
          envelope.version === 8 ||
          envelope.version === 9 ||
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
 * Call flush on pagehide/unmount; cancel discards pending work. Failed writes remain pending
 * for an explicit flush or new schedule, without automatic retries. Exceptions report false.
 */
export function createThrottledSave<T>(save: (value: T) => boolean | void, delayMs = 300) {
  if (!Number.isFinite(delayMs) || delayMs < 0) throw new RangeError('Invalid save interval');
  let timer: ReturnType<typeof setTimeout> | undefined;
  let pending: { value: T } | undefined;
  const flush = (): boolean => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    const next = pending;
    if (!next) return true;
    try {
      if (save(next.value) === false) return false;
      // A save callback may schedule a newer snapshot; only clear the one just saved.
      if (pending === next) pending = undefined;
      return true;
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
