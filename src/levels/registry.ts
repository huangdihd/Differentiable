// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { ComponentRegistry, createDefaultRegistry, type GraphJSON } from '../engine/graph';
import type { CircuitLevel, Level, Progress } from '../engine/levels';
import { configurableModule } from '../engine/parameterized-components';

/** Palette and compiler policy: a level cannot use the component it is building. */
export function availableComponents(level: CircuitLevel, progress: Progress): string[] {
  const excluded = new Set([...(level.forbidden ?? []), ...(level.unlockComponents ?? [])]);
  return [
    ...new Set([...level.available, ...progress.unlockedComponents, ...(level.presets ?? [])]),
  ].filter((name) => !excluded.has(name));
}

/** Independent registry snapshots; never mutate persisted graphs or native definitions. */
export function buildPlayerRegistry(
  levels: readonly Level[],
  progress: Progress,
): ComponentRegistry {
  const circuitLevels = levels.filter((level) => level.kind !== 'observation');
  const presets = new Set(circuitLevels.flatMap((level) => [...(level.presets ?? [])]));
  const definitions = new Map(
    createDefaultRegistry()
      .list()
      .filter(
        (definition) =>
          !definition.subgraph ||
          definition.builtin ||
          presets.has(definition.name) ||
          progress.unlockedComponents.includes(definition.name),
      )
      .map((definition) => [definition.name, definition]),
  );
  for (const level of circuitLevels) {
    for (const name of level.unlockComponents ?? []) {
      const passed =
        progress.passed.includes(level.id) && progress.unlockedComponents.includes(name);
      const solution = passed ? progress.passedSolutions[level.id] : undefined;
      const existing = definitions.get(name);
      // Native-only unlocks grant access; they are not implementations of the lesson.
      if (existing && !existing.subgraph) continue;
      // Earned components remain usable after a saved solution is revoked. Unearned presets
      // keep their self-contained built-ins: chapter references may need unearned components.
      const source =
        solution ??
        (progress.unlockedComponents.includes(name) || (presets.has(name) && !existing)
          ? level.referenceGraph
          : undefined);
      if (!source) continue;
      const output = 'out';
      const subgraph: GraphJSON = structuredClone(source);
      for (const node of subgraph.nodes) {
        if (node.component === 'output' && (node.attrs?.name ?? 'out') === level.interface.output)
          node.attrs = { ...node.attrs, name: output };
      }
      definitions.set(name, {
        ...existing,
        name,
        inputs: Object.keys(level.interface.inputs),
        outputs: [output],
        subgraph,
      });
    }
  }
  const registry = new ComponentRegistry();
  for (const definition of definitions.values()) registry.register(configurableModule(definition));
  return registry;
}
