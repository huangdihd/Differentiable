import { ComponentRegistry, createDefaultRegistry, type GraphJSON } from '../engine/graph';
import type { Level, Progress } from '../engine/levels';

/** Palette and compiler policy: a level cannot use the component it is building. */
export function availableComponents(level: Level, progress: Progress): string[] {
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
  const presets = new Set(levels.flatMap((level) => [...(level.presets ?? [])]));
  const definitions = new Map(
    createDefaultRegistry()
      .list()
      .filter(
        (definition) =>
          !definition.subgraph ||
          presets.has(definition.name) ||
          progress.unlockedComponents.includes(definition.name),
      )
      .map((definition) => [definition.name, definition]),
  );
  for (const level of levels) {
    for (const name of level.unlockComponents ?? []) {
      const passed =
        progress.passed.includes(level.id) && progress.unlockedComponents.includes(name);
      const solution = passed ? progress.passedSolutions[level.id] : undefined;
      const existing = definitions.get(name);
      // Native-only unlocks grant access; they are not implementations of the lesson.
      if (existing && !existing.subgraph) continue;
      // Keep a self-contained built-in preset until the player has supplied a solution.
      // Chapter reference graphs may depend on components the player has not earned yet.
      const source =
        solution ?? (presets.has(name) && !existing ? level.referenceGraph : undefined);
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
  for (const definition of definitions.values()) registry.register(definition);
  return registry;
}
