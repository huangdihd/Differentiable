// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { Level } from '../engine/levels';

const COMPONENT_SEEN_PREFIX = 'component:';

/** Stable per-browser marker shared by every level where a component may appear. */
export const componentSeenKey = (component: string) => `${COMPONENT_SEEN_PREFIX}${component}`;

/**
 * Existing players start with their current library as the baseline. Once that baseline exists,
 * components introduced by later levels remain new until the player uses them.
 */
export function initializeSeenComponents(
  seen: readonly string[],
  visibleComponents: readonly string[],
): string[] {
  if (seen.some((key) => key.startsWith(COMPONENT_SEEN_PREFIX))) return [...seen];
  return [...new Set([...seen, ...visibleComponents.map(componentSeenKey)])];
}

export function isNewComponent(seen: readonly string[], component: string): boolean {
  return !seen.includes(componentSeenKey(component));
}

function paletteComponents(level: Level): string[] {
  if (level.kind === 'observation') return [];
  const excluded = new Set([...(level.forbidden ?? []), ...(level.unlockComponents ?? [])]);
  return [...new Set([...level.available, ...(level.presets ?? [])])].filter(
    (component) => !excluded.has(component),
  );
}

/** Basic parts first exposed by the next lesson, which otherwise appear without an unlock notice. */
export function newlyIntroducedComponents(levels: readonly Level[], completed: Level): string[] {
  const index = levels.findIndex((level) => level.id === completed.id);
  if (index < 0) return [];
  const seen = new Set(
    levels
      .slice(0, index + 1)
      .flatMap((level) => [...paletteComponents(level), ...(level.unlockComponents ?? [])]),
  );
  const next = new Set(completed.unlock);
  return [
    ...new Set(
      levels
        .filter((level) => next.has(level.id))
        .flatMap(paletteComponents)
        .filter((component) => !seen.has(component)),
    ),
  ];
}
