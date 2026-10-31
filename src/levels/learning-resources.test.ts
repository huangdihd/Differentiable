// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { locales, type Locale } from '../i18n';
import { gameLevels } from './catalog';
import { levelLearningResources } from './learning-resources';

const registeredLocales = Object.keys(locales) as Locale[];

describe('learning resources', () => {
  it('attaches every concept to a real level only once', () => {
    const levelIds = new Set(gameLevels.map((level) => level.id));
    const concepts = new Set<string>();
    for (const [levelId, resources] of Object.entries(levelLearningResources)) {
      expect(levelIds.has(levelId), `unknown level ${levelId}`).toBe(true);
      for (const resource of resources) {
        expect(
          concepts.has(resource.titleKey),
          `${resource.titleKey} repeats after its introduction`,
        ).toBe(false);
        concepts.add(resource.titleKey);
      }
    }
    expect(levelLearningResources.axes?.[0].titleKey).toBe('learning.linearAlgebra.title');
    expect(levelLearningResources['4-1']?.[0].titleKey).toBe('learning.oneHot.title');
    expect(levelLearningResources['4-2']?.[0].titleKey).toBe('learning.embeddings.title');
  });

  it('has localized HTTPS links and translated metadata for every locale', () => {
    for (const resources of Object.values(levelLearningResources)) {
      for (const resource of resources) {
        expect(Object.keys(resource.url).sort()).toEqual([...registeredLocales].sort());
        for (const locale of registeredLocales) {
          expect(resource.url[locale]).toMatch(/^https:\/\//);
          expect(locales[locale]).toHaveProperty(resource.titleKey);
          expect(locales[locale]).toHaveProperty(resource.descriptionKey);
        }
      }
    }
  });
});
