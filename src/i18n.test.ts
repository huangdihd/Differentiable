import { describe, expect, it, vi } from 'vitest';
import { createDefaultRegistry } from './engine/graph';
import { exampleLevels } from './levels/examples';
import { gameLevels } from './levels/catalog';
import {
  createTranslator,
  diagnosticKey,
  formatNumber,
  locales,
  localeNames,
  matchLocale,
  translate,
  translateDiagnostic,
  translatePort,
} from './i18n';

describe('i18n', () => {
  it('uses the canonical component namespace without paired aliases', () => {
    for (const dictionary of Object.values(locales)) {
      for (const key of Object.keys(dictionary)) {
        expect(key).not.toMatch(/^component\./);
        if (key.startsWith('components.'))
          expect(dictionary).not.toHaveProperty(key.replace('components.', 'component.'));
      }
    }
  });
  it('translates component-specific ports before shared labels and preserves unknown ports', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      for (const locale of ['en', 'zh-CN'] as const) {
        const t = createTranslator(locale);
        expect(translatePort(t, 'one_hot', 'k')).toBe(locale === 'en' ? 'classes k' : '类别数 k');
        for (const component of ['attention', 'attn_scores', 'multi_head'])
          expect(translatePort(t, component, 'k')).toBe(locale === 'en' ? 'key k' : '键 k');
        expect(translatePort(t, 'reshape', 'd0')).toBe(t('ports.d0'));
        expect(translatePort(t, 'custom', 'x')).toBe(t('ports.x'));
        expect(translatePort(t, 'custom', 'custom_port')).toBe('custom_port');
        expect(translatePort(t, 'custom', 'toString')).toBe('toString');
      }
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
  it('falls back to English for a missing component-specific port translation', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ports.one_hot.k'];
    delete dictionary['ports.one_hot.k'];
    try {
      expect(translatePort(createTranslator('zh-CN'), 'one_hot', 'k')).toBe('classes k');
    } finally {
      dictionary['ports.one_hot.k'] = original;
      spy.mockRestore();
    }
  });
  it('renders shape-variant diagnostics with their input shapes in both locales', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const text = translateDiagnostic(locale, {
        key: 'levels.shape_variant',
        params: { shapes: 'a=[3×2], b=[2×5]' },
      });
      expect(text).toContain('a=[3×2], b=[2×5]');
      expect(text).not.toContain('{shapes}');
      expect(text).not.toBe('levels.shape_variant');
    }
  });
  it('derives native language names for every registered locale', () => {
    expect(Object.keys(localeNames)).toEqual(Object.keys(locales));
    for (const locale of Object.keys(locales) as (keyof typeof locales)[]) {
      expect(localeNames[locale]).toBe(
        new Intl.DisplayNames([locale], { type: 'language' }).of(locale) ?? locale,
      );
    }
  });
  it('matches browser language preferences in order', () => {
    expect(matchLocale(['fr', 'zh-TW', 'en'])).toBe('zh-CN');
    expect(matchLocale(['en-GB', 'zh-CN'])).toBe('en');
    expect(matchLocale(['ZH-cn'])).toBe('zh-CN');
    expect(matchLocale(['fr'])).toBe('en');
  });
  it('uses Intl plural rules, number formatting, and explicit locale factories', () => {
    const t = createTranslator('en');
    expect(t('ui.nodeCount', { count: 1 })).toBe('1 node');
    expect(t('ui.nodeCount', { count: 1234 })).toBe('1,234 nodes');
    expect(translate('zh-CN', 'ui.nodeCount', { count: 1 })).toBe('1 个节点');
    expect(formatNumber('en', 0.25, { style: 'percent' })).toBe('25%');
    expect(t('ui.step')).toBe('Step {count}');
  });
  it('falls back to English and warns in development without throwing', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ui.save'];
    delete dictionary['ui.save'];
    try {
      expect(translate('zh-CN', 'ui.save')).toBe('Save');
      expect(translate('en', 'not.a.key')).toBe('not.a.key');
      expect(translate('en', 'toString')).toBe('toString');
      expect(spy).toHaveBeenCalled();
    } finally {
      dictionary['ui.save'] = original;
      spy.mockRestore();
    }
  });
  it('uses English plural rules when falling back to an English message', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const dictionary = locales['zh-CN'] as Record<string, unknown>;
    const original = dictionary['ui.nodeCount'];
    delete dictionary['ui.nodeCount'];
    try {
      expect(translate('zh-CN', 'ui.nodeCount', { count: 1 })).toBe('1 node');
      expect(translate('zh-CN', 'ui.nodeCount', { count: 2 })).toBe('2 nodes');
    } finally {
      dictionary['ui.nodeCount'] = original;
      spy.mockRestore();
    }
  });
  it('covers every registered component, port and example level with both languages', () => {
    expect(Object.keys(locales.en).sort()).toEqual(Object.keys(locales['zh-CN']).sort());
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      for (const component of createDefaultRegistry().list()) {
        expect(keys.has(`components.${component.name}`)).toBe(true);
        for (const port of [...component.inputs, ...component.outputs])
          expect(keys.has(`ports.${port}`)).toBe(true);
      }
      for (const level of [...exampleLevels, ...gameLevels]) {
        expect(keys.has(level.titleKey)).toBe(true);
        expect(keys.has(level.descriptionKey)).toBe(true);
        for (const key of level.hintsKeys ?? []) expect(keys.has(key)).toBe(true);
        for (const name of level.unlockComponents ?? [])
          expect(keys.has(`components.${name}`)).toBe(true);
        for (const name of level.available) expect(keys.has(`components.${name}`)).toBe(true);
      }
    }
  });
  it('covers chapter-six weight ports in both languages', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const keys = new Set(Object.keys(locales[locale]));
      for (const port of ['wq', 'wk', 'wv', 'wo', 'w1', 'b1', 'w2', 'b2'])
        expect(keys.has(`ports.${port}`)).toBe(true);
    }
  });
  it('maps raw and namespaced diagnostics without exposing raw exceptions', () => {
    expect(diagnosticKey('missing_input')).toBe('errors.missing_input');
    expect(diagnosticKey('training.nonfinite')).toBe('training.nonfinite');
    expect(translateDiagnostic('en', { message: 'private raw exception' })).toBe(
      'Unable to complete this operation.',
    );
    expect(
      translateDiagnostic('en', { key: 'component_error', params: { message: 'raw' } }),
    ).not.toContain('raw');
  });
});
