import { describe, expect, it, vi } from 'vitest';
import { createDefaultRegistry } from './engine/graph';
import { exampleLevels } from './levels/examples';
import {
  createTranslator,
  diagnosticKey,
  formatNumber,
  locales,
  localeNames,
  matchLocale,
  translate,
  translateDiagnostic,
} from './i18n';

describe('i18n', () => {
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
      for (const level of exampleLevels) {
        expect(keys.has(level.titleKey)).toBe(true);
        expect(keys.has(level.descriptionKey)).toBe(true);
        for (const name of level.available) expect(keys.has(`components.${name}`)).toBe(true);
      }
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
