import { describe, expect, it } from 'vitest';
import {
  BACKEND_FALLBACK_DIALOG_KEY,
  dismissBackendFallbackDialog,
  isBackendFallbackDialogDismissed,
} from './backend-fallback-dialog';
import { locales } from './i18n';

describe('backend fallback dialog preference', () => {
  it('round-trips the dismiss flag through storage', () => {
    const store = new Map<string, string>();
    const storage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => {
        store.set(k, v);
      },
    };
    expect(isBackendFallbackDialogDismissed(storage)).toBe(false);
    dismissBackendFallbackDialog(storage);
    expect(store.get(BACKEND_FALLBACK_DIALOG_KEY)).toBe('1');
    expect(isBackendFallbackDialogDismissed(storage)).toBe(true);
  });

  it('treats missing or broken storage as not dismissed', () => {
    expect(isBackendFallbackDialogDismissed(undefined)).toBe(false);
    expect(
      isBackendFallbackDialogDismissed({
        getItem: () => {
          throw new Error('denied');
        },
      }),
    ).toBe(false);
    dismissBackendFallbackDialog({
      setItem: () => {
        throw new Error('denied');
      },
    });
  });
});

describe('backend fallback dialog copy', () => {
  const keys = [
    'ui.backendFallbackDialog.title',
    'ui.backendFallbackDialog.body',
    'ui.backendFallbackDialog.continue',
    'ui.backendFallbackDialog.dontShowAgain',
  ] as const;

  it('exists in both locales and keeps the corner fallback line', () => {
    for (const locale of ['zh-CN', 'en'] as const) {
      const dict = locales[locale];
      for (const key of keys) expect(dict[key], `${locale}:${key}`).toBeTruthy();
      expect(dict['ui.backendFallback']).toBeTruthy();
      expect(String(dict['levels.0-7.description'])).toMatch(/Maximum|最大值/);
    }
  });
});
