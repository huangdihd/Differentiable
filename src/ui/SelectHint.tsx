// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.

/** A one-line reminder of how to select and pan on the sandbox canvas. */

const SEEN_KEY = 'differentiable.sandbox.selectHintSeen';

/** A UI-only flag kept apart from the progress save, so it never touches save validation. */
export function readSelectHintSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1';
  } catch {
    return false;
  }
}
export function writeSelectHintSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    // Private mode or full storage: the hint just keeps showing.
  }
}

export function modKeyLabel(platform = typeof navigator === 'undefined' ? '' : navigator.platform) {
  return /mac|iphone|ipad/i.test(platform) ? '⌘' : 'Ctrl';
}

export default function SelectHint({
  t,
  quiet,
}: {
  t: (key: string, params?: Record<string, string | number>) => string;
  /** Once the player has multi-selected, the hint fades back (hover brings it up again). */
  quiet: boolean;
}) {
  const mod = modKeyLabel();
  const items = ['box', 'add', 'all', 'pan'] as const;
  return (
    <div
      className={`canvas-select-hint${quiet ? ' is-quiet' : ''}`}
      aria-label={t('sandbox.select.hintLabel')}
    >
      {items.map((item, i) => (
        <span className="csh-item" key={item}>
          {i > 0 && <span className="csh-dot" aria-hidden="true" />}
          <kbd>{t(`sandbox.select.${item}Key`, { mod })}</kbd>
          <span>{t(`sandbox.select.${item}`)}</span>
        </span>
      ))}
    </div>
  );
}
