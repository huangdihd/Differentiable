// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useState } from 'react';
import { saveSfxEnabled, sfx } from '../audio/sfx';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

/** Topbar switch for interface sounds, independent of the music. */
export function SfxToggle({ t }: { t: T }) {
  const [on, setOn] = useState(sfx.enabled);
  const label = t(on ? 'sfx.off' : 'sfx.on');
  return (
    <button
      type="button"
      className={`music-btn sfx-btn ${on ? 'is-on' : ''}`}
      aria-pressed={on}
      aria-label={label}
      title={label}
      onClick={() => {
        saveSfxEnabled(!on);
        setOn(!on);
        if (!on) sfx.play('connect');
      }}
    >
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
        <path d="M1.5 5h2.5l3-2.5v9L4 9H1.5z" fill="currentColor" />
        {on ? (
          <path
            d="M9.5 4.5a3.5 3.5 0 0 1 0 5M11 3a5.5 5.5 0 0 1 0 8"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.3"
          />
        ) : (
          <path d="M9.5 5l3.5 4M13 5l-3.5 4" fill="none" stroke="currentColor" strokeWidth="1.3" />
        )}
      </svg>
    </button>
  );
}
