// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useMemo, useState } from 'react';
import { createMusic } from '../audio/music';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

const KEY = 'differentiable.music';

function loadPrefs(): { on: boolean; volume: number } {
  try {
    const p = JSON.parse(localStorage.getItem(KEY) ?? '{}');
    return { on: p.on !== false, volume: typeof p.volume === 'number' ? p.volume : 0.5 };
  } catch {
    return { on: true, volume: 0.5 };
  }
}

/** Topbar music switch. Browsers only allow sound after a gesture, so playback waits for the first click. */
export function MusicToggle({ t, chapter }: { t: T; chapter: number }) {
  const music = useMemo(() => createMusic(), []);
  const [prefs, setPrefs] = useState(loadPrefs);
  const [playing, setPlaying] = useState(false);

  useEffect(() => () => music.dispose(), [music]);
  useEffect(() => music.setChapter(chapter), [music, chapter]);
  useEffect(() => {
    music.setVolume(prefs.volume);
    try {
      localStorage.setItem(KEY, JSON.stringify(prefs));
    } catch {
      // Preferences are a convenience; the game plays fine without them.
    }
  }, [music, prefs]);
  useEffect(() => {
    if (!prefs.on) {
      music.stop();
      setPlaying(false);
      return;
    }
    const begin = () => void music.start().then(() => setPlaying(music.playing));
    // Before any gesture Chrome can leave resume() pending forever, so only start once the
    // page has been interacted with; otherwise wait for the first interaction.
    if (navigator.userActivation?.hasBeenActive) begin();
    window.addEventListener('pointerdown', begin, { once: true });
    window.addEventListener('keydown', begin, { once: true });
    return () => {
      window.removeEventListener('pointerdown', begin);
      window.removeEventListener('keydown', begin);
    };
  }, [music, prefs.on]);

  const on = prefs.on;
  return (
    <span className={`music ${on ? 'is-on' : ''} ${on && playing ? 'is-playing' : ''}`}>
      <button
        type="button"
        className="music-btn"
        aria-pressed={on}
        aria-label={t(on ? 'music.off' : 'music.on')}
        title={t(on ? 'music.off' : 'music.on')}
        onClick={() => setPrefs((p) => ({ ...p, on: !p.on }))}
      >
        <span className="music-bars" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </button>
      <input
        type="range"
        className="music-volume"
        min={0}
        max={1}
        step={0.05}
        value={prefs.volume}
        disabled={!on}
        aria-label={t('music.volume')}
        onChange={(e) => setPrefs((p) => ({ ...p, volume: Number(e.target.value) }))}
      />
    </span>
  );
}
