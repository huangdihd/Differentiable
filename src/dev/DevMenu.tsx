// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Developer menu. Loaded only under the Vite dev server (`import.meta.env.DEV`); a production
 * build drops the import entirely. Toggle with the DEV button or Ctrl/⌘ + Shift + D.
 * Dev-only tooling, so its labels are not localized.
 */
import { useEffect, useState } from 'react';
import type { Level, Progress } from '../engine/levels';
import type { CatalogEntry } from '../levels/catalog';
import { devPass, devPassAll, devPassUntil, devReset, devUnlockAll } from './dev-progress';
import './dev-menu.css';

interface DevMenuProps {
  entries: readonly CatalogEntry[];
  progress: Progress;
  /** The open level, if any. */
  level?: Level;
  translate: (key: string) => string;
  onProgress: (progress: Progress) => void;
  onOpen: (level?: Level) => void;
}

export default function DevMenu({
  entries,
  progress,
  level,
  translate,
  onProgress,
  onOpen,
}: DevMenuProps) {
  const [open, setOpen] = useState(false);
  const [targetId, setTargetId] = useState(entries[0]?.level.id ?? '');
  const [confirmReset, setConfirmReset] = useState(false);
  const [notice, setNotice] = useState('');
  const levels = entries.map((entry) => entry.level);
  const target = entries.find((entry) => entry.level.id === targetId);
  const current = level && entries.find((entry) => entry.level.id === level.id);
  const nextOf = (of: Level) =>
    of.unlock.map((id) => levels.find((candidate) => candidate.id === id)).find(Boolean);
  const chapters = [...new Set(entries.map((entry) => entry.chapter))];

  useEffect(() => {
    const toggle = (event: KeyboardEvent) => {
      if (event.shiftKey && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        setOpen((value) => !value);
      }
    };
    window.addEventListener('keydown', toggle);
    return () => window.removeEventListener('keydown', toggle);
  }, []);
  // Follow the open level so "jump" defaults to where the developer is.
  useEffect(() => {
    if (level) setTargetId(level.id);
  }, [level]);

  const apply = (next: Progress, message: string, openLevel?: Level | null) => {
    onProgress(next);
    setNotice(message);
    setConfirmReset(false);
    // null: go home; undefined: stay where we are.
    if (openLevel !== undefined) onOpen(openLevel ?? undefined);
  };
  const label = (entry: CatalogEntry) => `${entry.code} · ${translate(entry.level.titleKey)}`;

  if (!open)
    return (
      <button
        type="button"
        className="dev-menu-toggle"
        title="Developer menu (Ctrl/⌘ + Shift + D)"
        aria-label="Developer menu"
        onClick={() => setOpen(true)}
      >
        <WrenchIcon />
      </button>
    );

  return (
    <aside className="dev-menu" aria-label="Developer menu">
      <header className="dev-menu-head">
        <span className="tag tag-accent">DEV</span>
        <span className="dev-menu-meta">
          {progress.passed.length}/{levels.length} passed
        </span>
        <button
          type="button"
          className="icon-btn"
          aria-label="Close developer menu"
          onClick={() => setOpen(false)}
        >
          ×
        </button>
      </header>

      {current && (
        <section className="dev-menu-section">
          <h3 className="label">Current · {label(current)}</h3>
          <div className="btn-row">
            <button
              type="button"
              className="btn"
              onClick={() => {
                const next = nextOf(current.level);
                apply(devPass(progress, current.level), `Passed ${current.code}`, next ?? null);
              }}
            >
              Pass & next
            </button>
          </div>
        </section>
      )}

      <section className="dev-menu-section">
        <h3 className="label">Jump to level</h3>
        <select
          className="field"
          value={targetId}
          onChange={(event) => setTargetId(event.target.value)}
        >
          {chapters.map((chapter) => (
            <optgroup key={chapter} label={translate(`chapter.${chapter}`)}>
              {entries
                .filter((entry) => entry.chapter === chapter)
                .map((entry) => (
                  <option key={entry.level.id} value={entry.level.id}>
                    {progress.passed.includes(entry.level.id) ? '✓ ' : ''}
                    {label(entry)} [{entry.level.id}]
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        <div className="btn-row">
          <button
            type="button"
            className="btn"
            disabled={!target}
            title="Unlock this level only and open it"
            onClick={() =>
              target &&
              apply(
                { ...progress, unlocked: [...new Set([...progress.unlocked, target.level.id])] },
                `Opened ${target.code}`,
                target.level,
              )
            }
          >
            Open
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!target}
            title="Pass every earlier level with its reference solution, then open this one"
            onClick={() =>
              target &&
              apply(
                devPassUntil(progress, levels, target.level.id),
                `Passed everything before ${target.code}`,
                target.level,
              )
            }
          >
            Pass all before & open
          </button>
        </div>
      </section>

      <section className="dev-menu-section">
        <h3 className="label">Whole save</h3>
        <div className="btn-row">
          <button
            type="button"
            className="btn"
            onClick={() => apply(devUnlockAll(progress, levels), 'Unlocked every level')}
          >
            Unlock all
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => apply(devPassAll(progress, levels), 'Passed every level')}
          >
            Pass all
          </button>
          <button
            type="button"
            className={`btn ${confirmReset ? 'dev-menu-danger' : 'btn-danger'}`}
            onClick={() => {
              if (!confirmReset) return setConfirmReset(true);
              apply(devReset(), 'Progress reset', null);
            }}
          >
            {confirmReset ? 'Click again to reset' : 'Reset progress'}
          </button>
        </div>
      </section>

      {notice && (
        <p className="dev-menu-notice" aria-live="polite">
          {notice}
        </p>
      )}
    </aside>
  );
}

function WrenchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path
        d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
