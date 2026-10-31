// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
// Anonymous per-level analytics for measuring how players move through the curriculum.
// Only random ids and level timings are sent; every failure is swallowed so gameplay never breaks.

export type AnalyticsEvent = 'enter' | 'attempt' | 'complete' | 'leave';

const ENDPOINT = '/api/event';
const VISITOR_KEY = 'differentiable.visitor';

function randomId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
}

function visitorId(): string {
  try {
    let id = localStorage.getItem(VISITOR_KEY);
    if (!id) {
      id = randomId();
      localStorage.setItem(VISITOR_KEY, id);
    }
    return id;
  } catch {
    return 'anonymous';
  }
}

const session = randomId();
let visitor: string | undefined;

function enabled(): boolean {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') return false;
  if (navigator.webdriver) return false;
  const host = location.hostname;
  return host !== 'localhost' && host !== '127.0.0.1';
}

export function track(
  event: AnalyticsEvent,
  data: { level: string; chapter: number; ms?: number; attempts?: number; passed?: boolean },
): void {
  try {
    if (!enabled()) return;
    visitor ??= visitorId();
    const body = JSON.stringify({ v: visitor, s: session, e: event, t: Date.now(), ...data });
    const blob = new Blob([body], { type: 'application/json' });
    if (navigator.sendBeacon?.(ENDPOINT, blob)) return;
    void fetch(ENDPOINT, { method: 'POST', body, keepalive: true }).catch(() => {});
  } catch {
    // Analytics must never interfere with the game.
  }
}

/** Tracks one visit to a level: enter, each graded attempt, completion, and leaving. */
export function levelTracker(level: string, chapter: number) {
  const start = Date.now();
  let attempts = 0;
  let completed = false;
  let left = false;
  const ms = () => Date.now() - start;
  const leave = () => {
    if (left) return;
    left = true;
    track('leave', { level, chapter, ms: ms(), attempts, passed: completed });
  };
  const onHide = () => {
    if (document.visibilityState === 'hidden') leave();
  };
  track('enter', { level, chapter });
  try {
    document.addEventListener('visibilitychange', onHide);
    window.addEventListener('pagehide', leave);
  } catch {
    // ignore
  }
  return {
    attempt(passed: boolean) {
      attempts++;
      track('attempt', { level, chapter, ms: ms(), attempts, passed });
      if (passed && !completed) {
        completed = true;
        track('complete', { level, chapter, ms: ms(), attempts, passed: true });
      }
    },
    dispose() {
      leave();
      try {
        document.removeEventListener('visibilitychange', onHide);
        window.removeEventListener('pagehide', leave);
      } catch {
        // ignore
      }
    },
  };
}
