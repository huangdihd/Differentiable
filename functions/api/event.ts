// Cloudflare Pages Function: POST /api/event stores one anonymous analytics event in D1.
// Requires a D1 binding named DB (see analytics/README.md). Without it, events are dropped.

interface D1Statement {
  bind(...values: unknown[]): { run(): Promise<unknown> };
}
interface Env {
  DB?: { prepare(sql: string): D1Statement };
}
interface Context {
  request: Request & { cf?: { country?: string; botManagement?: { verifiedBot?: boolean } } };
  env: Env;
  waitUntil(p: Promise<unknown>): void;
}

const EVENTS = new Set(['enter', 'attempt', 'complete', 'leave']);
const MAX_BODY = 1024;
const ID = /^[A-Za-z0-9-]{1,64}$/;
const LEVEL = /^[A-Za-z0-9_.-]{1,32}$/;
const BOT = /bot|crawl|spider|slurp|headless|lighthouse|preview|curl|wget|python|scrapy/i;
const DAY = 86_400_000;

// Best-effort per-isolate limit: at most LIMIT events per visitor per minute.
const LIMIT = 60;
const windows = new Map<string, { start: number; count: number }>();
function limited(key: string, now: number): boolean {
  if (windows.size > 5000) windows.clear();
  const w = windows.get(key);
  if (!w || now - w.start > 60_000) {
    windows.set(key, { start: now, count: 1 });
    return false;
  }
  return ++w.count > LIMIT;
}

const done = () => new Response(null, { status: 204 });

function int(v: unknown, max: number): number | null {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0
    ? Math.min(Math.round(v), max)
    : null;
}

export const onRequestPost = async ({ request, env, waitUntil }: Context): Promise<Response> => {
  try {
    if (!env.DB) return done();
    if (BOT.test(request.headers.get('user-agent') ?? '')) return done();
    if (request.cf?.botManagement?.verifiedBot) return done();
    if (Number(request.headers.get('content-length') ?? 0) > MAX_BODY) return done();
    const text = await request.text();
    if (text.length > MAX_BODY) return done();
    const b = JSON.parse(text) as Record<string, unknown>;
    if (
      typeof b.v !== 'string' ||
      !ID.test(b.v) ||
      typeof b.s !== 'string' ||
      !ID.test(b.s) ||
      typeof b.e !== 'string' ||
      !EVENTS.has(b.e) ||
      typeof b.level !== 'string' ||
      !LEVEL.test(b.level)
    )
      return done();
    const chapter = int(b.chapter, 99);
    if (chapter === null) return done();
    const now = Date.now();
    if (limited(b.v, now)) return done();
    const clientAt = int(b.t, now + DAY);
    waitUntil(
      env.DB.prepare(
        'INSERT INTO events (received_at, client_at, visitor, session, event, level, chapter, ms, attempts, passed, country) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
      )
        .bind(
          now,
          clientAt,
          b.v,
          b.s,
          b.e,
          b.level,
          chapter,
          int(b.ms, 7 * DAY),
          int(b.attempts, 10_000),
          typeof b.passed === 'boolean' ? Number(b.passed) : null,
          request.cf?.country ?? null,
        )
        .run()
        .catch(() => {}),
    );
  } catch {
    // Malformed input is ignored; the client never waits on this.
  }
  return done();
};
