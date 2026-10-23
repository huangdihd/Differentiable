-- Anonymous per-level analytics events (see functions/api/event.ts).
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  received_at INTEGER NOT NULL,
  client_at INTEGER,
  visitor TEXT NOT NULL,
  session TEXT NOT NULL,
  event TEXT NOT NULL,
  level TEXT NOT NULL,
  chapter INTEGER NOT NULL,
  ms INTEGER,
  attempts INTEGER,
  passed INTEGER,
  country TEXT
);
CREATE INDEX IF NOT EXISTS events_level ON events (level, event);
CREATE INDEX IF NOT EXISTS events_visitor ON events (visitor);
CREATE INDEX IF NOT EXISTS events_received ON events (received_at);
