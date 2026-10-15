# Generative music

No assets, dependencies or UI wiring. `createMusic()` is lazy and owns its context
(including contexts returned by the optional test factory); disposal closes it.
Call `start()` from a user gesture. Unsupported/blocked audio resolves silently;
`playing` becomes true only after resume succeeds. Stop sets `playing` false
immediately, fades for 350 ms, then disconnects voices/effects and suspends after
370 ms. Dispose is immediate and terminal. Volume defaults to 0.3; the master gain
is capped at 0.22, followed by a compressor. Volume changes ramp over 350 ms.

```ts
import { createMusic } from './audio/music';

const music = createMusic(); // Keep one instance per mounted game shell.
// Inside the future music button's user-gesture handler:
await music.start();
music.setVolume(0.3);
music.setChapter(entry?.chapter ?? 0); // App already obtains entryFor(level).
// Music toggle off:
music.stop();
// On shell unmount:
music.dispose();
```

All chapters use 70 BPM, 4/4, an eight-bar harmonic cycle (four chords, two bars
each). Notes are scale degrees rather than arbitrary chromatic pitches.

| Chapter | Tonality           | Voices (cumulative)                   |
| ------- | ------------------ | ------------------------------------- |
| 0       | A minor pentatonic | Soft sine pad, sparse triangle plucks |
| 1       | D minor pentatonic | + sine bass                           |
| 2       | E natural minor    | + second pluck layer                  |
| 3       | G minor pentatonic | + short, soft tonal pulse             |
| 4       | C natural minor    | + sustained sine upper layer          |
| 5       | F natural minor    | Denser second pluck layer             |
| 6       | A natural minor    | Additional pulse, fullest arrangement |

A 1.7 kHz lowpass softens the voices. A quiet single 640 ms delay provides space
without feedback buildup. Long envelopes overlap bar boundaries. Chapter changes
apply to the next not-yet-scheduled bar (up to 150 ms of audio is already queued);
previous notes finish their release rather than being abruptly cut off.

`generateBar(chapter, bar, seed)` returns seconds-relative note events. Each bar
uses a seeded PRNG keyed by absolute bar number; the harmonic cycle repeats but
the skipped plucks do not reset every cycle. The scheduler polls every 25 ms and
schedules 150 ms ahead using audio-clock time. After tab throttling it jumps to
the current bar and drops past events, never playing a backlog. This does not
promise uninterrupted music if the browser freezes the page for longer than the
look-ahead window. Start after stop begins the current bar afresh.

Vitest covers score constraints and a minimal fake AudioContext. Real-browser
listening, long-session comfort and device-specific audio behavior still need
manual acceptance; fake nodes cannot establish sound quality.
