/**
 * Short synthesized interface sounds: no assets, same soft sine/triangle palette and A minor
 * pentatonic as the chapter-0 music, so effects sit inside the score instead of on top of it.
 */
export type SfxName =
  'place' | 'connect' | 'disconnect' | 'remove' | 'reject' | 'pass' | 'fail' | 'win';

export interface Sfx {
  /** `step` raises `pass` along the scale, so a run of passing test cases climbs. */
  play(name: SfxName, step?: number): void;
  setEnabled(on: boolean): void;
  readonly enabled: boolean;
  dispose(): void;
}

interface Tone {
  /** Seconds after the sound starts. */
  at: number;
  /** MIDI note; `to` glides to a second note over the tone's length. */
  midi: number;
  to?: number;
  length: number;
  gain: number;
  wave: OscillatorType;
}

/** A minor pentatonic from A4, by scale degree; degrees past the octave keep climbing. */
const PENTATONIC = [0, 3, 5, 7, 10];
export const degree = (n: number) =>
  69 + 12 * Math.floor(n / PENTATONIC.length) + PENTATONIC[((n % 5) + 5) % 5];

export function tones(name: SfxName, step = 0): Tone[] {
  const blip = (at: number, midi: number, length: number, gain: number, to?: number): Tone => ({
    at,
    midi,
    to,
    length,
    gain,
    wave: 'triangle',
  });
  switch (name) {
    case 'place':
      return [blip(0, 76, 0.07, 0.5, 69)];
    case 'connect':
      return [blip(0, degree(2), 0.06, 0.45), blip(0.05, degree(4), 0.1, 0.45)];
    case 'disconnect':
      return [blip(0, degree(4), 0.06, 0.35), blip(0.05, degree(1), 0.1, 0.35)];
    case 'remove':
      return [{ ...blip(0, 57, 0.12, 0.5, 45), wave: 'sine' }];
    case 'reject':
      return [blip(0, 50, 0.09, 0.45), blip(0.1, 49, 0.12, 0.45)];
    case 'pass':
      return [{ ...blip(0, degree(Math.min(step, 9)), 0.12, 0.3), wave: 'sine' }];
    case 'fail':
      return [blip(0, degree(2), 0.12, 0.45), blip(0.13, degree(0) - 1, 0.28, 0.45)];
    case 'win':
      return [
        ...[0, 2, 4, 5].map((d, i) => blip(i * 0.09, degree(d), 0.22, 0.4)),
        { ...blip(0.36, degree(5), 0.9, 0.3), wave: 'sine' },
        { ...blip(0.36, degree(3), 0.9, 0.22), wave: 'sine' },
      ];
  }
}

const hz = (midi: number) => 440 * 2 ** ((midi - 69) / 12);
const LEVEL = 0.18;

export function createSfx(options: { context?: () => AudioContext; enabled?: boolean } = {}): Sfx {
  let context: AudioContext | undefined;
  let master: GainNode | undefined;
  let enabled = options.enabled ?? true;
  let disposed = false;

  function output(): { ctx: AudioContext; out: GainNode } | undefined {
    try {
      if (!context) {
        context = (options.context ?? (() => new AudioContext()))();
        master = context.createGain();
        master.gain.value = LEVEL;
        master.connect(context.destination);
      }
      // Browsers keep a context suspended until a gesture; every play is a chance to wake it.
      if (context.state === 'suspended') void context.resume().catch(() => {});
      return { ctx: context, out: master! };
    } catch {
      return undefined;
    }
  }

  return {
    play(name, step) {
      if (!enabled || disposed) return;
      const audio = output();
      if (!audio) return;
      const { ctx, out } = audio;
      const now = ctx.currentTime + 0.005;
      for (const tone of tones(name, step)) {
        const start = now + tone.at;
        const end = start + tone.length;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = tone.wave;
        osc.frequency.setValueAtTime(hz(tone.midi), start);
        if (tone.to !== undefined) osc.frequency.exponentialRampToValueAtTime(hz(tone.to), end);
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(tone.gain, start + 0.008);
        gain.gain.exponentialRampToValueAtTime(0.0001, end);
        osc.connect(gain);
        gain.connect(out);
        osc.onended = () => {
          osc.disconnect();
          gain.disconnect();
        };
        osc.start(start);
        osc.stop(end + 0.02);
      }
    },
    setEnabled(on) {
      enabled = on;
    },
    get enabled() {
      return enabled;
    },
    dispose() {
      disposed = true;
      master?.disconnect();
      void context?.close().catch(() => {});
      context = master = undefined;
    },
  };
}

const KEY = 'differentiable.sfx';

function savedEnabled(): boolean {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}').on !== false;
  } catch {
    return true;
  }
}

/** The game's one sound-effect player; the topbar switch turns it on and off. */
export const sfx = createSfx({ enabled: typeof localStorage === 'undefined' || savedEnabled() });

export function saveSfxEnabled(on: boolean) {
  sfx.setEnabled(on);
  try {
    localStorage.setItem(KEY, JSON.stringify({ on }));
  } catch {
    // Preferences are a convenience; the game plays fine without them.
  }
}
