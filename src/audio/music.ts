// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { BAR_SECONDS, clampChapter, generateBar, type Note } from './sequence';

export interface Music {
  /** Call after a user gesture. Repeated calls are idempotent. */
  start(): Promise<void>;
  stop(): void;
  setVolume(v: number): void;
  setChapter(chapter: number): void;
  readonly playing: boolean;
  dispose(): void;
}

const FADE = 0.35;
const LOOK_AHEAD = 0.15;
const SEED = 0x1ab012;

export function createMusic(options: { context?: () => AudioContext } = {}): Music {
  let context: AudioContext | undefined;
  let master: GainNode | undefined;
  let input: BiquadFilterNode | undefined;
  let graph: AudioNode[] = [];
  const sources = new Map<OscillatorNode, GainNode>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let fadeTimer: ReturnType<typeof setTimeout> | undefined;
  let pending: Promise<void> | undefined;
  let disposed = false;
  let playing = false;
  let generation = 0;
  let volume = 0.3;
  let chapter = 0;
  let bar = 0;
  let origin = 0;
  let queue: Note[] = [];
  let cursor = 0;

  function clearTimers() {
    clearTimeout(timer);
    clearTimeout(fadeTimer);
    timer = fadeTimer = undefined;
  }

  function clearSources() {
    for (const [source, gain] of sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
      gain.disconnect();
    }
    sources.clear();
  }

  function disconnectGraph() {
    clearSources();
    for (const node of graph) node.disconnect();
    graph = [];
    input = master = undefined;
  }

  function setup(ctx: AudioContext) {
    input = ctx.createBiquadFilter();
    input.type = 'lowpass';
    input.frequency.value = 1700;
    input.Q.value = 0.4;
    master = ctx.createGain();
    master.gain.value = 0;
    const delay = ctx.createDelay(2);
    delay.delayTime.value = 0.64;
    const wet = ctx.createGain();
    wet.gain.value = 0.18;
    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 18;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.01;
    compressor.release.value = 0.3;
    graph = [input, master, delay, wet, compressor];
    input.connect(master);
    input.connect(delay);
    delay.connect(wet);
    wet.connect(master); // Single quiet echo, no feedback accumulation.
    master.connect(compressor);
    compressor.connect(ctx.destination);
  }

  function ramp(value: number) {
    if (!context || !master) return;
    const now = context.currentTime;
    // cancelAndHoldAtTime is missing in Firefox; pin the current value instead.
    master.gain.cancelScheduledValues(now);
    master.gain.setValueAtTime(master.gain.value, now);
    master.gain.linearRampToValueAtTime(value, now + FADE);
  }

  function sound(note: Note, time: number) {
    const ctx = context!;
    const source = ctx.createOscillator();
    const gain = ctx.createGain();
    const sustained = note.voice === 'pad' || note.voice === 'air' || note.voice === 'bass';
    source.type = note.voice === 'pluck' || note.voice === 'echo' ? 'triangle' : 'sine';
    source.frequency.value = 440 * 2 ** ((note.midi - 69) / 12);
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(note.velocity, time + (sustained ? 0.65 : 0.015));
    gain.gain.exponentialRampToValueAtTime(0.0001, time + note.duration);
    gain.gain.linearRampToValueAtTime(0, time + note.duration + 0.1);
    source.connect(gain);
    gain.connect(input!);
    sources.set(source, gain);
    source.onended = () => {
      source.disconnect();
      gain.disconnect();
      sources.delete(source);
    };
    source.start(time);
    source.stop(time + note.duration + 0.12);
  }

  function loadBar() {
    queue = generateBar(chapter, bar, SEED);
    cursor = 0;
  }

  function schedule() {
    if (!playing || !context) return;
    const now = context.currentTime;
    // A throttled tab jumps directly to the current bar. Never replay missed notes.
    if (now >= origin + BAR_SECONDS) {
      const skipped = Math.floor((now - origin) / BAR_SECONDS);
      bar += skipped;
      origin += skipped * BAR_SECONDS;
      loadBar();
    }
    while (true) {
      while (cursor < queue.length) {
        const note = queue[cursor];
        const time = origin + note.time;
        if (time > now + LOOK_AHEAD) break;
        cursor++;
        if (time >= now) sound(note, time);
      }
      if (origin + BAR_SECONDS > now + LOOK_AHEAD) break;
      origin += BAR_SECONDS;
      bar++;
      loadBar();
    }
    timer = setTimeout(schedule, 25);
  }

  async function begin(token: number) {
    try {
      if (!context) {
        const factory = options.context ?? (() => new AudioContext());
        context = factory();
      }
      await context.resume();
      if (disposed || token !== generation) return;
      if (!master) setup(context);
      playing = true;
      origin = context.currentTime + 0.04;
      loadBar();
      ramp(volume * 0.22);
      schedule();
    } catch {
      // Unsupported or blocked audio must never prevent playing the game.
      playing = false;
      clearTimers();
      disconnectGraph();
    }
  }

  return {
    start() {
      if (disposed || playing) return Promise.resolve();
      if (pending) return pending;
      clearTimers();
      clearSources();
      const token = ++generation;
      const operation = begin(token);
      pending = operation;
      void operation.then(() => {
        if (pending === operation) pending = undefined;
      });
      return operation;
    },
    stop() {
      if (disposed) return;
      generation++;
      pending = undefined;
      playing = false;
      clearTimers();
      queue = [];
      ramp(0);
      if (!context) return;
      const token = generation;
      // The sole remaining timer finishes the fade, releases nodes, then suspends.
      fadeTimer = setTimeout(
        () => {
          fadeTimer = undefined;
          if (token !== generation || disposed) return;
          disconnectGraph();
          void context?.suspend().catch(() => {});
        },
        FADE * 1000 + 20,
      );
    },
    setVolume(v) {
      if (disposed) return;
      volume = Number.isNaN(v) ? 0 : Math.max(0, Math.min(1, v));
      if (playing) ramp(volume * 0.22);
    },
    setChapter(value) {
      if (!disposed) chapter = clampChapter(value);
    },
    get playing() {
      return playing;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      generation++;
      playing = false;
      pending = undefined;
      clearTimers();
      disconnectGraph();
      queue = [];
      void context?.close().catch(() => {});
    },
  };
}
