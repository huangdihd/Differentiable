import { afterEach, describe, expect, it, vi } from 'vitest';
import { createMusic } from './music';
import { BAR_SECONDS, CHAPTER_MUSIC, clampChapter, generateBar } from './sequence';

class Param {
  value = 0;
  setValueAtTime = vi.fn();
  cancelScheduledValues = vi.fn();
  linearRampToValueAtTime = vi.fn();
  exponentialRampToValueAtTime = vi.fn();
}
class Node {
  gain = new Param();
  frequency = new Param();
  Q = new Param();
  delayTime = new Param();
  threshold = new Param();
  knee = new Param();
  ratio = new Param();
  attack = new Param();
  release = new Param();
  type = '';
  onended: (() => void) | null = null;
  connect = vi.fn();
  disconnect = vi.fn();
  start = vi.fn();
  stop = vi.fn();
}
class Context {
  currentTime = 0;
  destination = new Node();
  nodes: Node[] = [];
  oscillators: Node[] = [];
  resume = vi.fn(async () => {});
  suspend = vi.fn(async () => {});
  close = vi.fn(async () => {});
  createGain = () => this.node();
  createBiquadFilter = () => this.node();
  createDelay = () => this.node();
  createDynamicsCompressor = () => this.node();
  createOscillator = () => {
    const node = this.node();
    this.oscillators.push(node);
    return node;
  };
  node() {
    const node = new Node();
    this.nodes.push(node);
    return node;
  }
}
function fixture() {
  vi.useFakeTimers();
  const ctx = new Context();
  const factory = vi.fn(() => ctx as unknown as AudioContext);
  return { ctx, factory, music: createMusic({ context: factory }) };
}
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('score', () => {
  it('is reproducible, seeded, in scale and in a restrained range', () => {
    for (let chapter = 0; chapter <= 6; chapter++) {
      const { root, scale } = CHAPTER_MUSIC[chapter];
      for (let bar = 0; bar < 64; bar++) {
        const notes = generateBar(chapter, bar, 42);
        expect(notes).toEqual(generateBar(chapter, bar, 42));
        for (const note of notes) {
          expect(scale).toContain((((note.midi - root) % 12) + 12) % 12);
          expect(note.midi).toBeGreaterThanOrEqual(36);
          expect(note.midi).toBeLessThanOrEqual(96);
          expect(note.time).toBeGreaterThanOrEqual(0);
          expect(note.time).toBeLessThan(BAR_SECONDS);
        }
      }
    }
    expect(generateBar(6, 5, 1)).not.toEqual(generateBar(6, 5, 2));
  });
  it('adds voices monotonically and increases density', () => {
    let previous = new Set<string>();
    const counts: number[] = [];
    for (let chapter = 0; chapter <= 6; chapter++) {
      const notes = Array.from({ length: 32 }, (_, bar) => generateBar(chapter, bar, 42)).flat();
      const voices = new Set<string>(notes.map((note) => note.voice));
      for (const voice of previous) expect(voices.has(voice)).toBe(true);
      previous = voices;
      counts.push(notes.length);
    }
    expect(counts[0]).toBeLessThan(counts[6]);
  });
  it('clamps chapter indices including nonfinite input', () => {
    expect(clampChapter(-5)).toBe(0);
    expect(clampChapter(99)).toBe(6);
    expect(clampChapter(NaN)).toBe(0);
    expect(clampChapter(Infinity)).toBe(6);
  });
});

describe('audio lifecycle', () => {
  it('creates lazily, starts idempotently, fades then clears all timers/nodes, and restarts', async () => {
    const { music, ctx, factory } = fixture();
    expect(factory).not.toHaveBeenCalled();
    await Promise.all([music.start(), music.start()]);
    await music.start();
    expect(factory).toHaveBeenCalledTimes(1);
    expect(ctx.resume).toHaveBeenCalledTimes(1);
    expect(music.playing).toBe(true);
    music.setVolume(2);
    expect(ctx.nodes[1].gain.linearRampToValueAtTime).toHaveBeenLastCalledWith(0.22, 0.35);
    music.stop();
    expect(music.playing).toBe(false);
    expect(vi.getTimerCount()).toBe(1); // Only fade completion, not a scheduler.
    await vi.advanceTimersByTimeAsync(400);
    expect(vi.getTimerCount()).toBe(0);
    expect(ctx.suspend).toHaveBeenCalledTimes(1);
    expect(ctx.nodes.every((node) => node.disconnect.mock.calls.length > 0)).toBe(true);
    await music.start();
    expect(factory).toHaveBeenCalledTimes(1);
    music.dispose();
    expect(vi.getTimerCount()).toBe(0);
    expect(ctx.close).toHaveBeenCalledTimes(1);
    await expect(music.start()).resolves.toBeUndefined();
    expect(() => {
      music.stop();
      music.setChapter(9);
      music.setVolume(1);
      music.dispose();
    }).not.toThrow();
  });
  it('does not catch up missed bars after a long throttled interval', async () => {
    const { music, ctx } = fixture();
    await music.start();
    const before = ctx.oscillators.length;
    ctx.currentTime = 600;
    await vi.advanceTimersByTimeAsync(25);
    const newSources = ctx.oscillators.slice(before);
    expect(newSources.length).toBeLessThan(10);
    for (const source of newSources)
      expect(source.start.mock.calls[0][0]).toBeGreaterThanOrEqual(600);
    music.dispose();
  });
  it('clamps setChapter and changes the score at a bar boundary', async () => {
    const { music, ctx } = fixture();
    music.setChapter(-100);
    await music.start();
    expect(ctx.oscillators).toHaveLength(3);
    music.setChapter(100);
    const before = ctx.oscillators.length;
    ctx.currentTime = BAR_SECONDS;
    await vi.advanceTimersByTimeAsync(25);
    expect(ctx.oscillators.length - before).toBeGreaterThanOrEqual(5);
    music.dispose();
  });
  it('cancels an in-flight start on disposal', async () => {
    const { music, ctx } = fixture();
    let resolve!: () => void;
    ctx.resume.mockImplementation(() => new Promise<void>((done) => (resolve = done)));
    const start = music.start();
    music.dispose();
    resolve();
    await start;
    expect(music.playing).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
    expect(ctx.nodes).toHaveLength(0);
  });
  it('silently handles missing Web Audio and rejected resume', async () => {
    vi.stubGlobal('AudioContext', undefined);
    const music = createMusic();
    await expect(music.start()).resolves.toBeUndefined();
    expect(music.playing).toBe(false);
    music.dispose();
    const f = fixture();
    f.ctx.resume.mockRejectedValue(new Error('blocked'));
    await expect(f.music.start()).resolves.toBeUndefined();
    expect(vi.getTimerCount()).toBe(0);
    f.music.dispose();
  });
});
