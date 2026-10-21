import { describe, expect, it, vi } from 'vitest';
import { createSfx, degree, tones, type SfxName } from './sfx';

const NAMES: SfxName[] = [
  'place',
  'connect',
  'disconnect',
  'remove',
  'reject',
  'pass',
  'fail',
  'win',
];

class Param {
  value = 0;
  setValueAtTime = vi.fn();
  linearRampToValueAtTime = vi.fn();
  exponentialRampToValueAtTime = vi.fn();
}
class Node {
  gain = new Param();
  frequency = new Param();
  type = '';
  onended: (() => void) | null = null;
  connect = vi.fn();
  disconnect = vi.fn();
  start = vi.fn();
  stop = vi.fn();
}
class Context {
  state = 'running';
  currentTime = 0;
  destination = new Node();
  oscillators: Node[] = [];
  resume = vi.fn(async () => {});
  close = vi.fn(async () => {});
  createGain = () => new Node();
  createOscillator = () => {
    const node = new Node();
    this.oscillators.push(node);
    return node;
  };
}

describe('sound effects', () => {
  it('are short, quiet and audible-range for every name', () => {
    for (const name of NAMES) {
      const list = tones(name, 3);
      expect(list.length).toBeGreaterThan(0);
      for (const tone of list) {
        expect(tone.at + tone.length).toBeLessThanOrEqual(1.3);
        expect(tone.gain).toBeLessThanOrEqual(0.5);
        expect(tone.midi).toBeGreaterThanOrEqual(40);
        expect(tone.midi).toBeLessThanOrEqual(96);
      }
    }
  });

  it('climbs the pentatonic scale as test cases pass', () => {
    const pitch = (step: number) => tones('pass', step)[0].midi;
    for (let i = 1; i < 10; i++) expect(pitch(i)).toBeGreaterThan(pitch(i - 1));
    expect(pitch(50)).toBe(pitch(9));
    expect(degree(5)).toBe(81);
  });

  it('plays nothing while disabled and creates no context until first played', () => {
    const ctx = new Context();
    const factory = vi.fn(() => ctx as unknown as AudioContext);
    const sfx = createSfx({ context: factory, enabled: false });
    sfx.play('connect');
    expect(factory).not.toHaveBeenCalled();
    sfx.setEnabled(true);
    sfx.play('connect');
    expect(factory).toHaveBeenCalledTimes(1);
    expect(ctx.oscillators).toHaveLength(tones('connect').length);
  });

  it('survives missing audio support', () => {
    const sfx = createSfx({
      context: () => {
        throw new Error('no audio');
      },
    });
    expect(() => sfx.play('win')).not.toThrow();
  });
});
