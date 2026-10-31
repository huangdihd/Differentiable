// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { createTranslator } from '../i18n';
import LossLandscapePlot, {
  lossLandscapeInputChanges,
  lossLandscapeModel,
} from './LossLandscapePlot';

describe('loss landscape plot', () => {
  it('models squared prediction error and its minimum', () => {
    const model = lossLandscapeModel('loss-intro', { p: [5], t: [3] })!;
    expect(model.mode).toBe('loss');
    expect(model.input).toBe('p');
    expect(model.prediction).toBe(5);
    expect(model.loss).toBe(4);
    expect(model.optimum).toBe(3);
    expect(model.gradient).toBeUndefined();
    expect(model.curve.every(([x, loss]) => loss === (x - 3) ** 2)).toBe(true);
  });

  it('models one gradient-descent step over the weight', () => {
    const model = lossLandscapeModel('3-1', { w: [0], x: [2], '∂L/∂p': [-6] })!;
    expect(model.mode).toBe('descent');
    expect(model.input).toBe('w');
    expect(model.prediction).toBe(0);
    expect(model.loss).toBe(9);
    expect(model.gradient).toBe(-12);
    expect(model.next).toBeCloseTo(1.2, 10);
    expect(model.optimum).toBe(1.5);
    expect(model.domain[0]).toBeLessThanOrEqual(model.current);
    expect(model.domain[1]).toBeGreaterThanOrEqual(model.optimum!);
  });

  it('keeps the flat x=0 landscape finite without inventing a minimum', () => {
    const model = lossLandscapeModel('3-1', { w: [4], x: [0], '∂L/∂p': [-6] })!;
    expect(model.prediction).toBe(0);
    expect(model.loss).toBe(9);
    expect(model.gradient).toBe(0);
    expect(model.next).toBe(4);
    expect(model.optimum).toBeUndefined();
    expect(model.curve.flat().every(Number.isFinite)).toBe(true);
    expect(new Set(model.curve.map(([, loss]) => loss))).toEqual(new Set([9]));
    expect(lossLandscapeModel('other', { p: [1], t: [0] })).toBeUndefined();
  });

  it('updates both w and its upstream gradient while preserving the plotted target', () => {
    const inputs = { w: [0], x: [2], '∂L/∂p': [-6] };
    const model = lossLandscapeModel('3-1', inputs)!;
    const changes = lossLandscapeInputChanges(model, inputs, 1);
    expect(changes).toEqual({ w: 1, '∂L/∂p': -2 });
    const moved = lossLandscapeModel('3-1', {
      ...inputs,
      w: [changes.w],
      '∂L/∂p': [changes['∂L/∂p']],
    })!;
    expect(moved.target).toBe(model.target);
    expect(moved.prediction).toBe(2);
    expect(moved.loss).toBe(1);
  });

  it('renders localized controls, legends and matching player points in both modes', () => {
    const english = renderToStaticMarkup(
      <LossLandscapePlot
        t={createTranslator('en')}
        levelId="loss-intro"
        inputs={{ p: [5], t: [3] }}
        actual={{ shape: [], values: [4] }}
        onPick={() => {}}
      />,
    );
    expect(english).toContain('How distance from the answer becomes loss');
    expect(english).toContain('Squared-loss curve; drag to change prediction p');
    expect(english).toContain('minimum loss');
    expect(english).toContain('your loss');
    expect(english).toContain('loss-player is-ok');

    const chinese = renderToStaticMarkup(
      <LossLandscapePlot
        t={createTranslator('zh-CN')}
        levelId="3-1"
        inputs={{ w: [0], x: [2], '∂L/∂p': [-6] }}
        actual={{ shape: [], values: [1.2] }}
        onPick={() => {}}
      />,
    );
    expect(chinese).toContain('权重怎样产生预测、损失和梯度');
    expect(chinese).toContain('梯度（上坡）');
    expect(chinese).toContain('应走的一步');
    expect(chinese).toContain('你的更新位置');
    expect(chinese).toContain('loss-player-update is-ok');
    expect(chinese).toContain('∂L/∂w = -12');
  });
});
