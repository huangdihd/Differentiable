import { isValidElement, useState, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createTranslator } from '../i18n';
import { vocabulary } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowSnapshot } from '../worker/fixed-window-protocol';
import { FixedWindowLimitLesson, FixedWindowLimitView } from './FixedWindowLimitLesson';
import {
  useFixedWindowExperiment,
  type FixedWindowExperimentState,
} from './use-fixed-window-experiment';

vi.mock('./use-fixed-window-experiment', () => ({ useFixedWindowExperiment: vi.fn() }));
vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return { ...actual, useState: vi.fn(actual.useState) };
});

beforeEach(async () => {
  const actual = await vi.importActual<typeof import('react')>('react');
  vi.mocked(useState).mockImplementation(actual.useState);
  vi.mocked(useFixedWindowExperiment)
    .mockReset()
    .mockReturnValue({ state: { status: 'idle', snapshots: [] }, train: vi.fn() });
});

function hookHost() {
  const slots: unknown[] = [];
  let cursor = 0;
  vi.mocked(useState).mockImplementation(((initial: unknown) => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
    return [
      slots[index],
      (next: unknown) => {
        slots[index] = typeof next === 'function' ? next(slots[index]) : next;
      },
    ];
  }) as typeof useState);
  return <Result,>(render: () => Result): Result => {
    cursor = 0;
    return render();
  };
}

type ElementProperties = { children?: ReactNode; onClick?: () => void };
function findButton(node: ReactNode, label: string): ElementProperties | undefined {
  if (Array.isArray(node)) return node.map((child) => findButton(child, label)).find(Boolean);
  if (!isValidElement<ElementProperties>(node)) return undefined;
  if (node.type === 'button' && renderToStaticMarkup(<>{node.props.children}</>).includes(label))
    return node.props;
  return findButton(node.props.children, label);
}

// Deliberately wrong: the view must show measurements, not the intended lesson outcome.
const snapshot: FixedWindowSnapshot = {
  model: 'fixed-window',
  parameterCount: 1234,
  sampleCount: 512,
  step: 160,
  stepsPerRun: 160,
  seed: 4105,
  learningRate: 0.03,
  loss: 0.8123,
  predictions: fixedWindowExamples.map(({ history }) => ({
    history,
    input: 'th',
    prediction: 'x',
    probabilities: vocabulary.map((character) =>
      character === 'x' ? 0.75 : 0.25 / (vocabulary.length - 1),
    ),
  })),
};

for (const locale of ['en', 'zh-CN'] as const) {
  const t = createTranslator(locale);
  const text = (key: string) => t(`levels.time-window-limit.${key}`);
  const render = (state: FixedWindowExperimentState) =>
    renderToStaticMarkup(
      <FixedWindowLimitView
        code="5-1"
        t={t}
        onContinue={() => {}}
        onTrain={() => {}}
        state={state}
      />,
    );
  describe(`fixed-window lesson (${locale})`, () => {
    it('mounts only the default fixed experiment, without predictions or a premature explanation', () => {
      const html = renderToStaticMarkup(
        <FixedWindowLimitLesson code="5-1" t={t} onContinue={() => {}} />,
      );
      expect(useFixedWindowExperiment).toHaveBeenCalledExactlyOnceWith(
        'fixed-window',
        undefined,
        undefined,
      );
      expect(html).toContain(text('title'));
      for (const example of fixedWindowExamples)
        expect(html).toContain(`aria-label="${example.history}"`);
      expect(html.replaceAll('&amp;', '&')).toContain(text('run'));
      expect(html).toContain(text('demonstrationNote'));
      expect(html).not.toMatch(
        /is-correct|is-wrong|<mark>|window-limit-conclusion|levels\.time-window-limit\./,
      );
      expect(html).not.toContain(text('continue'));
    });

    it('renders measured probabilities and loss, not canned outcomes', () => {
      const html = render({ status: 'ready', snapshots: [snapshot] });
      expect(html.match(/window-limit-result is-wrong/g)).toHaveLength(2);
      expect(html).not.toContain('is-correct');
      expect(html.match(/75.00%/g)).toHaveLength(2);
      expect(html).toContain('0.812');
      expect(html).toContain(text('trainMore'));
      expect(html).not.toContain('<mark>');
      expect(html).not.toContain(text('continue'));
    });

    it('uses 320 by default and offers inspection only after both runs', () => {
      const html = render({
        status: 'ready',
        snapshots: [snapshot, { ...snapshot, step: 320, loss: 0.731 }],
      });
      expect(html).toContain(t('levels.time-window-limit.snapshot', { step: 320 }));
      expect(html).toContain('0.812');
      expect(html).toContain('0.731');
      expect(html.match(/aria-pressed="true"/g)).toHaveLength(1);
      expect(html).toContain(text('inspect'));
      expect(html).not.toContain(text('trainMore'));
      expect(html).not.toContain('window-limit-conclusion');
    });

    it('retains the labelled prior snapshot while showing actual progress', () => {
      const html = render({
        status: 'training',
        snapshots: [snapshot],
        progress: { type: 'progress', step: 180, totalSteps: 320, loss: 0.78 },
      });
      expect(html).toContain(t('levels.time-window-limit.snapshot', { step: 160 }));
      expect(html).toContain('aria-valuemax="320" aria-valuenow="180"');
      expect(html).toContain('disabled=""');
      expect(html).not.toContain(text('trainMore'));
      expect(html).not.toContain(text('inspect'));
    });

    it('shows failure without fallback predictions', () => {
      const html = render({ status: 'error', snapshots: [] });
      expect(html).toContain('role="alert"');
      expect(html).toContain(text('retry'));
      expect(html).not.toMatch(/is-correct|is-wrong/);
      expect(html).not.toContain(text('continue'));
    });

    it('runs 160 → 320 → reveal th → directly build RNN, without an RNN experiment', () => {
      const host = hookHost();
      const train = vi.fn();
      const onContinue = vi.fn();
      let state: FixedWindowExperimentState = { status: 'idle', snapshots: [] };
      vi.mocked(useFixedWindowExperiment).mockImplementation(() => ({ state, train }));
      let tree: ReactNode;
      const rerender = () => {
        tree = host(() =>
          FixedWindowLimitView(FixedWindowLimitLesson({ code: '5-1', t, onContinue }).props),
        );
        return renderToStaticMarkup(tree);
      };
      const click = (key: string) => {
        const button = findButton(tree, renderToStaticMarkup(<>{text(key)}</>));
        expect(button, key).toBeDefined();
        button!.onClick!();
      };
      rerender();
      click('run');
      expect(train).toHaveBeenCalledOnce();
      state = { status: 'ready', snapshots: [snapshot] };
      rerender();
      click('trainMore');
      expect(train).toHaveBeenCalledTimes(2);
      state = { status: 'ready', snapshots: [snapshot, { ...snapshot, step: 320 }] };
      rerender();
      click('inspect');
      const html = rerender();
      expect(html.match(/<mark>th<\/mark>/g)).toHaveLength(2);
      expect(html).toContain(text('continue'));
      expect(html).not.toContain('window-limit-comparison');
      click('continue');
      expect(onContinue).toHaveBeenCalledOnce();
      expect(train).toHaveBeenCalledTimes(2);
      expect(
        vi
          .mocked(useFixedWindowExperiment)
          .mock.calls.every(
            (arguments_) =>
              arguments_.length === 3 &&
              arguments_[0] === 'fixed-window' &&
              arguments_[1] === undefined &&
              arguments_[2] === undefined,
          ),
      ).toBe(true);
    });
  });
}
