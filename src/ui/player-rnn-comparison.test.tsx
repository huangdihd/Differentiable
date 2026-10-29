import { isValidElement, useState, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initialProgress } from '../engine/levels';
import { createTranslator } from '../i18n';
import { vocabulary } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowSnapshot } from '../worker/fixed-window-protocol';
import { PlayerRnnComparison, PlayerRnnComparisonView } from './PlayerRnnComparison';
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

type ElementProperties = {
  children?: ReactNode;
  type?: string;
  disabled?: boolean;
  onClick?: () => void;
  onChange?: (event: { target: { checked: boolean } }) => void;
};
function findElement(
  node: ReactNode,
  predicate: (type: unknown, properties: ElementProperties) => boolean,
): ElementProperties | undefined {
  if (Array.isArray(node)) return node.map((child) => findElement(child, predicate)).find(Boolean);
  if (!isValidElement<ElementProperties>(node)) return undefined;
  if (predicate(node.type, node.props)) return node.props;
  // Expand presentational function components (they hold no hooks).
  if (typeof node.type === 'function')
    return findElement((node.type as (props: unknown) => ReactNode)(node.props), predicate);
  return findElement(node.props.children, predicate);
}

const snapshot: FixedWindowSnapshot = {
  model: 'fixed-window',
  parameterCount: 1234,
  sampleCount: 1495,
  step: 320,
  stepsPerRun: 320,
  seed: 4105,
  learningRate: 0.01,
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
const recurrentSnapshot: FixedWindowSnapshot = {
  ...snapshot,
  model: 'rnn',
  parameterCount: 4321,
  loss: 0.65,
  // Intentionally wrong and different: the UI must never assume that RNN wins.
  predictions: fixedWindowExamples.map(({ history }, index) => ({
    history,
    input: history,
    prediction: index === 0 ? 'b' : 'y',
    probabilities: vocabulary.map((character) =>
      character === (index === 0 ? 'b' : 'y') ? 0.6 : 0.4 / (vocabulary.length - 1),
    ),
  })),
  resetPredictions: snapshot.predictions,
};
const fixedComplete: FixedWindowExperimentState = {
  status: 'ready',
  snapshots: [snapshot, { ...snapshot, step: 640, loss: 0.711 }],
};

function comparisonView(locale: 'en' | 'zh-CN') {
  const host = hookHost();
  const t = createTranslator(locale);
  const onTrainFixed = vi.fn();
  const onTrainRecurrent = vi.fn();
  const onContinue = vi.fn();
  const onBack = vi.fn();
  let fixedState: FixedWindowExperimentState = { status: 'idle', snapshots: [] };
  let recurrentState: FixedWindowExperimentState = { status: 'idle', snapshots: [] };
  let tree: ReactNode;
  return {
    t,
    onTrainFixed,
    onTrainRecurrent,
    onContinue,
    onBack,
    text: (key: string, params?: Readonly<Record<string, string | number>>) =>
      t(`levels.time-player-comparison.${key}`, params),
    render: () => {
      tree = host(() =>
        PlayerRnnComparisonView({
          code: '5-2',
          locale,
          t,
          onContinue,
          onBack,
          onTrainFixed,
          onTrainRecurrent,
          fixedState,
          recurrentState,
        }),
      );
      return renderToStaticMarkup(tree);
    },
    fixed: (state: FixedWindowExperimentState) => {
      fixedState = state;
    },
    recurrent: (state: FixedWindowExperimentState) => {
      recurrentState = state;
    },
    click: (label: string) => {
      const button = findElement(
        tree,
        (type, properties) =>
          type === 'button' && renderToStaticMarkup(<>{properties.children}</>).includes(label),
      );
      expect(button, label).toBeDefined();
      expect(button!.disabled).not.toBe(true);
      button!.onClick!();
    },
    resetHistory: (checked: boolean) => {
      const input = findElement(
        tree,
        (type, properties) => type === 'input' && properties.type === 'checkbox',
      );
      expect(input).toBeDefined();
      expect(input!.disabled).toBe(false);
      input!.onChange!({ target: { checked } });
    },
  };
}

for (const locale of ['en', 'zh-CN'] as const) {
  describe(`player comparison (${locale})`, () => {
    it('passes the same player progress to both hooks and clearly disclaims old weights', () => {
      const progress = initialProgress([]);
      const t = createTranslator(locale);
      const html = renderToStaticMarkup(
        <PlayerRnnComparison
          code="5-2"
          locale={locale}
          t={t}
          progress={progress}
          backend="cpu"
          onContinue={() => {}}
          onBack={() => {}}
        />,
      );
      expect(useFixedWindowExperiment).toHaveBeenNthCalledWith(1, 'fixed-window', progress, 'cpu');
      expect(useFixedWindowExperiment).toHaveBeenNthCalledWith(2, 'rnn', progress, 'cpu');
      expect(html).toContain(t('levels.time-player-comparison.title'));
      expect(html).toContain(t('levels.time-player-comparison.freshWeights'));
      expect(html).not.toContain(t('levels.time-window-limit.title'));
      expect(html).not.toMatch(
        /window-limit-stages|is-correct|is-wrong|levels\.time-player-comparison\./,
      );
      expect(html).not.toContain(t('levels.time-player-comparison.continue'));
    });

    it('requires both real 320/640 runs, matches budgets, and shows wrong answers honestly', () => {
      const view = comparisonView(locale);
      view.render();
      view.click(view.text('runFixed'));
      expect(view.onTrainFixed).toHaveBeenCalledOnce();
      expect(view.onTrainRecurrent).not.toHaveBeenCalled();
      view.fixed({ status: 'ready', snapshots: [snapshot] });
      let html = view.render();
      expect(html.match(/75.00%/g)).toHaveLength(2);
      expect(html).not.toContain(view.text('matchedSnapshot', { step: 320 }));
      expect(html).not.toContain(view.text('continue'));
      view.click(view.text('trainFixedMore'));
      expect(view.onTrainFixed).toHaveBeenCalledTimes(2);
      view.fixed(fixedComplete);
      view.render();
      view.click(view.text('runRecurrent'));
      expect(view.onTrainRecurrent).toHaveBeenCalledOnce();
      view.recurrent({
        status: 'training',
        snapshots: [],
        progress: { type: 'progress', step: 40, totalSteps: 320, loss: 1 },
      });
      html = view.render();
      expect(html).toContain('aria-valuemax="320" aria-valuenow="40"');
      expect(html).not.toContain(view.text('continue'));
      view.click(view.text('back'));
      expect(view.onBack).toHaveBeenCalledOnce();
      view.recurrent({ status: 'ready', snapshots: [recurrentSnapshot] });
      html = view.render();
      expect(html).toContain(view.text('matchedSnapshot', { step: 320 }));
      expect(html).not.toContain(view.text('matchedSnapshot', { step: 640 }));
      expect(html.match(/75.00%/g)).toHaveLength(2);
      expect(html.match(/60.00%/g)).toHaveLength(2);
      expect(html.match(/experiment-token is-wrong/g)).toHaveLength(4);
      expect(html).not.toContain('is-correct');
      expect(html).toContain('0.812');
      expect(html).toContain('0.650');
      expect(html).not.toContain('0.711');
      expect(html).toContain('1,234');
      expect(html).toContain('4,321');
      expect(html).toContain('1,495');
      expect(html).not.toContain(view.text('continue'));
      view.click(view.text('trainRecurrentMore'));
      expect(view.onTrainRecurrent).toHaveBeenCalledTimes(2);
      view.recurrent({
        status: 'training',
        snapshots: [recurrentSnapshot],
        progress: { type: 'progress', step: 400, totalSteps: 640, loss: 0.5 },
      });
      html = view.render();
      expect(html).toContain(view.text('matchedSnapshot', { step: 320 }));
      expect(html).not.toContain(view.text('continue'));
      view.recurrent({
        status: 'ready',
        snapshots: [recurrentSnapshot, { ...recurrentSnapshot, step: 640, loss: 0.511 }],
      });
      html = view.render();
      expect(html).toContain(view.text('matchedSnapshot', { step: 640 }));
      expect(html).toContain('0.711');
      expect(html).toContain('0.511');
      view.click(view.text('steps', { step: 320 }));
      html = view.render();
      expect(html).toContain(view.text('matchedSnapshot', { step: 320 }));
      expect(html.match(/aria-pressed="true"/g)).toHaveLength(1);
      expect(html).not.toContain('0.711');
      expect(html).not.toContain('0.511');
      view.click(view.text('continue'));
      expect(view.onContinue).toHaveBeenCalledOnce();
      expect(html).not.toMatch(/levels\.time-player-comparison\./);
    });

    it('switches between measured full-history and reset inference without retraining', () => {
      const view = comparisonView(locale);
      view.fixed(fixedComplete);
      view.recurrent({ status: 'ready', snapshots: [recurrentSnapshot] });
      view.render();
      view.resetHistory(true);
      let html = view.render();
      expect(html.match(/75.00%/g)).toHaveLength(4);
      expect(html).not.toContain('60.00%');
      expect(html).toContain(view.text('resetNote'));
      view.resetHistory(false);
      html = view.render();
      expect(html.match(/60.00%/g)).toHaveLength(2);
      expect(html).toContain(view.text('fullHistory'));
      expect(view.onTrainFixed).not.toHaveBeenCalled();
      expect(view.onTrainRecurrent).not.toHaveBeenCalled();
    });

    it('does not fabricate a comparison for nonmatching steps or enable next early', () => {
      const view = comparisonView(locale);
      view.fixed(fixedComplete);
      view.recurrent({
        status: 'ready',
        snapshots: [
          { ...recurrentSnapshot, step: 160 },
          { ...recurrentSnapshot, step: 480 },
        ],
      });
      const html = view.render();
      expect(html).not.toContain('60.00%');
      expect(html).not.toContain('75.00%');
      expect(html).not.toContain(view.text('continue'));
      expect(html).not.toContain(view.text('matchedSnapshot', { step: 160 }));
    });

    for (const model of ['fixed', 'recurrent'] as const) {
      it(`allows back and retries only the failed ${model} graph without fallback results`, () => {
        const view = comparisonView(locale);
        if (model === 'recurrent') view.fixed(fixedComplete);
        view[model]({
          status: 'error',
          snapshots: [],
          error: {
            type: 'error',
            key: 'reshape_mismatch',
            node: 'player_flatten',
            params: { actual: [1495, 2, 8], expected: [512, 16] },
          },
        });
        let html = view.render();
        expect(html).toContain('role="alert"');
        expect(html).toContain(view.text('failed'));
        expect(html).toContain('reshape_mismatch');
        expect(html).toContain('player_flatten');
        expect(html).toContain('[1495×2×8]');
        expect(html).toContain('[512×16]');
        expect(html).not.toMatch(
          /没有使用预制|无效的通关电路|No prebuilt model|invalid passed circuits/,
        );
        expect(html).not.toContain('60.00%');
        expect(html).not.toContain('is-correct');
        expect(html).not.toContain(view.text('continue'));
        view.click(view.text('back'));
        expect(view.onBack).toHaveBeenCalledOnce();
        expect(view.onContinue).not.toHaveBeenCalled();
        view.click(view.text('retry'));
        expect(
          model === 'fixed' ? view.onTrainFixed : view.onTrainRecurrent,
        ).toHaveBeenCalledOnce();
        expect(
          model === 'fixed' ? view.onTrainRecurrent : view.onTrainFixed,
        ).not.toHaveBeenCalled();
        view[model]({ status: 'training', snapshots: [] });
        html = view.render();
        expect(html).not.toContain(view.text('continue'));
        expect(html).not.toContain('60.00%');
      });
    }
  });
}
