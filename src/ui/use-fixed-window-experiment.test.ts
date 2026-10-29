import { useEffect, useRef, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { FixedWindowResponse, FixedWindowSnapshot } from '../worker/fixed-window-protocol';
import { useFixedWindowExperiment } from './use-fixed-window-experiment';
import { initialProgress } from '../engine/levels';

vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return {
    ...actual,
    useState: vi.fn(actual.useState),
    useRef: vi.fn(actual.useRef),
    useEffect: vi.fn(actual.useEffect),
  };
});
afterEach(() => {
  vi.mocked(useState).mockReset();
  vi.mocked(useRef).mockReset();
  vi.mocked(useEffect).mockReset();
  vi.unstubAllGlobals();
});

function hookHost() {
  const slots: unknown[] = [];
  const cleanups: (() => void)[] = [];
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
  vi.mocked(useRef).mockImplementation(((initial: unknown) => {
    const index = cursor++;
    if (!(index in slots)) slots[index] = { current: initial };
    return slots[index];
  }) as typeof useRef);
  vi.mocked(useEffect).mockImplementation((effect) => {
    const index = cursor++;
    if (index in slots) return;
    slots[index] = true;
    const cleanup = effect();
    if (cleanup) cleanups.push(cleanup);
  });
  return {
    render: <Result>(render: () => Result): Result => {
      cursor = 0;
      return render();
    },
    unmount: () => cleanups.forEach((cleanup) => cleanup()),
  };
}

class ExperimentWorker {
  static instances: ExperimentWorker[] = [];
  onmessage?: (event: MessageEvent<FixedWindowResponse>) => void;
  onerror?: (event: ErrorEvent) => void;
  onmessageerror?: () => void;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor() {
    ExperimentWorker.instances.push(this);
  }
  emit(data: FixedWindowResponse) {
    this.onmessage?.({ data } as MessageEvent<FixedWindowResponse>);
  }
}

describe('isolated fixed-window worker', () => {
  it('sends player graphs only when starting or retrying, and retains the worker for continuation', () => {
    ExperimentWorker.instances = [];
    vi.stubGlobal('Worker', ExperimentWorker);
    const host = hookHost();
    const progress = initialProgress([]);
    const before = JSON.stringify(progress);
    const render = () => host.render(() => useFixedWindowExperiment('rnn', progress, 'webgpu'));
    const snapshot: FixedWindowSnapshot = {
      model: 'rnn',
      parameterCount: 1683,
      sampleCount: 1495,
      step: 320,
      stepsPerRun: 320,
      seed: 4105,
      learningRate: 0.01,
      loss: 1,
      predictions: [],
    };
    render().train();
    const worker = ExperimentWorker.instances[0];
    expect(worker.postMessage).toHaveBeenLastCalledWith({
      type: 'train',
      model: 'rnn',
      backend: 'webgpu',
      playerProgress: progress,
    });
    worker.emit({ type: 'result', snapshot });
    render().train();
    expect(worker.postMessage).toHaveBeenLastCalledWith({ type: 'train', model: 'rnn' });
    worker.emit({ type: 'error', key: 'failed' });
    render().train();
    expect(ExperimentWorker.instances[1].postMessage).toHaveBeenLastCalledWith({
      type: 'train',
      model: 'rnn',
      backend: 'webgpu',
      playerProgress: progress,
    });
    expect(JSON.stringify(progress)).toBe(before);
    host.unmount();
    expect(ExperimentWorker.instances[1].terminate).toHaveBeenCalledOnce();
  });

  for (const failure of ['construction', 'post', 'runtime', 'message'] as const) {
    it(`exposes ${failure} failures without fabricated snapshots`, () => {
      ExperimentWorker.instances = [];
      const createWorker = vi.fn(function () {
        if (failure === 'construction') throw new Error('unavailable');
        const worker = new ExperimentWorker();
        if (failure === 'post')
          worker.postMessage.mockImplementation(() => {
            throw new Error('unavailable');
          });
        return worker;
      });
      vi.stubGlobal('Worker', createWorker);
      const host = hookHost();
      const render = () => host.render(() => useFixedWindowExperiment());
      render().train();
      const worker = ExperimentWorker.instances[0];
      if (failure === 'runtime')
        worker.onerror?.({ preventDefault: vi.fn() } as unknown as ErrorEvent);
      if (failure === 'message') worker.onmessageerror?.();
      expect(render().state).toEqual({
        status: 'error',
        snapshots: [],
        progress: undefined,
        error: { type: 'error', key: 'worker.execution_error' },
      });
      if (worker) expect(worker.terminate).toHaveBeenCalledOnce();
      host.unmount();
    });
  }

  it('guards duplicate runs, continues one session, restarts on error and ignores late results', () => {
    ExperimentWorker.instances = [];
    vi.stubGlobal('Worker', ExperimentWorker);
    // Progress is batched per animation frame; flush frames synchronously here.
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const host = hookHost();
    const render = () => host.render(() => useFixedWindowExperiment());
    const snapshot: FixedWindowSnapshot = {
      model: 'fixed-window',
      parameterCount: 1234,
      sampleCount: 512,
      step: 160,
      stepsPerRun: 160,
      seed: 4105,
      learningRate: 0.03,
      loss: 0.8,
      predictions: [],
    };
    let experiment = render();
    experiment.train();
    experiment.train();
    const first = ExperimentWorker.instances[0];
    expect(first.postMessage).toHaveBeenCalledOnce();
    first.emit({ type: 'progress', step: 25, totalSteps: 160, loss: 2 });
    expect(render().state.progress?.step).toBe(25);
    first.emit({ type: 'result', snapshot });
    experiment = render();
    experiment.train();
    expect(ExperimentWorker.instances).toHaveLength(1);
    expect(first.postMessage).toHaveBeenCalledTimes(2);
    const diagnostic = {
      type: 'error' as const,
      key: 'reshape_mismatch',
      node: 'flatten',
      params: { actual: [1495, 2, 8], expected: [512, 16] },
    };
    first.emit(diagnostic);
    experiment = render();
    expect(experiment.state.status).toBe('error');
    expect(experiment.state.error).toEqual(diagnostic);
    experiment.train();
    expect(render().state.snapshots).toEqual([]);
    const restarted = ExperimentWorker.instances[1];
    expect(restarted.postMessage).toHaveBeenCalledOnce();
    expect(first.terminate).toHaveBeenCalledOnce();
    first.emit({ type: 'result', snapshot: { ...snapshot, step: 320 } });
    expect(render().state.snapshots).toEqual([]);
    restarted.emit({ type: 'result', snapshot });
    render().train();
    restarted.emit({ type: 'result', snapshot: { ...snapshot, step: 320 } });
    render().train();
    expect(restarted.postMessage).toHaveBeenCalledTimes(2);
    expect(render().state.snapshots.map((run) => run.step)).toEqual([160, 320]);
    host.unmount();
    expect(first.terminate).toHaveBeenCalledOnce();
    expect(restarted.terminate).toHaveBeenCalledOnce();
  });
});
