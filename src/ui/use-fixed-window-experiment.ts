// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { useEffect, useRef, useState } from 'react';
import type { Progress } from '../engine/levels';
import type {
  CharacterComparisonModel,
  FixedWindowPrediction,
  FixedWindowRequest,
  FixedWindowResponse,
  FixedWindowSnapshot,
} from '../worker/fixed-window-protocol';

export interface FixedWindowExperimentState {
  status: 'idle' | 'training' | 'ready' | 'error';
  snapshots: readonly FixedWindowSnapshot[];
  progress?: Extract<FixedWindowResponse, { type: 'progress' }>;
  error?: Extract<FixedWindowResponse, { type: 'error' }>;
  /** Loss values as they arrive (a few steps late, never a blocking readback). */
  lossHistory?: readonly { step: number; loss: number }[];
  /** Latest mid-training look at the two sentences. */
  peek?: { step: number; predictions: readonly FixedWindowPrediction[] };
  /** True when the snapshots came from the local experiment cache, not this page's training. */
  cached?: boolean;
}

export interface FixedWindowExperimentOptions {
  /** Previously measured runs (e.g. 5-1's result reused in 5-2). */
  initialSnapshots?: readonly FixedWindowSnapshot[];
  /** Called after every completed run with all runs so far. */
  onSnapshots?: (snapshots: readonly FixedWindowSnapshot[]) => void;
}

/** Workers borrow circuit definitions only; experiment weights never enter player storage. */
export function useFixedWindowExperiment(
  model: CharacterComparisonModel = 'fixed-window',
  playerProgress?: Progress,
  backend?: 'cpu' | 'webgpu',
  options: FixedWindowExperimentOptions = {},
) {
  const onSnapshots = useRef(options.onSnapshots);
  onSnapshots.current = options.onSnapshots;
  const losses = useRef<{ step: number; loss: number }[]>([]);
  const worker = useRef<Worker | null>(null);
  const busy = useRef(false);
  type ProgressMessage = Extract<FixedWindowResponse, { type: 'progress' }>;
  const pendingProgress = useRef<ProgressMessage | undefined>(undefined);
  const frame = useRef<number | undefined>(undefined);
  const schedule = (callback: () => void) =>
    typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame(callback)
      : (setTimeout(callback, 16) as unknown as number);
  const unschedule = (id: number) =>
    typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame(id) : clearTimeout(id);
  const cancelProgress = () => {
    if (frame.current !== undefined && frame.current !== -1) unschedule(frame.current);
    frame.current = undefined;
    pendingProgress.current = undefined;
  };
  // Workers post every step; merge into one state update per animation frame. Late loss
  // reports never move the bar backwards: keep the furthest step and the newest known loss.
  const queueProgress = (data: ProgressMessage) => {
    const previous = pendingProgress.current;
    pendingProgress.current = {
      ...data,
      step: Math.max(data.step, previous?.step ?? 0),
      loss: data.loss ?? previous?.loss,
    };
    if (frame.current !== undefined) return;
    frame.current = -1;
    const id = schedule(() => {
      frame.current = undefined;
      const next = pendingProgress.current;
      pendingProgress.current = undefined;
      if (!next || !busy.current) return;
      setState((state) => {
        const before = state.progress;
        const sameRun = before && before.totalSteps === next.totalSteps;
        return {
          ...state,
          lossHistory: losses.current.slice(),
          progress: {
            ...next,
            step: sameRun ? Math.max(before.step, next.step) : next.step,
            loss: next.loss ?? (sameRun ? before.loss : undefined),
          },
        };
      });
    });
    if (frame.current === -1) frame.current = id;
  };
  const [state, setState] = useState<FixedWindowExperimentState>(() =>
    options.initialSnapshots?.length
      ? { status: 'ready', snapshots: [...options.initialSnapshots], cached: true }
      : { status: 'idle', snapshots: [] },
  );

  useEffect(
    () => () => {
      worker.current?.terminate();
      worker.current = null;
      busy.current = false;
      cancelProgress();
    },
    [],
  );

  const train = () => {
    if (busy.current || (state.status !== 'error' && state.snapshots.length >= 2)) return;
    busy.current = true;
    const restart = state.status === 'error';
    if (restart) {
      worker.current?.terminate();
      worker.current = null;
    }
    if (restart) losses.current = [];
    setState({
      status: 'training',
      snapshots: restart ? [] : state.snapshots,
      lossHistory: losses.current.slice(),
    });

    const fail = (
      error: Extract<FixedWindowResponse, { type: 'error' }> = {
        type: 'error',
        key: 'worker.execution_error',
      },
    ) => {
      worker.current?.terminate();
      worker.current = null;
      busy.current = false;
      cancelProgress();
      setState(({ lossHistory, peek: _peek, ...previous }) => ({
        ...previous,
        ...(lossHistory?.length ? { lossHistory } : {}),
        status: 'error',
        progress: undefined,
        error,
      }));
    };
    try {
      const starting = !worker.current;
      if (starting) {
        const current = new Worker(new URL('../worker/fixed-window-worker.ts', import.meta.url), {
          type: 'module',
        });
        worker.current = current;
        current.onmessage = ({ data }: MessageEvent<FixedWindowResponse>) => {
          if (worker.current !== current || !busy.current) return;
          if (data.type === 'error') {
            fail(data);
            return;
          }
          if (data.type === 'progress') {
            if (data.loss !== undefined && Number.isFinite(data.loss)) {
              const last = losses.current.at(-1);
              if (!last || data.step > last.step)
                losses.current.push({ step: data.step, loss: data.loss });
            }
            queueProgress(data);
            return;
          }
          if (data.type === 'peek') {
            setState((previous) =>
              previous.status === 'training'
                ? { ...previous, peek: { step: data.step, predictions: data.predictions } }
                : previous,
            );
            return;
          }
          cancelProgress();
          busy.current = false;
          const lossHistory = losses.current.slice();
          setState((previous) => {
            const snapshots = [...previous.snapshots, data.snapshot];
            onSnapshots.current?.(snapshots);
            return { status: 'ready', snapshots, lossHistory };
          });
        };
        current.onerror = (event) => {
          event.preventDefault();
          if (worker.current === current) fail();
        };
        current.onmessageerror = () => {
          if (worker.current === current) fail();
        };
      }
      worker.current!.postMessage({
        type: 'train',
        model,
        ...(starting && backend ? { backend } : {}),
        ...(starting && playerProgress ? { playerProgress } : {}),
        ...(starting && typeof location !== 'undefined' && /[?&]devDebug=1\b/.test(location.search)
          ? { debug: true }
          : {}),
      } satisfies FixedWindowRequest);
    } catch {
      fail();
    }
  };

  return { state, train };
}
