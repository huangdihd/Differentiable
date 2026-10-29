import { useEffect, useRef, useState } from 'react';
import type { Progress } from '../engine/levels';
import type {
  CharacterComparisonModel,
  FixedWindowRequest,
  FixedWindowResponse,
  FixedWindowSnapshot,
} from '../worker/fixed-window-protocol';

export interface FixedWindowExperimentState {
  status: 'idle' | 'training' | 'ready' | 'error';
  snapshots: readonly FixedWindowSnapshot[];
  progress?: Extract<FixedWindowResponse, { type: 'progress' }>;
  error?: Extract<FixedWindowResponse, { type: 'error' }>;
}

/** Workers borrow circuit definitions only; experiment weights never enter player storage. */
export function useFixedWindowExperiment(
  model: CharacterComparisonModel = 'fixed-window',
  playerProgress?: Progress,
  backend?: 'cpu' | 'webgpu',
) {
  const worker = useRef<Worker | null>(null);
  const busy = useRef(false);
  const [state, setState] = useState<FixedWindowExperimentState>({
    status: 'idle',
    snapshots: [],
  });

  useEffect(
    () => () => {
      worker.current?.terminate();
      worker.current = null;
      busy.current = false;
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
    setState({ status: 'training', snapshots: restart ? [] : state.snapshots });

    const fail = (
      error: Extract<FixedWindowResponse, { type: 'error' }> = {
        type: 'error',
        key: 'worker.execution_error',
      },
    ) => {
      worker.current?.terminate();
      worker.current = null;
      busy.current = false;
      setState((previous) => ({ ...previous, status: 'error', progress: undefined, error }));
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
            setState((previous) => ({ ...previous, progress: data }));
            return;
          }
          busy.current = false;
          setState((previous) => ({
            status: 'ready',
            snapshots: [...previous.snapshots, data.snapshot],
          }));
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
