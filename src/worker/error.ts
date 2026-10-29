import { GraphError } from '../engine/graph';
import { BackendError } from '../engine/webgpu';
import type { WorkerError } from './protocol';

/** Preserve structured diagnostics, never exception prose or stack traces. */
export function workerError(error: unknown): WorkerError {
  if (error instanceof GraphError) {
    const { message: _message, ...params } = error.params;
    return { key: error.key, node: error.node, port: error.port, params };
  }
  if (error instanceof BackendError) return { key: error.key, params: error.params };
  const key =
    error instanceof Error && /^(engine|training|levels|worker)\.[a-z0-9_.]+$/.test(error.message)
      ? error.message
      : 'worker.execution_error';
  return { key, params: {} };
}
