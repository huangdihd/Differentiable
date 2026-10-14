import type { OptimizerOptions } from '../engine/training';
import type { Progress } from '../engine/levels';
import type { GraphEndpoint, GraphJSON } from '../engine/graph';
import type {
  BackendPreference,
  Command,
  ReplyData,
  WorkerError,
  WorkerRequest,
  WorkerResponse,
} from './protocol';

export interface WorkerTransport {
  postMessage(message: WorkerRequest): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerResponse>) => void): void;
  removeEventListener(
    type: 'message',
    listener: (event: MessageEvent<WorkerResponse>) => void,
  ): void;
  terminate(): void;
}
export class WorkerClientError extends Error {
  constructor(readonly detail: WorkerError) {
    super(detail.key);
  }
}
/** Promise-based RPC. Increment revision on every graph/level edit. Stale promises
 * reject with worker.stale_revision; stale training events never reach subscribers.
 * start resolves on acknowledgement, not completion; subscribe for progress/result.
 */
export class WorkerClient {
  private nextId = 1;
  private revision = 0;
  private closed = false;
  private trainingId?: number;
  private pending = new Map<
    number,
    {
      revision: number;
      command: Command['type'];
      resolve: (data: any) => void;
      reject: (error: WorkerClientError) => void;
    }
  >();
  private listeners = new Set<
    (event: Extract<WorkerResponse, { type: 'progress' | 'result' }>) => void
  >();
  constructor(
    private worker: WorkerTransport = new Worker(new URL('./entry.ts', import.meta.url), {
      type: 'module',
    }),
  ) {
    worker.addEventListener('message', this.receive);
  }
  private receive = ({ data }: MessageEvent<WorkerResponse>) => {
    if (this.closed || data.revision !== this.revision) return;
    if (data.type === 'progress' || data.type === 'result') {
      if (data.requestId === this.trainingId) {
        if (data.type === 'result') this.trainingId = undefined;
        this.listeners.forEach((fn) => fn(data));
      }
      return;
    }
    const pending = this.pending.get(data.requestId);
    if (!pending || pending.revision !== data.revision) return;
    this.pending.delete(data.requestId);
    if (data.type === 'error') pending.reject(new WorkerClientError(data.error));
    else if (data.command !== pending.command)
      pending.reject(new WorkerClientError({ key: 'worker.invalid_reply', params: {} }));
    else {
      if (data.command === 'start') this.trainingId = data.requestId;
      pending.resolve(data.data);
    }
  };
  setRevision(revision: number) {
    if (!Number.isSafeInteger(revision) || revision < this.revision)
      throw new WorkerClientError({ key: 'worker.invalid_revision', params: {} });
    if (revision === this.revision) return;
    this.revision = revision;
    this.trainingId = undefined;
    this.rejectPending('worker.stale_revision');
    // Cancellation must reach the runtime even if the edited graph cannot compile.
    void this.send<'cancel'>({ type: 'cancel' }).catch(() => {});
  }
  private rejectPending(key: string) {
    this.pending.forEach((p) => p.reject(new WorkerClientError({ key, params: {} })));
    this.pending.clear();
  }
  private send<K extends Command['type']>(
    command: Extract<Command, { type: K }> | (Command & { type: K }),
  ): Promise<ReplyData[K]> {
    if (this.closed)
      return Promise.reject(new WorkerClientError({ key: 'worker.disposed', params: {} }));
    const requestId = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, {
        revision: this.revision,
        command: command.type,
        resolve,
        reject,
      });
      try {
        this.worker.postMessage({
          ...command,
          requestId,
          revision: this.revision,
        } as WorkerRequest);
      } catch {
        this.pending.delete(requestId);
        reject(new WorkerClientError({ key: 'worker.transport_error', params: {} }));
      }
    });
  }
  subscribe(listener: (event: Extract<WorkerResponse, { type: 'progress' | 'result' }>) => void) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }
  init(backend: BackendPreference = 'auto') {
    return this.send<'init'>({ type: 'init', backend });
  }
  /** Synchronization invalidates pending results and training, even without a graph edit. */
  syncProgress(progress: Progress, revision = this.revision + 1) {
    if (!Number.isSafeInteger(revision) || revision <= this.revision)
      throw new WorkerClientError({ key: 'worker.invalid_revision', params: {} });
    this.setRevision(revision);
    return this.send<'syncProgress'>({ type: 'syncProgress', progress });
  }
  compile(levelId: string, graph: GraphJSON) {
    return this.send<'compile'>({ type: 'compile', levelId, graph });
  }
  probe(
    levelId: string,
    graph: GraphJSON,
    endpoint: GraphEndpoint,
    inputs: Record<string, number[]>,
  ) {
    return this.send<'probe'>({ type: 'probe', levelId, graph, endpoint, inputs });
  }
  /** Read-only backward graph; supplying inputs also requests a numeric trace. */
  backward(levelId: string, graph: GraphJSON, inputs?: Record<string, number[]>) {
    return this.send<'backward'>({ type: 'backward', levelId, graph, inputs });
  }
  evaluate(
    levelId: string,
    graph: GraphJSON,
    learningRate?: number,
    optimizer?: OptimizerOptions['kind'],
  ) {
    return this.send<'evaluate'>({ type: 'evaluate', levelId, graph, learningRate, optimizer });
  }
  start(
    levelId: string,
    graph: GraphJSON,
    learningRate?: number,
    optimizer?: OptimizerOptions['kind'],
  ) {
    return this.send<'start'>({ type: 'start', levelId, graph, learningRate, optimizer });
  }
  trainStep(
    levelId: string,
    graph: GraphJSON,
    learningRate?: number,
    optimizer?: OptimizerOptions['kind'],
  ) {
    return this.send<'trainStep'>({ type: 'trainStep', levelId, graph, learningRate, optimizer });
  }
  trainReset() {
    return this.send<'trainReset'>({ type: 'trainReset' });
  }
  pause() {
    return this.send<'pause'>({ type: 'pause' });
  }
  resume() {
    return this.send<'resume'>({ type: 'resume' });
  }
  stop() {
    return this.send<'stop'>({ type: 'stop' });
  }
  dispose() {
    this.closed = true;
    this.rejectPending('worker.disposed');
    this.listeners.clear();
    this.worker.removeEventListener('message', this.receive);
    this.worker.terminate();
  }
}
