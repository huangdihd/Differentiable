import { WorkerRuntime } from './runtime';
import type { WorkerRequest, WorkerResponse } from './protocol';
// Structural typing avoids requiring DOM and WebWorker lib globals together.
const scope = globalThis as unknown as {
  postMessage(message: WorkerResponse): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};
const runtime = new WorkerRuntime((message) => scope.postMessage(message));
scope.onmessage = (event) => {
  void runtime.handle(event.data);
};
