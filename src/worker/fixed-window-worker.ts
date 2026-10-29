import { FixedWindowWorkerHandler } from './fixed-window-experiment';
import type { FixedWindowRequest, FixedWindowResponse } from './fixed-window-protocol';

const scope = globalThis as unknown as {
  postMessage(message: FixedWindowResponse): void;
  onmessage: ((event: MessageEvent<FixedWindowRequest>) => void) | null;
};
const handler = new FixedWindowWorkerHandler((message) => scope.postMessage(message));
scope.onmessage = (event) => {
  void handler.handle(event.data);
};
