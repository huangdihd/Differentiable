// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
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
