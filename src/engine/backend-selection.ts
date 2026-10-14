import type { Backend } from './backend';
import { CpuBackend } from './cpu';
import { WebGpuBackend } from './webgpu';

/** Selection is explicit: install once before constructing a game's tensors/graphs. */
export async function createBackend(): Promise<{
  backend: Backend;
  kind: 'cpu' | 'webgpu';
  fallbackReason?: unknown;
}> {
  try {
    return { backend: await WebGpuBackend.create(), kind: 'webgpu' };
  } catch (fallbackReason) {
    return { backend: new CpuBackend(), kind: 'cpu', fallbackReason };
  }
}
