import type { Backend } from './backend';
import { CpuBackend } from './cpu';
import { BackendError, WebGpuBackend } from './webgpu';

/** Selection is explicit: install once before constructing a game's tensors/graphs. */
export async function createBackend(preference: 'auto' | 'cpu' | 'webgpu' = 'auto'): Promise<{
  backend: Backend;
  kind: 'cpu' | 'webgpu';
  fallbackReason?: unknown;
}> {
  if (preference === 'cpu') return { backend: new CpuBackend(), kind: 'cpu' };
  try {
    return { backend: await WebGpuBackend.create(), kind: 'webgpu' };
  } catch (error) {
    if (preference === 'webgpu') {
      if (error instanceof BackendError) throw error;
      throw new BackendError('engine.gpu.device', { message: String(error) });
    }
    return { backend: new CpuBackend(), kind: 'cpu', fallbackReason: error };
  }
}
