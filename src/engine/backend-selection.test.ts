import { afterEach, describe, expect, it, vi } from 'vitest';
import { CpuBackend } from './cpu';
import { createBackend } from './backend-selection';
import { BackendError, WebGpuBackend } from './webgpu';

afterEach(() => vi.restoreAllMocks());

describe('backend selection', () => {
  it('uses CPU without probing WebGPU when CPU is requested', async () => {
    const create = vi.spyOn(WebGpuBackend, 'create');

    const selection = await createBackend('cpu');

    expect(selection.kind).toBe('cpu');
    expect(selection.backend).toBeInstanceOf(CpuBackend);
    expect(create).not.toHaveBeenCalled();
  });

  it('falls back to CPU only for automatic selection', async () => {
    const failure = new Error('adapter unavailable');
    vi.spyOn(WebGpuBackend, 'create').mockRejectedValue(failure);

    const selection = await createBackend('auto');

    expect(selection.kind).toBe('cpu');
    expect(selection.backend).toBeInstanceOf(CpuBackend);
    expect(selection.fallbackReason).toBe(failure);
  });

  it('does not silently change an explicit WebGPU request to CPU', async () => {
    const failure = new BackendError('engine.gpu.noAdapter');
    vi.spyOn(WebGpuBackend, 'create').mockRejectedValue(failure);

    await expect(createBackend('webgpu')).rejects.toBe(failure);
  });

  it('returns WebGPU when it is available', async () => {
    const backend = { dispose: vi.fn() } as unknown as WebGpuBackend;
    vi.spyOn(WebGpuBackend, 'create').mockResolvedValue(backend);

    const selection = await createBackend('webgpu');

    expect(selection).toEqual({ backend, kind: 'webgpu' });
  });
});
