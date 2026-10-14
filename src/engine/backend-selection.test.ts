import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { compileGraph, createDefaultRegistry } from './graph';
import { defaultBackend, setDefaultBackend, tensor } from './tensor';

it('captures the game backend at compilation while new tensors use the selected default', async () => {
  const previous = defaultBackend;
  const selected = new CpuBackend();
  let plan: ReturnType<typeof compileGraph> | undefined;
  try {
    setDefaultBackend(selected);
    plan = compileGraph(
      {
        version: 1,
        nodes: [
          { id: 'constant', component: 'const', attrs: { value: 3, shape: [] } },
          { id: 'out', component: 'output', attrs: { name: 'y' } },
        ],
        edges: [{ from: { node: 'constant', port: 'out' }, to: { node: 'out', port: 'x' } }],
      },
      { registry: createDefaultRegistry(), inputShapes: {} },
    );
    const value = tensor([2], []);
    expect(value.backend).toBe(selected);
    value.dispose();
    setDefaultBackend(previous);
    const outputs = plan.execute({});
    expect(outputs.y.backend).toBe(selected);
    expect(await outputs.y.read()).toEqual(new Float32Array([3]));
    plan.releaseExecution(outputs);
  } finally {
    plan?.dispose();
    setDefaultBackend(previous);
  }
  expect(selected.liveBuffers).toBe(0);
});
