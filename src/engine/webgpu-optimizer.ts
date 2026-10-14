import { type Command, size } from './backend';

export function optimizerKernel(c: Command): { code: string; workgroups: [number] } | undefined {
  if (!c.op.startsWith('optimizer_')) return undefined;
  const f = (x: unknown) => `${Number(x).toExponential(9)}`;
  const a = c.attrs;
  let expression: string;
  switch (c.op) {
    case 'optimizer_sgd':
      expression = `x0[i] - ${f(a.learningRate)} * x1[i]`;
      break;
    case 'optimizer_m':
      expression = `${f(a.beta1)} * x0[i] + ${f(1 - Number(a.beta1))} * x1[i]`;
      break;
    case 'optimizer_v':
      expression = `${f(a.beta2)} * x0[i] + ${f(1 - Number(a.beta2))} * x1[i] * x1[i]`;
      break;
    case 'optimizer_adam':
      // x3 = [1 - beta1^t, 1 - beta2^t]: step-dependent, so passed as data, not baked into code.
      expression = `x0[i] - ${f(a.learningRate)} * (x1[i] / x3[0]) / (sqrt(x2[i] / x3[1]) + ${f(a.epsilon)})`;
      break;
    default:
      throw new Error('engine.op.unknown');
  }
  const code =
    c.inputs
      .map((_, i) => `@group(0) @binding(${i}) var<storage, read> x${i}: array<f32>;`)
      .join('\n') +
    `
@group(0) @binding(${c.inputs.length}) var<storage, read_write> result: array<f32>;
@group(0) @binding(${c.inputs.length + 1}) var<storage, read_write> error: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x;
 if (i < ${size(c.output.shape)}u) { result[i] = ${expression}; }
}`;
  return { code, workgroups: [Math.ceil(size(c.output.shape) / 64)] };
}
