import type { Attributes } from './backend';

/** Port names for the variadic tensor ops; `count`/`parts` decides how many are live. */
export const TENSOR_OP_PORTS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
export const SPLIT_OUTPUTS = TENSOR_OP_PORTS.map((_, i) => `out${i}`);
export const MAX_PORTS = TENSOR_OP_PORTS.length;

/** Ports the editor shows for a node with these settings (others are dropped on change). */
export function tensorOpPorts(
  component: string,
  attrs: Attributes | undefined,
  side: 'inputs' | 'outputs',
): readonly string[] | undefined {
  if ((component === 'stack' || component === 'concat') && side === 'inputs')
    return TENSOR_OP_PORTS.slice(0, clampCount(attrs?.count, 2));
  if (component === 'split' && side === 'outputs')
    return SPLIT_OUTPUTS.slice(0, clampCount(attrs?.parts, 2));
  return undefined;
}
export const clampCount = (value: unknown, fallback: number) =>
  typeof value === 'number' && Number.isInteger(value)
    ? Math.min(MAX_PORTS, Math.max(1, value))
    : fallback;
