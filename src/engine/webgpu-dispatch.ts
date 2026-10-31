// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Generated kernels use logical grids; WebGPU limits each physical grid dimension. */
export interface DispatchKernel {
  code: string;
  workgroups: [number, number?, number?];
}

export function dispatchGrid(
  workgroups: DispatchKernel['workgroups'],
  limit: number,
): [number, number, number] {
  const logical = [workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1];
  const count = logical.reduce((a, b) => a * b, 1);
  if (
    !Number.isSafeInteger(limit) ||
    limit < 1 ||
    logical.some((n) => !Number.isSafeInteger(n) || n < 1) ||
    !Number.isSafeInteger(count) ||
    count > 0xffffffff
  )
    throw new RangeError('Invalid dispatch grid');
  if (logical.every((n) => n <= limit)) return logical as [number, number, number];
  const x = Math.min(count, limit);
  const y = Math.min(Math.ceil(count / x), limit);
  const z = Math.ceil(count / (x * y));
  // The shader's flattened workgroup index is a u32, including padded groups.
  if (z > limit || x * y * z > 0x100000000)
    throw new RangeError('Dispatch grid exceeds addressable capacity');
  return [x, y, z];
}

/**
 * Adapt literal-workgroup-size generated WGSL, without changing kernel bodies.
 * The padding guard depends only on workgroup_id, so all lanes take the same
 * branch before any workgroupBarrier (including tiled matmul and reductions).
 */
export function limitDispatch(spec: DispatchKernel, limit: number): DispatchKernel {
  const physical = dispatchGrid(spec.workgroups, limit);
  if (spec.workgroups.every((n) => n === undefined || n <= limit)) return spec;
  const logical = [spec.workgroups[0], spec.workgroups[1] ?? 1, spec.workgroups[2] ?? 1];
  const count = logical.reduce((a, b) => a * b, 1);
  const size = /@workgroup_size\(\s*(\d+)u?\s*(?:,\s*(\d+)u?\s*)?(?:,\s*(\d+)u?\s*)?\)/.exec(
    spec.code,
  );
  if (!size) throw new Error('Dispatch remapping requires a literal workgroup size');
  const workgroupSize = [Number(size[1]), Number(size[2] ?? 1), Number(size[3] ?? 1)];
  let matched = false;
  const code = spec.code.replace(/fn\s+main\s*\(([^{}]*)\)\s*\{/, (_, parameters: string) => {
    matched = true;
    let group: string | undefined;
    let global: string | undefined;
    let local: string | undefined;
    let groups: string | undefined;
    let params = parameters.replace(
      /@builtin\(\s*(workgroup_id|global_invocation_id|local_invocation_id|num_workgroups)\s*\)\s*(\w+)\s*:\s*vec3\s*<\s*u32\s*>/g,
      (declaration: string, builtin: string, name: string) => {
        if (builtin === 'local_invocation_id') {
          local = name;
          return declaration;
        }
        if (builtin === 'workgroup_id') group = name;
        if (builtin === 'global_invocation_id') global = name;
        if (builtin === 'num_workgroups') groups = name;
        return declaration.replace(new RegExp(`\\b${name}\\b`), `dispatchPhysical_${builtin}`);
      },
    );
    const extra: string[] = [];
    if (!group) extra.push('@builtin(workgroup_id) dispatchPhysical_workgroup_id: vec3<u32>');
    if (global && !local) {
      local = 'dispatchLocal';
      extra.push('@builtin(local_invocation_id) dispatchLocal: vec3<u32>');
    }
    params = [params.trim().replace(/,\s*$/, ''), ...extra].filter(Boolean).join(', ');
    return `fn main(${params}) {
  let dispatchIndex = dispatchPhysical_workgroup_id.x + ${physical[0]}u * (dispatchPhysical_workgroup_id.y + ${physical[1]}u * dispatchPhysical_workgroup_id.z);
  if (dispatchIndex >= ${count}u) { return; }
  let dispatchLogical = vec3<u32>(dispatchIndex % ${logical[0]}u, (dispatchIndex / ${logical[0]}u) % ${logical[1]}u, dispatchIndex / ${logical[0] * logical[1]}u);
  ${group ? `let ${group} = dispatchLogical;` : ''}
  ${global ? `let ${global} = dispatchLogical * vec3<u32>(${workgroupSize.map((n) => `${n}u`).join(', ')}) + ${local};` : ''}
  ${groups ? `let ${groups} = vec3<u32>(${logical.map((n) => `${n}u`).join(', ')});` : ''}`;
  });
  if (!matched) throw new Error('Dispatch remapping requires a main entry point');
  return { code, workgroups: physical };
}
