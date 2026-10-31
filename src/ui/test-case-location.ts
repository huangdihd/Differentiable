// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Locate the probe case represented by a flattened level-test result. */
export function testCaseLocation(
  tests: readonly { id: string }[],
  resultIndex: number,
  groups: readonly { cases: readonly { id: string }[] }[],
): { group: number; index: number } | undefined {
  const result = tests[resultIndex];
  if (!result) return undefined;

  const parse = (id: string) => {
    const match = /^shape-(\d+)\/(.*)$/.exec(id);
    return {
      group: match ? Number(match[1]) : 0,
      caseId: match ? match[2] : id,
    };
  };
  const { group, caseId } = parse(result.id);
  if (!Number.isSafeInteger(group) || group < 0) return undefined;

  // Results are flattened group by group. Count within this group rather than using the global
  // result index; fixed and generated fixtures are also allowed to share an id.
  let index = -1;
  for (let i = 0; i <= resultIndex; i++) if (parse(tests[i].id).group === group) index++;
  return groups[group]?.cases[index]?.id === caseId ? { group, index } : undefined;
}
