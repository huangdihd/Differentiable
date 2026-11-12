// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { SANDBOX_EXAMPLES, builtinRows, builtinTask } from './datasets';
import { buildSandboxLevel } from './level';
import { plotView, rowsPlot, summarize } from './preview';

describe('sandbox example data', () => {
  it('every example previews as a fit curve or a decision plane that matches the level', async () => {
    for (const id of SANDBOX_EXAMPLES) {
      const rows = await builtinRows(id);
      const task = builtinTask(id);
      const view = plotView(rows, task);
      expect(view).not.toBe('none');
      const level = buildSandboxLevel('p', { kind: 'builtin', id });
      expect(level.interface.inputs.x[0]).toBe(rows.x.length);
      const plot = rowsPlot(rows, task)!;
      if (view === 'fit') expect(plot.kind).toBe('regression');
      else {
        expect(plot.kind).toBe('classes');
        if (plot.kind === 'classes') expect(new Set(plot.label).size).toBe(2);
      }
    }
  });
  it('summarizes imported tables and declines to draw more than two features', () => {
    const rows = { x: [[1, 2, 3]], t: [[0, 1]] };
    expect(summarize(rows, 'classification')).toEqual({
      rows: 1,
      features: 3,
      outputs: 2,
      task: 'classification',
    });
    expect(plotView(rows, 'classification')).toBe('none');
    expect(rowsPlot(rows, 'classification')).toBeUndefined();
  });
});
