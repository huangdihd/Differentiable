/** Preserve every value in tooltips; previews alone may be truncated. */
export function fullValues(values: readonly number[]): string {
  return values.join(', ');
}

/** Read-only rows follow the final tensor dimension, without dropping any elements. */
export function tensorRows(shape: readonly number[], values: readonly number[]): string {
  const width = Math.max(1, shape.at(-1) ?? values.length);
  const rows: string[] = [];
  for (let i = 0; i < values.length; i += width) rows.push(values.slice(i, i + width).join(', '));
  return rows.join('\n');
}

/** Split values into matrices of last-axis rows; rank ≤ 2 is a single block. */
export function tensorBlocks(shape: readonly number[], values: readonly number[]): number[][][] {
  const width = Math.max(1, shape.at(-1) ?? values.length);
  const height = shape.length >= 2 ? Math.max(1, shape.at(-2)!) : 1;
  const blocks: number[][][] = [];
  for (let i = 0; i < values.length; i += width) {
    const row = values.slice(i, i + width);
    const last = blocks.at(-1);
    if (last && last.length < height) last.push(row);
    else blocks.push([row]);
  }
  return blocks;
}

export function caseInputs(inputs: Record<string, number[]>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(inputs).map(([name, values]) => [name, fullValues(values)]),
  );
}
