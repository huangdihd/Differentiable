import { tensorBlocks } from './tensor-values';

/** Read-only tensor view: rows follow the last axis, higher axes stack as separate blocks. */
export default function TensorGrid({
  shape,
  values,
  format,
  label,
  off,
  rows,
  cols,
}: {
  shape: readonly number[];
  values: readonly number[];
  format: (n: number) => string;
  label?: string;
  /** Flat indices that disagree with the expected answer. */
  off?: ReadonlySet<number>;
  /** Row and column headers for a matrix; ignored for other ranks. */
  rows?: readonly string[];
  cols?: readonly string[];
}) {
  const labelled = shape.length === 2 && (rows?.length || cols?.length);
  const rowNames = rows?.length ? rows : undefined;
  let index = 0;
  return (
    <div className="tensor-view" role="group" aria-label={label}>
      {tensorBlocks(shape, values).map((grid, b) => (
        <table className={`tensor-grid${labelled ? ' is-labelled' : ''}`} key={b}>
          {labelled && cols && (
            <thead>
              <tr>
                {rowNames && <th />}
                {cols.map((c, k) => (
                  <th key={k} scope="col">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {grid.map((row, r) => (
              <tr key={r}>
                {labelled && rowNames && <th scope="row">{rowNames[r]}</th>}
                {row.map((n, c) => {
                  const i = index++;
                  const cls = !Number.isFinite(n) ? 'err' : off?.has(i) ? 'off' : undefined;
                  return (
                    <td key={c} title={String(n)} className={cls}>
                      {format(n)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  );
}
