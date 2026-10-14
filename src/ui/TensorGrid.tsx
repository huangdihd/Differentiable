import { tensorBlocks } from './tensor-values';

/** Read-only tensor view: rows follow the last axis, higher axes stack as separate blocks. */
export default function TensorGrid({
  shape,
  values,
  format,
  label,
  off,
}: {
  shape: readonly number[];
  values: readonly number[];
  format: (n: number) => string;
  label?: string;
  /** Flat indices that disagree with the expected answer. */
  off?: ReadonlySet<number>;
}) {
  let index = 0;
  return (
    <div className="tensor-view" role="group" aria-label={label}>
      {tensorBlocks(shape, values).map((rows, b) => (
        <table className="tensor-grid" key={b}>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r}>
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
