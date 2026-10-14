import { tensorBlocks } from './tensor-values';

/** Read-only tensor view: rows follow the last axis, higher axes stack as separate blocks. */
export default function TensorGrid({
  shape,
  values,
  format,
  label,
}: {
  shape: readonly number[];
  values: readonly number[];
  format: (n: number) => string;
  label?: string;
}) {
  return (
    <div className="tensor-view" role="group" aria-label={label}>
      {tensorBlocks(shape, values).map((rows, b) => (
        <table className="tensor-grid" key={b}>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r}>
                {row.map((n, c) => (
                  <td key={c} title={String(n)} className={Number.isFinite(n) ? undefined : 'err'}>
                    {format(n)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  );
}
