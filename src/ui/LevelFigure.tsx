import type { ReactNode } from 'react';

type T = (key: string, vars?: Record<string, string | number>) => string;

/** Causal mask: a T×T score grid before and after Softmax; future cells are blocked. */
function CausalMaskFigure({ t }: { t: T }) {
  const tokens = ['l', 'i', 't', 't'];
  const n = tokens.length;
  const cell = 32;
  const left = 44;
  const top = 34;
  const gap = 40;
  const width = left + n * cell * 2 + gap + 8;
  const height = top + n * cell + 26;
  // Equal weights over visible keys, just to show each row sums to 1.
  const weight = (row: number, col: number) => (col <= row ? 1 / (row + 1) : 0);
  const grid = (x0: number, after: boolean) =>
    tokens.flatMap((_, row) =>
      tokens.map((__, col) => {
        const future = col > row;
        const x = x0 + col * cell;
        const y = top + row * cell;
        const label = future
          ? after
            ? '0'
            : '−∞'
          : after
            ? weight(row, col)
                .toFixed(2)
                .replace(/^0/, '')
                .replace(/^1\.00$/, '1')
            : 's';
        return (
          <g key={`${after}-${row}-${col}`}>
            <rect
              x={x}
              y={y}
              width={cell - 2}
              height={cell - 2}
              rx={3}
              className={future ? 'mask-cell is-blocked' : 'mask-cell'}
            />
            <text x={x + (cell - 2) / 2} y={y + cell / 2 + 3} className="mask-text">
              {label}
            </text>
          </g>
        );
      }),
    );
  const x1 = left;
  const x2 = left + n * cell + gap;
  const axis = (x0: number) =>
    tokens.map((tok, i) => (
      <text
        key={`c${x0}-${i}`}
        x={x0 + i * cell + (cell - 2) / 2}
        y={top - 6}
        className="mask-axis"
      >
        {tok}
      </text>
    ));
  return (
    <figure className="level-figure">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t('figures.causalMask.alt')}>
        <text x={x1} y={12} className="mask-title" textAnchor="start">
          {t('figures.causalMask.before')}
        </text>
        <text x={x2} y={12} className="mask-title" textAnchor="start">
          {t('figures.causalMask.after')}
        </text>
        {axis(x1)}
        {axis(x2)}
        {tokens.map((tok, i) => (
          <text
            key={`r${i}`}
            x={left - 8}
            y={top + i * cell + cell / 2 + 3}
            className="mask-axis"
            style={{ textAnchor: 'end' }}
          >
            {tok}
          </text>
        ))}
        {grid(x1, false)}
        <text x={x1 + n * cell + gap / 2 - 2} y={top + (n * cell) / 2 + 4} className="mask-arrow">
          →
        </text>
        {grid(x2, true)}
      </svg>
      <figcaption>{t('figures.causalMask.caption')}</figcaption>
    </figure>
  );
}

/** Position embedding: character row + position row → a distinct vector per (char, position). */
function PositionFigure({ t }: { t: T }) {
  const chars = t('figures.position.chars').split('');
  const cell = 44;
  const left = 64;
  const top = 10;
  const rowH = 40;
  const width = left + chars.length * cell + 8;
  const height = top + rowH * 3 + 4;
  const rows: [string, (c: string, i: number) => string, string][] = [
    [t('figures.position.char'), (c) => c, 'mask-cell'],
    [t('figures.position.pos'), (_, i) => `p${i}`, 'mask-cell is-pos'],
    [t('figures.position.sum'), (c, i) => `${c}+p${i}`, 'mask-cell is-sum'],
  ];
  return (
    <figure className="level-figure">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t('figures.position.alt')}>
        {rows.map(([label, text, cls], r) => (
          <g key={label}>
            <text
              x={left - 8}
              y={top + r * rowH + 20}
              className="mask-axis"
              style={{ textAnchor: 'end' }}
            >
              {label}
            </text>
            {chars.map((c, i) => (
              <g key={i}>
                <rect
                  x={left + i * cell}
                  y={top + r * rowH}
                  width={cell - 4}
                  height={28}
                  rx={3}
                  className={cls}
                />
                <text
                  x={left + i * cell + (cell - 4) / 2}
                  y={top + r * rowH + 18}
                  className="mask-text"
                >
                  {text(c, i)}
                </text>
              </g>
            ))}
          </g>
        ))}
      </svg>
      <figcaption>{t('figures.position.caption')}</figcaption>
    </figure>
  );
}

const figures: Record<string, (props: { t: T }) => ReactNode> = {
  '5-2': CausalMaskFigure,
  '5-4': PositionFigure,
};

/** Optional diagram shown under a level's description. */
export function LevelFigure({ levelId, t }: { levelId: string; t: T }) {
  const Figure = figures[levelId];
  return Figure ? <Figure t={t} /> : null;
}
