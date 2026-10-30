import { useState } from 'react';
import type { ShapePorts } from '../engine/graph';
import type { DatasetUse } from '../datasets';
import { corpus } from '../levels/corpus';
import { shapeText } from './Canvas';
import DataPlot, { type PlotData } from './DataPlot';
import './dataset-card.css';

type T = (key: string, params?: Readonly<Record<string, string | number>>) => string;

/** Visible stand-ins for whitespace, so a window of the rhyme reads character by character. */
const glyph = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);

/**
 * What a training level learns from: the dataset's name and origin, the shapes it arrives in,
 * and a look at the samples themselves (points, a table, or the text a window is cut from).
 */
export default function DatasetCard({
  t,
  data,
  inputs,
  plot,
}: {
  t: T;
  data: DatasetUse;
  inputs: ShapePorts;
  /** Points and labels (with predictions while training) for line, points and table data. */
  plot?: PlotData;
}) {
  const { dataset } = data;
  return (
    <div className={`dataset-card is-${dataset.modality}`}>
      <div className="dataset-head">
        <span className="dataset-kind">{t(`datasets.modality.${dataset.modality}`)}</span>
        <strong className="dataset-name">{t(`datasets.${dataset.id}.name`)}</strong>
      </div>
      <p className="dataset-source">{t(`datasets.${dataset.id}.source`)}</p>
      {(dataset.modality === 'line' || dataset.modality === 'points') && plot && (
        <PlotWithSamples t={t} plot={plot} label={t(`datasets.${dataset.id}.name`)} />
      )}
      {dataset.modality === 'table' && plot?.kind === 'classes' && (
        <SampleTable t={t} plot={plot} />
      )}
      {dataset.modality === 'text' && <TextSamples t={t} context={data.context ?? 1} />}
      <p className="dataset-shapes">
        {Object.entries(inputs)
          .map(([name, shape]) => `${name} ${shapeText(shape, t)}`)
          .join(' · ')}
      </p>
    </div>
  );
}

/** Short decimals for the table; whole numbers stay whole. */
const num = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

/**
 * The plot, what its axes and colours mean in terms of x and y, and the rows themselves:
 * pointing at a row rings its point, pointing at a point marks its row.
 */
function PlotWithSamples({ t, plot, label }: { t: T; plot: PlotData; label: string }) {
  const [hover, setHover] = useState<number>();
  const count = plot.kind === 'regression' ? plot.x.length : plot.points.length;
  const highlight = hover !== undefined && hover < count ? hover : undefined;
  return (
    <>
      <DataPlot
        data={plot}
        label={label}
        highlight={highlight}
        onHover={setHover}
        axisLabels={
          plot.kind === 'regression'
            ? { x: 'x', y: t('datasets.axis.target') }
            : { x: t('datasets.axis.column', { n: 1 }), y: t('datasets.axis.column', { n: 2 }) }
        }
      />
      <ul className="dataset-legend">
        {plot.kind === 'regression' ? (
          <>
            <li>
              <span className="legend-dot" /> {t('datasets.legend.sample')}
            </li>
            {plot.p && (
              <li>
                <span className="legend-line" /> {t('datasets.legend.fit')}
              </li>
            )}
          </>
        ) : (
          [0, 1].map((c) => (
            <li key={c}>
              <span className={`legend-dot is-class-${c}`} />{' '}
              {t('datasets.legend.class', { c, n: c + 1 })}
            </li>
          ))
        )}
      </ul>
      <SampleTable t={t} plot={plot} highlight={highlight} onHover={setHover} />
    </>
  );
}

/** One row per sample, in tensor order: its features, the answer, and the model's answer. */
function SampleTable({
  t,
  plot,
  highlight,
  onHover,
}: {
  t: T;
  plot: PlotData;
  highlight?: number;
  onHover?: (index: number | undefined) => void;
}) {
  const row = (i: number, cells: React.ReactNode) => (
    <tr
      key={i}
      className={i === highlight ? 'is-highlight' : undefined}
      onMouseEnter={onHover && (() => onHover(i))}
      onMouseLeave={onHover && (() => onHover(undefined))}
    >
      <td className="row-index">{i + 1}</td>
      {cells}
    </tr>
  );
  return (
    <div className="dataset-table-wrap">
      <table className="dataset-table">
        <thead>
          {plot.kind === 'regression' ? (
            <tr>
              <th>{t('datasets.row')}</th>
              <th>x</th>
              <th>t</th>
              {plot.p && <th>{t('datasets.prediction')}</th>}
            </tr>
          ) : (
            <tr>
              <th>{t('datasets.row')}</th>
              <th>x₁</th>
              <th>x₂</th>
              <th>{t('datasets.answer')}</th>
              {plot.predicted && <th>{t('datasets.modelAnswer')}</th>}
            </tr>
          )}
        </thead>
        <tbody>
          {plot.kind === 'regression'
            ? plot.x.map((x, i) =>
                row(
                  i,
                  <>
                    <td>{num(x)}</td>
                    <td>{num(plot.t[i])}</td>
                    {plot.p && <td>{num(plot.p[i])}</td>}
                  </>,
                ),
              )
            : plot.points.map(([a, b], i) => {
                const wrong = plot.predicted && plot.predicted[i] !== plot.label[i];
                return row(
                  i,
                  <>
                    <td>{num(a)}</td>
                    <td>{num(b)}</td>
                    <td>
                      <span className={`legend-dot is-class-${plot.label[i] % 4}`} />{' '}
                      {plot.label[i]}
                    </td>
                    {plot.predicted && (
                      <td className={wrong ? 'is-wrong' : 'is-right'}>
                        {plot.predicted[i]} {wrong ? '✗' : '✓'}
                      </td>
                    )}
                  </>,
                );
              })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * How the rhyme becomes samples. Short contexts show the first few (context → next character);
 * windows show one input row over its target row, shifted by one.
 */
function TextSamples({ t, context }: { t: T; context: number }) {
  const cells = (text: string, mark?: number) =>
    [...text].map((c, i) => (
      <span key={i} className={`char-cell${i === mark ? ' is-answer' : ''}`}>
        {glyph(c)}
      </span>
    ));
  const windows = context > 2;
  // Long windows show their end with a leading ellipsis so the strip fits without scrolling.
  const shown = Math.min(context, 10);
  const clipped = shown < context;
  return (
    <>
      <p className="dataset-note">
        {t(windows ? 'datasets.text.windows' : 'datasets.text.pairs', { context })}
      </p>
      <div className="char-samples">
        {windows ? (
          <>
            <div className="char-row">
              <span className="char-label">x</span>
              {clipped && <span className="char-more">…</span>}
              {cells(corpus.slice(context - shown, context))}
            </div>
            <div className="char-row">
              <span className="char-label">{t('datasets.answer')}</span>
              {clipped && <span className="char-more">…</span>}
              {cells(corpus.slice(context - shown + 1, context + 1), shown - 1)}
            </div>
          </>
        ) : (
          [0, 1, 2, 3].map((start) => (
            <div key={start} className="char-row">
              {cells(corpus.slice(start, start + context))}
              <span className="char-arrow">→</span>
              {cells(corpus[start + context], 0)}
            </div>
          ))
        )}
      </div>
      <details className="dataset-text">
        <summary>{t('datasets.text.full', { count: corpus.length })}</summary>
        <pre>
          <mark>{corpus.slice(0, context + 1)}</mark>
          {corpus.slice(context + 1)}
        </pre>
      </details>
    </>
  );
}
