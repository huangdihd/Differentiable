import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { TemporalBenchmarkEvidence } from './TemporalBenchmarkEvidence';

const t = (key: string, params?: Readonly<Record<string, string | number>>) =>
  `${key}${params ? ` ${JSON.stringify(params)}` : ''}`;

describe('temporal benchmark evidence', () => {
  it('renders nothing outside the temporal route', () => {
    expect(
      renderToStaticMarkup(<TemporalBenchmarkEvidence levelId="5-1" locale="en" t={t} />),
    ).toBe('');
  });

  it('focuses the RNN row for recurrent lessons without legacy gated models', () => {
    const html = renderToStaticMarkup(
      <TemporalBenchmarkEvidence levelId="time-rnn-delay" locale="en" t={t} />,
    );
    expect(html).toContain('benchmark.model.rnn');
    expect(html).toMatch(/<tr class="is-focus"><th scope="row">benchmark\.model\.rnn<\/th>/);
    expect(html).not.toContain('benchmark.model.lstm');
    expect(html).not.toContain('benchmark.model.gru');
  });

  it('focuses RNN plus attention and shows the retrieval probe only on its lesson', () => {
    const attention = renderToStaticMarkup(
      <TemporalBenchmarkEvidence levelId="time-attention" locale="zh-CN" t={t} />,
    );
    const forecast = renderToStaticMarkup(
      <TemporalBenchmarkEvidence levelId="time-forecast" locale="zh-CN" t={t} />,
    );
    expect(attention).toMatch(
      /<tr class="is-focus"><th scope="row">benchmark\.model\.rnn-attention<\/th>/,
    );
    expect(attention).toContain('class="attention-probe"');
    expect(attention.match(/class="attention-weight"/g)).toHaveLength(4);
    expect(forecast).not.toContain('class="attention-probe"');
  });
});
