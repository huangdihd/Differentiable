import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { TemporalBenchmarkEvidence, formatMse } from './TemporalBenchmarkEvidence';

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
    expect(html).toMatch(/<tr class="is-focus"><th scope="row">benchmark\.model\.rnn<small>/);
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
      /<tr class="is-focus"><th scope="row">benchmark\.model\.rnn-attention<small>/,
    );
    expect(attention).toContain('class="attention-probe"');
    expect(attention.match(/class="attention-weight( is-match)?"/g)).toHaveLength(4);
    expect(forecast).not.toContain('class="attention-probe"');
  });

  it('hides attention evidence before attention is taught', () => {
    for (const levelId of [
      'time-window-limit',
      'time-rnn',
      'time-rnn-window',
      'time-rnn-delay',
      'time-recall-bottleneck',
    ]) {
      const html = renderToStaticMarkup(
        <TemporalBenchmarkEvidence levelId={levelId} locale="en" t={t} />,
      );
      expect(html).not.toContain('rnn-attention');
      expect(html).not.toContain('attention-probe');
      expect(html.match(/<tr/g)).toHaveLength(3);
    }
  });

  it('formats MSE with one scientific style and labels each cell', () => {
    expect(formatMse(0.0782880187)).toBe('7.83e-2');
    expect(formatMse(0.00013941309)).toBe('1.39e-4');
    const html = renderToStaticMarkup(
      <TemporalBenchmarkEvidence levelId="time-rnn" locale="en" t={t} />,
    );
    expect(html).toContain('data-label="benchmark.delay {&quot;delay&quot;:2}"');
  });
});
