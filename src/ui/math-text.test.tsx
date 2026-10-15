import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MathText } from './MathText';

const html = (text: string) => renderToStaticMarkup(<MathText text={text} />);

describe('MathText', () => {
  it('leaves copy without formulas as text and line breaks', () => {
    expect(html('one\ntwo')).toBe('one<br/>two');
  });

  it('sets variables italic, functions and numbers upright', () => {
    const out = html('so $y = exp(x₁) + 2$ here');
    expect(out).toContain('<span class="math" role="math">');
    expect(out).toContain('<var>y</var>');
    expect(out).toContain('<span class="m-fn">exp</span>');
    expect(out).toContain('<var>x<sub>1</sub></var>');
    expect(out).toContain('<span class="m-num">2</span>');
  });

  it('splits an unknown letter run into a product of variables', () => {
    expect(html('$xW$')).toContain('<var>x</var><var>W</var>');
  });

  it('turns a $$ line into a displayed formula without an extra break', () => {
    const out = html('before\n$$a − b$$\nafter');
    expect(out).toMatch(/^before<span class="math-block" role="math">.*<\/span>after$/);
  });
});
