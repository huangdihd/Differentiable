import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { MathText } from './MathText';
import zh from '../locales/zh-CN.json';
import en from '../locales/en.json';

const html = (text: string) => renderToStaticMarkup(<MathText text={text} />);

const mathCopyKey = /^levels\.[^.]+\.(description|hint\d+)$/;

function validateMathCopy(text: string) {
  for (const line of text.split('\n')) {
    if (line.includes('$$')) {
      expect(line).toMatch(/^\$\$[^$\n]+\$\$$/);
    } else {
      // Removing complete inline pairs must leave no unmatched delimiters.
      expect(line.replace(/\$[^$\n]+\$/g, '')).not.toContain('$');
    }
  }
}

describe('locale math markup', () => {
  for (const [locale, messages] of Object.entries({ zh, en })) {
    for (const [key, value] of Object.entries(messages)) {
      it(`${locale}: ${key}`, () => {
        if (mathCopyKey.test(key)) {
          expect(typeof value).toBe('string');
          validateMathCopy(value as string);
        } else {
          // Also covers nested plural forms, not just string-valued entries.
          expect(JSON.stringify(value)).not.toContain('$');
        }
      });
    }
  }

  it('rejects unmatched inline dollars and non-standalone display delimiters', () => {
    for (const text of ['$x', 'x$', '$x\ny$', 'before $$x$$', '$$x$$ after', '$$$$']) {
      expect(() => validateMathCopy(text)).toThrow();
    }
  });
});

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

  it('renders braced superscripts and subscripts as scripts without showing braces', () => {
    const out = html('$e^{100} + x_{row}$');
    expect(out).toContain('<var>e</var><sup><span class="m-num">100</span></sup>');
    expect(out).toContain('<var>x</var><sub><var>r</var><var>o</var><var>w</var></sub>');
    expect(out).not.toContain('^{');
    expect(out).not.toContain('_{');
  });

  it('renders an expanded two-feature neuron with numeric subscripts', () => {
    const out = html('$$p=w_{1}x_{1}+w_{2}x_{2}+b$$');
    expect(out.match(/<sub><span class="m-num">1<\/span><\/sub>/g)).toHaveLength(2);
    expect(out.match(/<sub><span class="m-num">2<\/span><\/sub>/g)).toHaveLength(2);
    expect(out).not.toContain('_');
  });

  it('renders chain-rule derivatives as stacked fractions', () => {
    const out = html('$$\\frac{∂f}{∂w}=\\frac{∂f}{∂z}·\\frac{∂z}{∂w}$$');
    expect(out).toContain('class="math-block has-frac"');
    expect(out.match(/class="m-frac"/g)).toHaveLength(3);
    expect(out.match(/class="m-op"/g)).toHaveLength(2);
    expect(out).toContain('<span class="m-frac-num"><var>∂</var><var>f</var></span>');
    expect(out).toContain('<span class="m-frac-den"><var>∂</var><var>w</var></span>');
    expect(out).not.toContain('∂f/∂w');
  });
});
