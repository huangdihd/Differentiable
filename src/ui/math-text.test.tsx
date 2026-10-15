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
});
