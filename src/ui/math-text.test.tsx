// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import en from '../locales/en.json';
import zh from '../locales/zh-CN.json';
import { MathText } from './MathText';

const html = (text: string) => renderToStaticMarkup(<MathText text={text} />);

const mathCopyKey = /^levels\.[^.]+\.(description|hint\d+)$/;

function validateMathCopy(text: string) {
  let inDisplayMath = false;
  let displayMathLines = 0;

  for (const line of text.split('\n')) {
    if (line === '$$') {
      if (inDisplayMath) {
        expect(displayMathLines).toBeGreaterThan(0);
      } else {
        displayMathLines = 0;
      }
      inDisplayMath = !inDisplayMath;
      continue;
    }

    if (inDisplayMath) {
      expect(line).not.toContain('$');
      if (line.trim()) displayMathLines += 1;
      continue;
    }

    expect(line).not.toContain('$$');
    // Removing complete inline pairs must leave no unmatched delimiters.
    expect(line.replace(/\$[^$\n]+\$/g, '')).not.toContain('$');
  }

  expect(inDisplayMath).toBe(false);
}

describe('locale math markup', () => {
  for (const [locale, messages] of Object.entries({ zh, en })) {
    for (const [key, value] of Object.entries(messages)) {
      it(`${locale}: ${key}`, () => {
        if (mathCopyKey.test(key)) {
          expect(typeof value).toBe('string');
          validateMathCopy(value as string);
          const output = html(value as string);
          expect(output).not.toContain('katex-error');
          if ((value as string).includes('$$')) {
            expect(output).toContain('class="katex-display"');
          }
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
  it('renders paragraphs and preserves authored line breaks', () => {
    const output = html('one\ntwo\n\nthree');
    expect(output).toMatch(/<p>one<br\/>\s*two<\/p>/);
    expect(output).toContain('<p>three</p>');
  });

  it('renders emphasis, strong text, inline code, and fenced code', () => {
    const output = html('**strong**, *emphasis*, and `inline`\n\n```ts\nconst answer = 42;\n```');
    expect(output).toContain('<strong>strong</strong>');
    expect(output).toContain('<em>emphasis</em>');
    expect(output).toContain('<code>inline</code>');
    expect(output).toContain('<code class="language-ts">const answer = 42;');
  });

  it('renders headings, lists, block quotes, and links', () => {
    const output = html(
      '# Heading\n\n> Quote\n\n1. first\n2. second\n\n[Documentation](https://example.com)',
    );
    expect(output).toContain('<h1>Heading</h1>');
    expect(output).toContain('<blockquote>');
    expect(output).toContain('<ol>');
    expect(output).toContain('<li>first</li>');
    expect(output).toContain('<a href="https://example.com">Documentation</a>');
  });

  it('renders GFM tables, strikethrough, and task lists', () => {
    const output = html(
      '| Name | Value |\n| --- | ---: |\n| x | 1 |\n\n~~obsolete~~\n\n- [x] checked\n- [ ] pending',
    );
    expect(output).toContain('<table>');
    expect(output).toContain('<th>Name</th>');
    expect(output).toContain('<td>x</td>');
    expect(output).toContain('<del>obsolete</del>');
    expect(output).toContain('class="contains-task-list"');
    expect(output).toContain('type="checkbox"');
  });

  it('renders inline and display math with KaTeX', () => {
    const output = html(
      'Inline $y = \\frac{a_1}{b^2}$.\n\n$$\n\\operatorname{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}\n$$',
    );
    expect(output).toContain('class="katex"');
    expect(output).toContain('class="mfrac"');
    expect(output).toContain('class="katex-display"');
    expect(output).not.toContain('katex-error');
  });

  it('does not inject raw HTML', () => {
    const output = html('safe <b onclick="attack()">text</b> <script>alert(1)</script>');
    expect(output).toContain('safe');
    expect(output).not.toContain('<b');
    expect(output).not.toContain('<script');
    expect(output).not.toContain('onclick');
  });
});
